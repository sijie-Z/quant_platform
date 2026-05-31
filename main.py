#!/usr/bin/env python3
"""Entry point for the A-Share Multi-Factor Quant Platform.

Usage:
    python main.py run                      # Run full pipeline with default config
    python main.py run --config myconf.yaml # Run with custom config
    python main.py analyze                  # Analyze existing results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly via python main.py (pip install -e . also supported)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_platform.utils.config import load_config
from quant_platform.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared pipeline core — used by run / compare / sweep
# ---------------------------------------------------------------------------

def _resolve_config_path(config_path_arg) -> Path:
    """Resolve config path from arg, env, or default."""
    if config_path_arg is not None:
        return Path(config_path_arg)
    import os
    env_path = os.environ.get("QUANT_CONFIG")
    if env_path is not None:
        return Path(env_path)
    return Path(__file__).resolve().parent / "config" / "default.yaml"


def _load_data(config, use_tushare: bool = True, use_baostock: bool = False):
    """Load and clean data. Returns (prices, returns, benchmark, metadata, financials, turnover).

    Respects the config.data.provider setting:
    - "baostock":  Baostock free A-share data (default, no API key needed)
    - "tushare":   Tushare Pro (requires TUSHARE_TOKEN env var)
    - "postgres":  PostgreSQL stored data
    - "synthetic": Synthetic data (for testing only)
    """
    from quant_platform.data.pipeline import DataPipeline
    from quant_platform.data.providers.baostock_provider import BaostockDataProvider
    from quant_platform.data.providers.synthetic import SyntheticDataProvider
    from quant_platform.data.providers.tushare_loader import TushareProvider

    provider = None
    configured_provider = getattr(config.data, 'provider', 'synthetic')

    # Try configured provider first
    if configured_provider == "baostock" or use_baostock:
        try:
            provider = BaostockDataProvider()
            logger.info("Using Baostock real data provider (free, no API key)")
        except Exception as e:
            logger.warning("Baostock unavailable (%s)", e)

    if provider is None and (configured_provider == "tushare" or use_tushare):
        try:
            provider = TushareProvider(
                start_date=config.data.start_date,
                end_date=config.data.end_date,
            )
            logger.info("Using Tushare real data provider")
        except (RuntimeError, ImportError) as e:
            logger.warning("Tushare unavailable (%s), falling back...", e)

    if provider is None and configured_provider == "postgres":
        try:
            from quant_platform.data.providers.postgres_provider import PostgresDataProvider
            provider = PostgresDataProvider()
            logger.info("Using PostgreSQL data provider")
        except Exception as e:
            logger.warning("PostgreSQL unavailable (%s), falling back...", e)

    # Final fallback: synthetic
    if provider is None:
        from quant_platform.data.providers.synthetic import DEFAULT_EMBEDDED_ALPHA
        embedded = getattr(config.data.synthetic, 'embedded_alpha', False)
        if configured_provider != "synthetic":
            logger.warning("Configured provider '%s' unavailable, falling back to synthetic", configured_provider)
        provider = SyntheticDataProvider(
            n_stocks=config.universe.n_stocks,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            embedded_alpha=embedded,
        )

    pipeline = DataPipeline(
        provider=provider,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        exclude_st=config.universe.exclude_st,
        exclude_suspended=config.universe.exclude_suspended,
    )
    pipeline.run()

    return (
        pipeline.get_close(),
        pipeline.returns,
        pipeline.benchmark,
        pipeline.metadata,
        pipeline.financials,
        pipeline.get_turnover(),
    )


def _compute_factors(prices, returns, financials, metadata, turnover=None, config=None):
    """Compute and process all registered factors. Returns (processed_factors, ic_results, sector_map, fin_unstacked)."""
    from quant_platform.factors.evaluation import ic_summary, rank_ic
    from quant_platform.factors.fundamental import register_all as register_fundamental
    from quant_platform.factors.processing import process_factor
    from quant_platform.factors.registry import get_registry
    from quant_platform.factors.technical import register_all as register_technical

    register_technical()
    register_fundamental()
    registry = get_registry()

    # Build set of enabled factor names from config (if provided)
    enabled_factors: set[str] | None = None
    if config is not None and hasattr(config, 'factors'):
        enabled_factors = set()
        enabled_factors.update(getattr(config.factors, 'enabled_technicals', ()))
        enabled_factors.update(getattr(config.factors, 'enabled_fundamentals', ()))

    fin_unstacked = financials.unstack("asset") if financials is not None else None

    raw_factors = {}
    for factor_name in registry.list_all():
        if enabled_factors is not None and factor_name not in enabled_factors:
            continue
        cls = registry.get(factor_name)
        inst = cls()
        try:
            kwargs = {}
            if turnover is not None:
                kwargs["turnover"] = turnover
            if inst.category.value == "fundamental" and fin_unstacked is not None:
                result = inst.run(prices, fin_unstacked, **kwargs)
            else:
                result = inst.run(prices, **kwargs)
            raw_factors[result.name] = result.values
        except Exception as e:
            logger.warning("Failed to compute %s: %s", factor_name, e)

    logger.info("Computed %d factors: %s", len(raw_factors), list(raw_factors.keys()))

    sector_map = metadata.get("sector", {}) if metadata is not None else {}
    mcap = fin_unstacked.get("market_cap") if fin_unstacked is not None else None

    processed_factors = {}
    for name, factor in raw_factors.items():
        processed_factors[name] = process_factor(
            factor, sector_map=sector_map, market_cap=mcap,
        )

    ic_results = {}
    for name, factor in processed_factors.items():
        ic = rank_ic(factor, returns)
        ic_results[name] = ic_summary(ic)

    return processed_factors, ic_results, sector_map, fin_unstacked


def _run_backtest(config, signal, prices, returns, benchmark, sector_map, financials,
                  optimizer_override=None, frequency_override=None):
    """Set up and run backtest engine. Returns results dict."""
    from quant_platform.backtest.cost_model import CostModel
    from quant_platform.backtest.engine import BacktestEngine
    from quant_platform.portfolio.constraints import PortfolioConstraints

    constraints = PortfolioConstraints(
        long_only=config.portfolio.constraints.long_only,
        max_weight=config.portfolio.constraints.max_weight,
        max_sector_exposure=config.portfolio.constraints.max_sector_exposure,
        max_turnover=config.portfolio.constraints.max_turnover,
        lot_size=config.portfolio.constraints.lot_size,
    )

    cost_model = CostModel(
        commission=config.costs.commission,
        stamp_tax=config.costs.stamp_tax,
        slippage=config.costs.slippage,
        slippage_model=config.costs.slippage_model,
    )

    engine = BacktestEngine(
        initial_capital=config.backtest.initial_capital,
        rebalance_frequency=frequency_override or config.backtest.rebalance_frequency,
        cost_model=cost_model,
        constraints=constraints,
        optimizer=optimizer_override or config.portfolio.optimizer,
        benchmark=config.backtest.benchmark,
        covariance_method=config.portfolio.covariance.method,
        covariance_lookback=config.portfolio.covariance.lookback,
    )

    return engine.run(
        signal=signal, prices=prices, returns=returns,
        benchmark_returns=benchmark, sector_map=sector_map,
        financials=financials,
    )


def _generate_signal(config, processed_factors, returns, prices=None, volume=None):
    """Generate alpha signal from processed factors."""
    from quant_platform.alpha.pipeline import AlphaPipeline

    alpha_pipe = AlphaPipeline(
        method=config.alpha.method,
        lookback=config.alpha.lookback,
        min_icir=config.alpha.min_icir,
        tradability_gate=getattr(config.alpha, 'tradability_gate', False),
        min_tradability=getattr(config.alpha, 'min_tradability', 0.3),
    )
    return alpha_pipe.run(processed_factors, returns, prices=prices, volume=volume)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    """Execute full pipeline: data -> factors -> alpha -> portfolio -> backtest -> report."""
    import yaml

    from quant_platform.utils.cache import PipelineCache

    config_path = _resolve_config_path(args.config)

    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    cache_key = PipelineCache.make_config_hash(raw_config)
    cache = PipelineCache(args.cache_dir)
    use_cache = not args.force and not args.no_cache

    config = load_config(raw=raw_config)

    # Auto-save config version before running
    from quant_platform.utils.version_manager import VersionManager
    vm = VersionManager()
    desc = args.description or f"Run: alpha={config.alpha.method} optimizer={config.portfolio.optimizer}"
    version_id = vm.save(raw_config, description=desc)
    logger.info("Config auto-saved as version %s", version_id)

    logger.info("=" * 60)
    logger.info("Quant Platform: Starting full pipeline run")
    logger.info("  Config: %s  Cache key: %s  Use cache: %s",
                 config_path, cache_key, use_cache)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    logger.info("[1/6] Loading data...")
    data_tuple = None
    if use_cache:
        data_tuple = cache.load_stage("data", cache_key)

    if data_tuple is not None:
        prices, returns, benchmark, metadata, financials = data_tuple
        turnover = None
        logger.info("Data loaded from cache")
    else:
        prices, returns, benchmark, metadata, financials, turnover = _load_data(config, use_baostock=args.use_baostock)
        if use_cache:
            cache.save_stage("data", cache_key,
                             (prices, returns, benchmark, metadata, financials))

    # ------------------------------------------------------------------
    # 2. Factors
    # ------------------------------------------------------------------
    logger.info("[2/6] Computing factors...")
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config)

    for name, summary in ic_results.items():
        logger.info("  %-20s Rank IC=%.4f  ICIR=%.2f", name, summary["mean_ic"], summary["icir"])

    # ------------------------------------------------------------------
    # 3. Alpha Signal (or Screener mode)
    # ------------------------------------------------------------------
    screener_enabled = getattr(config, 'screener', None) and config.screener.enabled

    if screener_enabled:
        logger.info("[3/6] Running Factor Screener (boolean rules)...")
        from quant_platform.portfolio.screener import FactorScreener, ScreenConfig
        from quant_platform.portfolio.screener import ScreenRule

        # Build screener from config rules
        rules = []
        for r in config.screener.rules:
            rules.append(ScreenRule(
                factor=r.factor,
                operator=r.operator,
                value=r.value,
            ))
        screener_config = ScreenConfig(
            enabled=True,
            rules=rules,
            logic=config.screener.logic,
            min_stocks=config.screener.min_stocks,
            max_stocks=config.screener.max_stocks,
        )
        screener = FactorScreener(screener_config)
        qualifiers = screener.screen(processed_factors)

        if not qualifiers:
            logger.warning("Screener returned no qualifying stocks — skipping backtest")
            print("\n  SCREENER RESULT: No stocks matched the rules.\n")
            return 0

        # Build equal-weight signal from qualifying stocks
        import pandas as pd
        logger.info("Screener: %d qualifying stocks — using equal weight", len(qualifiers))
        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for date in signal.index:
            w = 1.0 / len(qualifiers)
            for asset in qualifiers:
                if asset in signal.columns:
                    signal.loc[date, asset] = w
    else:
        logger.info("[3/6] Generating alpha signal...")
        signal = _generate_signal(config, processed_factors, returns,
                                  prices=prices, volume=turnover)

    # ------------------------------------------------------------------
    # 4-5. Portfolio + Backtest
    # ------------------------------------------------------------------
    logger.info("[4/6] Portfolio optimization (integrated in backtest)...")
    logger.info("[5/6] Running backtest...")
    results = _run_backtest(config, signal, prices, returns, benchmark,
                            sector_map, fin_unstacked)

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    logger.info("[6/6] Generating report...")
    from quant_platform.reporting.dashboard import generate_dashboard

    report = generate_dashboard(
        results=results, metadata=metadata,
        output_dir=config.output.results_dir,
        save_plots=config.output.save_plots,
        plot_format=config.output.plot_format,
        ic_results=ic_results,
    )

    print(report)
    logger.info("Results saved to: %s", config.output.results_dir)
    logger.info("Pipeline complete.")
    return 0


def cmd_analyze(args) -> int:
    """Analyze existing results from output directory.

    Loads saved CSV files (daily_returns.csv, benchmark_returns.csv,
    weights_history.csv) and regenerates the full analysis dashboard.
    """
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        print(f"Error: Results directory '{results_dir}' does not exist.")
        print("Run 'python main.py run' first to generate results.")
        return 1

    returns_file = results_dir / "daily_returns.csv"
    bench_file = results_dir / "benchmark_returns.csv"
    weights_file = results_dir / "weights_history.csv"

    if not returns_file.exists():
        logger.error("daily_returns.csv not found in %s", results_dir)
        print(f"Error: 'daily_returns.csv' not found in {results_dir}")
        return 1

    import pandas as pd

    # Load strategy returns
    strategy_returns = pd.read_csv(returns_file, index_col=0, parse_dates=True).squeeze()
    strategy_returns.name = "strategy_return"
    logger.info("Loaded strategy returns: %d days", len(strategy_returns))

    # Load benchmark returns (optional)
    benchmark_returns = None
    if bench_file.exists():
        benchmark_returns = pd.read_csv(bench_file, index_col=0, parse_dates=True).squeeze()
        benchmark_returns.name = "benchmark"
        logger.info("Loaded benchmark returns: %d days", len(benchmark_returns))

    # Load weights history (optional)
    weights_history = {}
    if weights_file.exists():
        weights_df = pd.read_csv(weights_file, index_col=0, parse_dates=True)
        for date, row in weights_df.iterrows():
            weights_history[pd.Timestamp(date)] = row.dropna()
        logger.info("Loaded weights history: %d rebalance dates", len(weights_history))

    # Build results dict
    results = {
        "daily_returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "weights_history": weights_history,
    }

    # --- Run full analysis ---
    logger.info("=" * 60)
    logger.info("Analyzing existing results...")
    logger.info("=" * 60)

    from quant_platform.reporting.dashboard import generate_dashboard

    report = generate_dashboard(
        results=results,
        metadata=None,
        output_dir=str(results_dir),
        save_plots=True,
        plot_format="png",
    )

    print(report)
    logger.info("Analysis complete. Charts regenerated in: %s", results_dir)
    return 0


def cmd_compare(args) -> int:
    """Compare multiple strategy configurations side by side."""
    import pandas as pd

    optimizers = args.optimizers.split(",") if args.optimizers else [
        "equal_weight", "mean_variance", "risk_parity"
    ]
    optimizers = [o.strip() for o in optimizers]

    config = load_config(args.config)
    # Smaller universe for faster comparison
    config.universe.n_stocks = min(config.universe.n_stocks, 300)

    # Load data once, reuse across optimizers
    prices, returns, benchmark, metadata, financials, turnover = _load_data(config, use_tushare=False)
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config)
    signal = _generate_signal(config, processed_factors, returns)

    results_table = []

    for opt in optimizers:
        logger.info("=" * 50)
        logger.info("Comparing optimizer: %s", opt)
        logger.info("=" * 50)

        try:
            bt_results = _run_backtest(
                config, signal, prices, returns, benchmark,
                sector_map, fin_unstacked, optimizer_override=opt,
            )
            summary = bt_results["summary"]
            results_table.append({
                "Optimizer": opt,
                "Total Return %": f"{summary.get('total_return', 0)*100:.2f}",
                "Ann. Return %": f"{summary.get('annual_return', 0)*100:.2f}",
                "Ann. Vol %": f"{summary.get('annual_volatility', 0)*100:.2f}",
                "Sharpe": f"{summary.get('sharpe_ratio', 0):.2f}",
                "Sortino": f"{summary.get('sortino_ratio', 0):.2f}",
                "Max DD %": f"{summary.get('max_drawdown', 0)*100:.2f}",
                "Calmar": f"{summary.get('calmar_ratio', 0):.2f}",
                "Win Rate %": f"{summary.get('win_rate', 0)*100:.1f}",
                "IR": f"{summary.get('information_ratio', 0):.2f}",
            })
        except Exception as e:
            logger.error("Optimizer %s failed: %s", opt, e)
            results_table.append({"Optimizer": opt, "Error": str(e)[:80]})

    if results_table:
        df = pd.DataFrame(results_table)
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

    return 0


def cmd_sweep(args) -> int:
    """Parameter sweep: grid search over key parameters."""
    import itertools

    import pandas as pd

    config = load_config(args.config)

    optimizers = args.optimizers.split(",") if args.optimizers else [
        "equal_weight", "mean_variance", "risk_parity"
    ]
    optimizers = [o.strip() for o in optimizers]
    frequencies = args.frequencies.split(",") if args.frequencies else [
        "monthly", "weekly"
    ]
    frequencies = [f.strip() for f in frequencies]
    n_stocks_list = [int(n) for n in args.n_stocks.split(",")] if args.n_stocks else [200]

    sweep_results = []
    total = len(optimizers) * len(frequencies) * len(n_stocks_list)
    count = 0

    for opt, freq, n_stocks in itertools.product(optimizers, frequencies, n_stocks_list):
        count += 1
        logger.info("[%d/%d] Sweep: opt=%s freq=%s n=%d", count, total, opt, freq, n_stocks)

        config.portfolio.optimizer = opt
        config.backtest.rebalance_frequency = freq
        config.universe.n_stocks = n_stocks

        try:
            prices, returns, benchmark, metadata, financials, turnover = _load_data(config, use_tushare=False)
            processed_factors, _ic_results, sector_map, fin_unstacked = _compute_factors(
                prices, returns, financials, metadata, turnover, config=config)
            signal = _generate_signal(config, processed_factors, returns)
            bt_results = _run_backtest(
                config, signal, prices, returns, benchmark,
                sector_map, fin_unstacked,
                optimizer_override=opt, frequency_override=freq,
            )

            summary = bt_results["summary"]
            sweep_results.append({
                "Optimizer": opt,
                "Frequency": freq,
                "N Stocks": n_stocks,
                "Sharpe": round(summary.get("sharpe_ratio", 0), 2),
                "Ann. Ret %": round(summary.get("annual_return", 0) * 100, 1),
                "Max DD %": round(summary.get("max_drawdown", 0) * 100, 1),
                "Sortino": round(summary.get("sortino_ratio", 0), 2),
                "Win Rate %": round(summary.get("win_rate", 0) * 100, 1),
            })
        except Exception as e:
            logger.error("Sweep [%s/%s/%d] failed: %s", opt, freq, n_stocks, e)
            sweep_results.append({
                "Optimizer": opt, "Frequency": freq, "N Stocks": n_stocks,
                "Sharpe": "ERR", "Error": str(e)[:60],
            })

    if sweep_results:
        df = pd.DataFrame(sweep_results)
        if "Sharpe" in df.columns:
            df = df.sort_values("Sharpe", ascending=False,
                                key=lambda x: pd.to_numeric(x, errors="coerce"))
        print("\n" + "=" * 80)
        print("PARAMETER SWEEP RESULTS")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

    return 0


def cmd_cache(args) -> int:
    """Manage pipeline cache."""
    from quant_platform.utils.cache import PipelineCache

    cache = PipelineCache(args.cache_dir)

    if args.subcommand == "list":
        entries = cache.list_cached()
        if entries:
            print(f"Cached stages ({len(entries)}):")
            for e in entries:
                print(f"  {e}")
        else:
            print("Cache is empty.")
    elif args.subcommand == "clear":
        count = cache.clear()
        print(f"Cleared {count} cached files.")

    return 0


def cmd_web(args) -> int:
    """Start the web server (FastAPI + Vue frontend)."""
    import uvicorn

    from quant_platform.app import create_app

    app = create_app(serve_frontend=not args.no_frontend)
    mode = "API + Frontend" if not args.no_frontend else "API only"
    print("Quant Platform Web Server starting...")
    print(f"  Mode: {mode}")
    print(f"  API:  http://{args.host}:{args.port}/api/docs")
    if not args.no_frontend:
        from quant_platform.app import _DIST_DIR
        if _DIST_DIR.exists():
            print(f"  UI:   http://{args.host}:{args.port}/")
        else:
            print("  UI:   Frontend not built. Run: cd frontend && npm run build")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_ml(args) -> int:
    """ML alpha signal operations."""
    config = load_config(_resolve_config_path(args.config))

    if args.subcommand == "train":
        from quant_platform.alpha.ml_signal import MLSignalConfig, MLSignalGenerator

        prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
        processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config)

        ml_config = MLSignalConfig(
            model_type=args.model,
            n_splits=args.splits,
        )
        gen = MLSignalGenerator(config=ml_config)
        perf = gen.train(processed_factors, returns)

        print("=" * 60)
        print(f"  ML MODEL TRAINING — {args.model.upper()}")
        print("=" * 60)
        print(f"  Test IC:        {perf.test_ic:.4f}")
        print(f"  Test ICIR:      {perf.test_icir:.4f}")
        print(f"  Train IC:       {perf.train_ic:.4f}")
        print(f"  Train samples:  {perf.n_train_samples}")
        print(f"  Date:           {perf.date}")
        print("-" * 60)
        print("  Feature Importance:")
        for name, imp in sorted(perf.feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name:<20} {imp:.4f}")
        print("=" * 60)
        return 0

    elif args.subcommand == "signal":
        from quant_platform.alpha.ml_signal import MLSignalConfig, MLSignalGenerator

        prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
        processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config)

        ml_config = MLSignalConfig(model_type=args.model)
        gen = MLSignalGenerator(config=ml_config)
        signal = gen.generate(processed_factors, returns)

        # Show top/bottom 10 stocks by signal
        last_signal = signal.iloc[-1].dropna().sort_values()
        print("=" * 60)
        print(f"  ML SIGNAL — {args.model.upper()} (last date)")
        print("=" * 60)
        print("  Top 10 (most bullish):")
        for stock, val in last_signal.tail(10).items():
            print(f"    {stock:<15} {val:+.4f}")
        print("  Bottom 10 (most bearish):")
        for stock, val in last_signal.head(10).items():
            print(f"    {stock:<15} {val:+.4f}")
        print("=" * 60)
        return 0

    else:
        print("Usage: python main.py ml train [--model lightgbm|xgboost]")
        print("       python main.py ml signal [--model lightgbm|xgboost]")
        return 1


def cmd_research(args) -> int:
    """LLM research agent operations."""
    if args.subcommand == "report":
        import json
        from pathlib import Path

        from quant_platform.agent.research_agent import ResearchAgent

        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            return 1

        # Try to load backtest results
        metrics_file = results_dir / "metrics.json"
        if not metrics_file.exists():
            print("No metrics.json found. Run 'python main.py run' first.")
            return 1

        metrics = json.loads(metrics_file.read_text())
        agent = ResearchAgent(mode="keyword")

        # Generate attribution summary
        factor_contrib = metrics.get("factor_contributions", {})
        if factor_contrib:
            import pandas as pd
            returns_data = metrics.get("daily_returns", [])
            daily_returns = pd.Series(returns_data) if returns_data else None
            summary = agent.summarize_attribution(factor_contrib, daily_returns)
            print(summary)
        else:
            print("No factor contributions in results. Full metrics:")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v}")

        return 0

    else:
        print("Usage: python main.py research report [--results-dir ./results]")
        return 1


def cmd_screen(args) -> int:
    """Run factor screener: boolean-rule stock selection.

    Computes factors for the configured universe, then applies user-defined
    screen rules. Prints qualifying stocks with their factor values.
    """
    import json
    from datetime import datetime

    import pandas as pd

    from quant_platform.portfolio.screener import (
        FactorScreener,
        ScreenRule,
    )

    config = load_config(_resolve_config_path(args.config))

    # Parse rules from CLI arg or use config defaults
    rules: list[ScreenRule] = []
    if args.rules:
        for rule_str in args.rules.split(","):
            rule_str = rule_str.strip()
            if not rule_str:
                continue
            # Format: factor_name=operator:value  e.g. pe_ratio=lt:30
            if "=" in rule_str:
                factor_expr = rule_str.split("=", 1)
                factor_name = factor_expr[0].strip()
                rest = factor_expr[1].strip()
                if ":" in rest:
                    op, val_str = rest.split(":", 1)
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
                    rules.append(ScreenRule(factor=factor_name, operator=op, value=val))
                else:
                    logger.warning("Invalid rule format (expected op:val): %s", rule_str)
            else:
                logger.warning("Invalid rule format (expected factor=op:val): %s", rule_str)
    else:
        # Use config rules
        for r in config.screener.rules:
            rules.append(ScreenRule(factor=r.factor, operator=r.operator, value=r.value))

    if not rules:
        print("No screen rules defined. Use --rules or configure screener.rules in config.")
        return 1

    # Load data & compute factors
    print("Loading data...")
    prices, returns, benchmark, metadata, financials, turnover = _load_data(
        config, use_baostock=args.use_baostock
    )
    print(f"  Data: {len(prices)} days, {len(prices.columns)} assets")

    print("Computing factors...")
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config,
    )

    # Run screener
    screener = FactorScreener({
        "enabled": True,
        "rules": [{"factor": r.factor, "operator": r.operator, "value": r.value}
                  for r in rules],
        "logic": args.logic or config.screener.logic,
        "min_stocks": args.min_stocks or config.screener.min_stocks,
        "max_stocks": args.max_stocks or config.screener.max_stocks,
    })
    qualifiers = screener.screen(processed_factors, rules)

    # Build output table
    if qualifiers:
        # Get latest factor values for qualifying stocks
        last_date = next(iter(processed_factors.values())).index[-1]
        print(f"\n{'='*80}")
        print(f"  SCREENER RESULTS — {last_date}")
        print(f"{'='*80}")
        print(f"  Rules: {', '.join(str(r.factor) + ' ' + r.operator + ' ' + str(r.value) for r in rules)}")
        print(f"  Logic: {args.logic or config.screener.logic}")
        print(f"  Passing: {len(qualifiers)} / {len(next(iter(processed_factors.values())).columns)} stocks")
        print(f"{'='*80}\n")

        # Build detail DataFrame
        detail_rows = []
        for asset in qualifiers:
            row = {"code": asset}
            for r in rules:
                if r.factor in processed_factors:
                    df = processed_factors[r.factor]
                    if last_date in df.index:
                        row[r.factor] = round(df.loc[last_date, asset], 4)
            detail_rows.append(row)

        df_out = pd.DataFrame(detail_rows)

        # Print table
        pd.set_option("display.max_rows", 120)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 20)
        print(df_out.to_string(index=False))

        # Save to CSV if requested
        if args.output:
            df_out.to_csv(args.output, index=False)
            print(f"\n  Saved to: {args.output}")

        # Print summary stats
        print(f"\n  {'─'*60}")
        print(f"  Total qualifying stocks: {len(qualifiers)}")
        for r in rules:
            if r.factor in processed_factors:
                df = processed_factors[r.factor]
                if last_date in df.index:
                    vals = df.loc[last_date, qualifiers].dropna()
                    print(f"  {r.factor}: mean={vals.mean():.4f}  "
                          f"min={vals.min():.4f}  max={vals.max():.4f}")
        print(f"  {'─'*60}\n")
    else:
        print("\n  No stocks matched the screen rules.\n")
        return 1

    return 0


def cmd_execute(args) -> int:
    """Execute full pipeline: factors -> signal -> orders -> fills -> positions."""
    config = load_config(_resolve_config_path(args.config))

    print("Loading data...")
    prices, returns, benchmark, metadata, financials, turnover = _load_data(
        config, use_baostock=args.use_baostock
    )
    print(f"  Data: {len(prices)} days, {len(prices.columns)} assets")

    print("Computing factors...")
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config,
    )

    print("Generating signal...")
    signal = _generate_signal(config, processed_factors, returns, prices=prices, volume=turnover)

    print("Setting up execution engine...")
    from quant_platform.execution.engine import ExecutionEngine, OrderSide
    from quant_platform.strategy.multi_strategy import MultiStrategyManager, StrategyConfig
    from quant_platform.strategy.portfolio_orchestrator import PortfolioOrchestrator

    ms = MultiStrategyManager(total_capital=1_000_000)
    strat_id = ms.add_strategy(StrategyConfig(
        name="test_strat", optimizer=config.portfolio.optimizer,
        allocation_pct=1.0, is_active=True,
    ))
    orchestrator = PortfolioOrchestrator(ms)

    # Get latest prices
    last_prices = prices.iloc[-1].to_dict()
    orchestrator._last_prices.update(last_prices)

    last_date = str(prices.index[-1])[:10]
    print(f"Processing signal for {last_date}...")
    orchestrator.on_signal(last_date, signal, strategy_id=strat_id)

    print("Executing rebalance...")
    orders = orchestrator.rebalance()
    print(f"  Created {len(orders)} orders")

    print("Processing fills...")
    orchestrator.process_fills(last_prices)

    summary = orchestrator.portfolio_summary()
    print()
    print("=" * 70)
    print("  EXECUTION RESULT")
    print("=" * 70)
    print(f"  Date:          {last_date}")
    print(f"  Positions:     {summary['n_positions']}")
    print(f"  Cash:          {summary['cash_available']:,.2f}")
    print(f"  Position Val:  {summary['positions_value']:,.2f}")
    print(f"  Unrealized PnL:{summary['unrealized_pnl']:+,.2f}")
    print(f"  Realized PnL:  {summary['realized_pnl']:+,.2f}")
    print(f"  Total PnL:     {summary['total_pnl']:+,.2f}")
    print(f"  Alerts:        {summary['alerts']}")
    print("-" * 70)
    for p in summary['positions'][:10]:
        print(f"  {p['ticker']:<8} {p['quantity']:>5} @ {p['avg_cost']:<8.2f} = {p['market_value']:>10.2f}  PnL:{p['unrealized_pnl']:>+8.2f}")
    if len(summary['positions']) > 10:
        print(f"  ... and {len(summary['positions']) - 10} more")
    print("=" * 70)
    return 0

def cmd_config(args) -> int:
    """Manage configuration versions: list, show, diff, rollback."""
    from quant_platform.utils.version_manager import VersionManager

    vm = VersionManager()

    if args.subcommand == "list":
        versions = vm.list()
        if not versions:
            print("No config versions saved yet.")
            print("  Run 'python main.py run' to auto-save the first version.")
            return 0

        print(f"{'ID':<6} {'Timestamp':<22} {'Description'}")
        print("-" * 80)
        for v in versions:
            desc = (v.description[:60] + "..") if len(v.description) > 62 else v.description
            print(f"{v.id:<6} {v.timestamp:<22} {desc}")
        print(f"\n{len(versions)} versions total.")

    elif args.subcommand == "show":
        try:
            config = vm.show(args.version)
            import yaml
            print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

    elif args.subcommand == "diff":
        try:
            diff_text = vm.diff(args.v1, args.v2)
            if diff_text:
                print(diff_text)
            else:
                print("Configs are identical.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

    elif args.subcommand == "rollback":
        target = args.target
        try:
            vm.rollback(args.version, target_path=target)
            print(f"Config rolled back to {args.version}.")
            if target:
                print(f"  Target: {target}")
            else:
                print("  Target: config/default.yaml")
            print("  Run 'python main.py run' to verify.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

    elif args.subcommand == "delete":
        confirm = args.force or input(f"Delete {args.version}? [y/N] ").lower() == "y"
        if confirm:
            vm.delete(args.version)
            print(f"Deleted {args.version}.")
        else:
            print("Cancelled.")

    return 0


def cmd_check_lookahead(args) -> int:
    """Run lookahead bias detection on the factor pipeline."""
    config = load_config(_resolve_config_path(args.config))
    from quant_platform.risk.lookahead_detector import LookaheadDetector

    print("Loading data...")
    prices, returns, benchmark, metadata, financials, turnover = _load_data(
        config, use_baostock=args.use_baostock
    )

    detector = LookaheadDetector(
        threshold=args.threshold,
        max_check_dates=args.max_dates,
    )
    result = detector.detect(prices, returns, financials, metadata, config)
    detector.print_report(result)

    suggestions = detector.suggest_fixes(result)
    if suggestions:
        print("Suggested fixes:")
        for s in suggestions:
            print(f"  - {s}")
    return 1 if result["has_bias"] else 0


def cmd_profile(args) -> int:
    """Profile pipeline performance — shows time per stage."""
    import time

    config = load_config(_resolve_config_path(args.config))

    timings = {}
    total_start = time.perf_counter()

    # Stage 1: Data
    t0 = time.perf_counter()
    prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
    timings["1_data"] = time.perf_counter() - t0

    # Stage 2: Factors
    t0 = time.perf_counter()
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config)
    timings["2_factors"] = time.perf_counter() - t0

    # Stage 3: Alpha
    t0 = time.perf_counter()
    signal = _generate_signal(config, processed_factors, returns)
    timings["3_alpha"] = time.perf_counter() - t0

    # Stage 4: Backtest
    t0 = time.perf_counter()
    _run_backtest(config, signal, prices, returns, benchmark, sector_map, fin_unstacked)
    timings["4_backtest"] = time.perf_counter() - t0

    total = time.perf_counter() - total_start

    print("=" * 60)
    print("  PIPELINE PERFORMANCE PROFILE")
    print("=" * 60)
    for stage, duration in timings.items():
        pct = duration / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {stage:<15} {duration:>6.2f}s  {pct:>5.1f}%  {bar}")
    print("-" * 60)
    print(f"  {'TOTAL':<15} {total:>6.2f}s")
    print("=" * 60)
    return 0


def cmd_trade(args) -> int:
    """Execute live trading via LiveRunner (Paper or QMT)."""
    from quant_platform.trading.live_runner import LiveRunner

    broker_type = args.broker
    universe = [c.strip() for c in args.universe.split(",")] if args.universe else [
        "600519", "000858", "000001", "002001", "300750",
    ]

    print("=" * 60)
    print("  LIVE TRADING RUNNER")
    print("=" * 60)
    print(f"  Broker:    {broker_type}")
    print(f"  Universe:  {len(universe)} stocks")
    print(f"  Days:      {args.days}")
    print(f"  Capital:   {args.cash:,.0f} CNY")
    print(f"  DualTrack: {not args.no_dual_track}")
    print("=" * 60)

    runner = LiveRunner(
        broker_type=broker_type,
        initial_cash=args.cash,
        dual_track=not args.no_dual_track,
    )
    runner.set_universe(universe)

    report = runner.run(days=args.days, seed=args.seed)
    d = report.to_dict()

    print(f"\nSession: {d['session_id']}")
    print(f"  Broker:         {d.get('broker_type', broker_type)}")
    print(f"  Days traded:    {d['days_traded']}")
    print(f"  Total orders:   {d['total_orders']}")
    print(f"  Total fills:    {d['total_fills']}")
    print(f"  Initial:        {d['initial_capital']:,.0f} CNY")
    print(f"  Final:          {d['final_value']:,.0f} CNY")
    print(f"  Return:         {d['total_return_pct']:.2f}%")
    print(f"  Annualized:     {d['annualized_return_pct']:.2f}%")
    print(f"  Sharpe:         {d['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:   {d['max_drawdown_pct']:.2f}%")
    print(f"  Avg Volume:     {d['avg_daily_volume']:,.0f} CNY")
    return 0


def cmd_walkforward(args) -> int:
    """Walk-forward validation: the gold standard for overfitting control.

    Each fold recomputes factor weights using only train-period data.
    This eliminates look-ahead from IC/ICIR weighting.
    """
    import pandas as pd

    config = load_config(_resolve_config_path(args.config))

    print("=" * 70)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"  Method:    {args.method}")
    print(f"  Folds:     {args.folds}")
    print(f"  Train:     {args.train_days} days")
    print(f"  Test:      {args.test_days} days")
    print("=" * 70)

    # Load data
    prices, returns, benchmark, metadata, financials, turnover = _load_data(
        config, use_baostock=args.use_baostock
    )
    print(f"  Data:      {len(prices)} days, {len(prices.columns)} assets")

    # Compute factors (full period, for initial signal)
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config
    )

    # Generate full-period signal (used as reference)
    signal = _generate_signal(config, processed_factors, returns)

    # Run walk-forward with per-fold signal recomputation
    from quant_platform.backtest.walkforward import WalkForwardValidator

    validator = WalkForwardValidator(
        train_period=args.train_days,
        test_period=args.test_days,
        step_size=args.test_days,
        mode=args.method,
    )

    engine_kwargs = {
        "initial_capital": config.backtest.initial_capital,
        "rebalance_frequency": config.backtest.rebalance_frequency,
        "benchmark": config.backtest.benchmark,
    }

    results = validator.run(
        signal=signal,
        prices=prices,
        returns=returns,
        benchmark_returns=benchmark,
        sector_map=sector_map,
        financials=fin_unstacked,
        engine_kwargs=engine_kwargs,
        factors=processed_factors,
        alpha_kwargs={
            "method": config.alpha.method,
            "lookback": config.alpha.lookback,
            "min_icir": config.alpha.min_icir,
        },
    )

    # Print results
    fold_metrics = results.get("fold_metrics", [])
    print("\n" + "=" * 70)
    print("  WALK-FORWARD RESULTS (OOS)")
    print("=" * 70)

    for i, m in enumerate(fold_metrics):
        print(f"\n  Fold {i+1}:")
        print(f"    OOS Sharpe:  {m.get('sharpe_ratio', 0):.3f}")
        print(f"    Ann. Ret:    {m.get('annual_return', 0)*100:.2f}%")
        print(f"    Max DD:      {m.get('max_drawdown', 0)*100:.2f}%")
        print(f"    Sortino:     {m.get('sortino_ratio', 0):.3f}")

    agg = results.get("aggregate_metrics", {})
    print("\n" + "-" * 70)
    print(f"  OOS Mean Sharpe:    {agg.get('sharpe_ratio', 0):.3f}")
    print(f"  Info Ratio:         {agg.get('information_ratio', 0):.3f}")
    print(f"  Annual Return:      {agg.get('annual_return', 0)*100:.2f}%")
    print(f"  Max Drawdown:       {agg.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Folds:              {len(fold_metrics)}")
    print("=" * 70)

    sharpe = agg.get('sharpe_ratio', 0)
    if sharpe < 0.5:
        print("\n  WARNING: Low OOS Sharpe — likely overfitting.")
        print("  Try: fewer factors, longer training window, equal-weight combination.")
    elif sharpe < 1.0:
        print("\n  NOTE: Moderate OOS Sharpe. Promising but needs more validation.")
    else:
        print("\n  Good OOS Sharpe. Consider forward testing (paper trading).")

    return 0


def _resolve_qmt_kwargs(config) -> dict:
    """Extract QMT kwargs from config, resolving password from env."""
    import os
    from typing import Any
    kwargs: dict[str, Any] = {}
    if hasattr(config, "execution") and hasattr(config.execution, "qmt"):
        qmt = config.execution.qmt
        kwargs["account"] = qmt.account
        kwargs["server"] = qmt.server
        kwargs["mode"] = qmt.mode
        kwargs["data_server"] = qmt.data_server or qmt.server
        password = qmt.password or os.environ.get("QMT_PASSWORD", "")
        if password:
            kwargs["password"] = password
    return kwargs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A-Share Multi-Factor Quant Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--config", "-c", type=str, default=None,
                            help="Path to config YAML file")
    run_parser.add_argument("--use-baostock", action="store_true",
                            help="Use Baostock real A-share data instead of synthetic")
    run_parser.add_argument("--force", "-f", action="store_true",
                            help="Force recomputation, bypass cache")
    run_parser.add_argument("--no-cache", action="store_true",
                            help="Disable caching entirely")
    run_parser.add_argument("--cache-dir", type=str, default=".quant_cache",
                            help="Cache directory path")
    run_parser.add_argument("--description", "-d", type=str, default=None,
                            help="Optional description for this run (saved in config version)")

    # trade
    trade_parser = subparsers.add_parser("trade", help="Run live trading (Paper or QMT)")
    trade_parser.add_argument("--broker", "-b", type=str, default="simulated",
                            choices=["simulated", "paper", "qmt", "qmt_sim"],
                            help="Broker type")
    trade_parser.add_argument("--universe", "-u", type=str, default=None,
                            help="Comma-separated stock codes (default: 5 blue chips)")
    trade_parser.add_argument("--days", "-d", type=int, default=30,
                            help="Number of days to simulate")
    trade_parser.add_argument("--cash", type=float, default=10_000_000,
                            help="Initial capital in CNY")
    trade_parser.add_argument("--seed", "-s", type=int, default=42,
                            help="Random seed for price generation")
    trade_parser.add_argument("--no-dual-track", action="store_true",
                            help="Disable PaperBroker parallel tracking")

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze existing results")
    analyze_parser.add_argument("--results-dir", "-r", type=str, default="./results",
                                help="Results directory path")

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare multiple strategies")
    compare_parser.add_argument("--config", "-c", type=str, default=None,
                                help="Path to config YAML file")
    compare_parser.add_argument("--optimizers", type=str, default=None,
                                help="Comma-separated optimizers (default: equal_weight,mean_variance,risk_parity)")
    compare_parser.add_argument("--cache-dir", type=str, default=".quant_cache",
                                help="Cache directory path")

    # sweep
    sweep_parser = subparsers.add_parser("sweep", help="Grid search over parameters")
    sweep_parser.add_argument("--config", "-c", type=str, default=None,
                              help="Path to config YAML file")
    sweep_parser.add_argument("--optimizers", type=str, default=None,
                              help="Comma-separated optimizers")
    sweep_parser.add_argument("--frequencies", type=str, default=None,
                              help="Comma-separated frequencies (default: monthly,weekly)")
    sweep_parser.add_argument("--n-stocks", type=str, default=None,
                              help="Comma-separated universe sizes (default: 200)")
    sweep_parser.add_argument("--cache-dir", type=str, default=".quant_cache",
                              help="Cache directory path")

    # web
    web_parser = subparsers.add_parser("web", help="Start web server (FastAPI + Vue)")
    web_parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    web_parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    web_parser.add_argument("--no-frontend", action="store_true", help="API only mode")

    # cache
    cache_parser = subparsers.add_parser("cache", help="Manage pipeline cache")
    cache_sub = cache_parser.add_subparsers(dest="subcommand", help="Cache action")
    cache_sub.add_parser("list", help="List cached stages")
    cache_sub.add_parser("clear", help="Clear all cached files")
    cache_parser.add_argument("--cache-dir", type=str, default=".quant_cache",
                              help="Cache directory path")

    # ml
    ml_parser = subparsers.add_parser("ml", help="ML alpha signal operations")
    ml_sub = ml_parser.add_subparsers(dest="subcommand", help="ML action")
    ml_train = ml_sub.add_parser("train", help="Train ML model and show performance")
    ml_train.add_argument("--model", type=str, default="lightgbm",
                          choices=["xgboost", "lightgbm"], help="Model type")
    ml_train.add_argument("--splits", type=int, default=5, help="CV splits")
    ml_train.add_argument("--config", "-c", type=str, default=None, help="Config path")
    ml_signal = ml_sub.add_parser("signal", help="Generate ML signals")
    ml_signal.add_argument("--model", type=str, default="lightgbm",
                           choices=["xgboost", "lightgbm"], help="Model type")
    ml_signal.add_argument("--config", "-c", type=str, default=None, help="Config path")

    # research
    research_parser = subparsers.add_parser("research", help="LLM research agent")
    research_sub = research_parser.add_subparsers(dest="subcommand", help="Research action")
    research_analyze = research_sub.add_parser("report", help="Analyze backtest results with LLM")
    research_analyze.add_argument("--results-dir", "-r", type=str, default="./results",
                                  help="Results directory")

    # walkforward (standalone)
    wf_parser = subparsers.add_parser("walkforward", help="Walk-forward validation")
    wf_parser.add_argument("--config", "-c", type=str, default=None,
                           help="Path to config YAML file")
    wf_parser.add_argument("--folds", type=int, default=6,
                           help="Number of walk-forward folds")
    wf_parser.add_argument("--method", type=str, default="expanding",
                           choices=["expanding", "rolling"],
                           help="Walk-forward method")
    wf_parser.add_argument("--train-days", type=int, default=504,
                           help="Training window length in days")
    wf_parser.add_argument("--test-days", type=int, default=126,
                           help="Test window length in days")
    wf_parser.add_argument("--use-baostock", action="store_true",
                           help="Use Baostock data")

    # screen
    screen_parser = subparsers.add_parser("screen", help="Screen stocks using boolean factor rules")
    screen_parser.add_argument("--config", "-c", type=str, default=None,
                               help="Path to config YAML file")
    screen_parser.add_argument("--rules", "-r", type=str, default=None,
                               help='Rules: "factor=op:value,factor=op:value" e.g. "pe_ratio=lt:30,roe=gt:0.15"')
    screen_parser.add_argument("--logic", "-l", type=str, default=None,
                               choices=["and", "or"], help="Rule combination logic")
    screen_parser.add_argument("--output", "-o", type=str, default=None,
                               help="Save results to CSV")
    screen_parser.add_argument("--min-stocks", type=int, default=None,
                               help="Minimum qualifying stocks (auto-relax)")
    screen_parser.add_argument("--max-stocks", type=int, default=None,
                               help="Maximum qualifying stocks (cap with scoring)")
    screen_parser.add_argument("--use-baostock", action="store_true",
                               help="Use Baostock real data")

    # config
    config_parser = subparsers.add_parser("config", help="Manage configuration versions")
    config_sub = config_parser.add_subparsers(dest="subcommand", help="Config action")

    config_list = config_sub.add_parser("list", help="List all config versions")
    config_show = config_sub.add_parser("show", help="Show config for a version")
    config_show.add_argument("version", type=str, help="Version ID (e.g. v3)")
    config_diff = config_sub.add_parser("diff", help="Diff two config versions")
    config_diff.add_argument("v1", type=str, help="First version (before)")
    config_diff.add_argument("v2", type=str, help="Second version (after)")
    config_rollback = config_sub.add_parser("rollback", help="Restore a version as active config")
    config_rollback.add_argument("version", type=str, help="Version ID to restore (e.g. v3)")
    config_rollback.add_argument("--target", "-t", type=str, default=None,
                                  help="Target config path (default: config/default.yaml)")
    config_delete = config_sub.add_parser("delete", help="Delete a config version")
    config_delete.add_argument("version", type=str, help="Version ID to delete")
    config_delete.add_argument("--force", "-f", action="store_true",
                                help="Skip confirmation")

    # lookahead
    la_parser = subparsers.add_parser("check-lookahead", help="Detect lookahead bias in factor pipeline")
    la_parser.add_argument("--config", "-c", type=str, default=None,
                           help="Path to config YAML file")
    la_parser.add_argument("--threshold", type=float, default=1e-4,
                           help="Signal diff threshold for bias detection")
    la_parser.add_argument("--max-dates", type=int, default=20,
                           help="Maximum number of dates to check")
    la_parser.add_argument("--use-baostock", action="store_true",
                           help="Use Baostock real data")

    # execute
    exec_parser = subparsers.add_parser("execute", help="Full pipeline: factors -> signal -> orders -> fills")
    exec_parser.add_argument("--config", "-c", type=str, default=None, help="Config path")
    exec_parser.add_argument("--use-baostock", action="store_true", help="Use Baostock data")

    # profile
    profile_parser = subparsers.add_parser("profile", help="Profile pipeline performance")
    profile_parser.add_argument("--config", "-c", type=str, default=None, help="Config path")
    profile_parser.add_argument("--force", "-f", action="store_true", help="Force recomputation")

    args = parser.parse_args()

    # Configure logging once — all child loggers inherit handlers from root
    setup_logging()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "trade":
        return cmd_trade(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "sweep":
        return cmd_sweep(args)
    elif args.command == "cache":
        return cmd_cache(args)
    elif args.command == "web":
        return cmd_web(args)
    elif args.command == "ml":
        return cmd_ml(args)
    elif args.command == "research":
        return cmd_research(args)
    elif args.command == "walkforward":
        return cmd_walkforward(args)
    elif args.command == "screen":
        return cmd_screen(args)
    elif args.command == "execute":
        return cmd_execute(args)
    elif args.command == "config":
        return cmd_config(args)
    elif args.command == "profile":
        return cmd_profile(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
