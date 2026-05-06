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

# Ensure the parent directory (containing quant_platform package) is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_platform.utils.config import load_config
from quant_platform.utils.logging import setup_logging, get_logger

logger = None  # Set after config load


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


def _load_data(config, use_tushare: bool = True):
    """Load and clean data. Returns (prices, returns, benchmark, metadata, financials)."""
    from quant_platform.data.providers.synthetic import SyntheticDataProvider
    from quant_platform.data.providers.tushare_loader import TushareProvider
    from quant_platform.data.pipeline import DataPipeline

    provider = None
    if use_tushare:
        try:
            provider = TushareProvider(
                start_date=config.data.start_date,
                end_date=config.data.end_date,
            )
            logger.info("Using Tushare real data provider")
        except (RuntimeError, ImportError) as e:
            logger.warning("Tushare unavailable (%s), falling back to synthetic data", e)

    if provider is None:
        provider = SyntheticDataProvider(
            n_stocks=config.universe.n_stocks,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
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
    )


def _compute_factors(prices, returns, financials, metadata):
    """Compute and process all registered factors. Returns (processed_factors, ic_results, sector_map, fin_unstacked)."""
    from quant_platform.factors.technical import register_all as register_technical
    from quant_platform.factors.fundamental import register_all as register_fundamental
    from quant_platform.factors.registry import get_registry
    from quant_platform.factors.processing import process_factor
    from quant_platform.factors.evaluation import rank_ic, ic_summary

    register_technical()
    register_fundamental()
    registry = get_registry()

    fin_unstacked = financials.unstack("asset") if financials is not None else None

    raw_factors = {}
    for factor_name in registry.list_all():
        cls = registry.get(factor_name)
        inst = cls()
        try:
            if inst.category.value == "fundamental" and fin_unstacked is not None:
                result = inst.run(prices, fin_unstacked)
            else:
                result = inst.run(prices)
            raw_factors[result.name] = result.values
        except Exception as e:
            logger.warning("Failed to compute %s: %s", factor_name, e)

    logger.info("Computed %d factors: %s", len(raw_factors), list(raw_factors.keys()))

    sector_map = metadata["sector"]
    mcap = fin_unstacked["market_cap"] if fin_unstacked is not None else None

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
    from quant_platform.backtest.engine import BacktestEngine
    from quant_platform.backtest.cost_model import CostModel
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


def _generate_signal(config, processed_factors, returns):
    """Generate alpha signal from processed factors."""
    from quant_platform.alpha.pipeline import AlphaPipeline

    alpha_pipe = AlphaPipeline(
        method=config.alpha.method,
        lookback=config.alpha.lookback,
        min_icir=config.alpha.min_icir,
    )
    return alpha_pipe.run(processed_factors, returns)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    """Execute full pipeline: data -> factors -> alpha -> portfolio -> backtest -> report."""
    import yaml
    from quant_platform.utils.cache import PipelineCache

    config_path = _resolve_config_path(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    cache_key = PipelineCache.make_config_hash(raw_config)
    cache = PipelineCache(args.cache_dir)
    use_cache = not args.force and not args.no_cache

    config = load_config(args.config)
    global logger
    logger = setup_logging()

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
        logger.info("Data loaded from cache")
    else:
        prices, returns, benchmark, metadata, financials = _load_data(config)
        if use_cache:
            cache.save_stage("data", cache_key,
                             (prices, returns, benchmark, metadata, financials))

    # ------------------------------------------------------------------
    # 2. Factors
    # ------------------------------------------------------------------
    logger.info("[2/6] Computing factors...")
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata)

    for name, summary in ic_results.items():
        logger.info("  %-20s Rank IC=%.4f  ICIR=%.2f", name, summary["mean_ic"], summary["icir"])

    # ------------------------------------------------------------------
    # 3. Alpha Signal
    # ------------------------------------------------------------------
    logger.info("[3/6] Generating alpha signal...")
    signal = _generate_signal(config, processed_factors, returns)

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
    global logger
    logger = setup_logging()

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

    global logger
    logger = setup_logging()

    optimizers = args.optimizers.split(",") if args.optimizers else [
        "equal_weight", "mean_variance", "risk_parity"
    ]
    optimizers = [o.strip() for o in optimizers]

    config = load_config(args.config)
    # Smaller universe for faster comparison
    config.universe.n_stocks = min(config.universe.n_stocks, 300)

    # Load data once, reuse across optimizers
    prices, returns, benchmark, metadata, financials = _load_data(config, use_tushare=False)
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata)
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

    global logger
    logger = setup_logging()

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
            prices, returns, benchmark, metadata, financials = _load_data(config, use_tushare=False)
            processed_factors, _ic_results, sector_map, fin_unstacked = _compute_factors(
                prices, returns, financials, metadata)
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
    logger = setup_logging()

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
    run_parser.add_argument("--force", "-f", action="store_true",
                            help="Force recomputation, bypass cache")
    run_parser.add_argument("--no-cache", action="store_true",
                            help="Disable caching entirely")
    run_parser.add_argument("--cache-dir", type=str, default=".quant_cache",
                            help="Cache directory path")

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

    # cache
    cache_parser = subparsers.add_parser("cache", help="Manage pipeline cache")
    cache_sub = cache_parser.add_subparsers(dest="subcommand", help="Cache action")
    cache_sub.add_parser("list", help="List cached stages")
    cache_sub.add_parser("clear", help="Clear all cached files")
    cache_parser.add_argument("--cache-dir", type=str, default=".quant_cache",
                              help="Cache directory path")

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "sweep":
        return cmd_sweep(args)
    elif args.command == "cache":
        return cmd_cache(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
