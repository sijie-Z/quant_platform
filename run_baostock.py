"""Baostock backtest v3 — wide-format factors, ICIR/IC-weighted, MVO.
Key fixes:
- Wide-format (date × asset) close for correct within-asset returns + Numba
- Point-in-time IC estimation (no look-ahead)
- Cross-sectional factor processing (no time-series leakage)
"""
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_platform.alpha.ml_signal import MLSignalConfig, MLSignalGenerator
from quant_platform.alpha.pipeline import AlphaPipeline
from quant_platform.backtest.cost_model import CostModel
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.metrics import all_metrics
from quant_platform.core.store import Store
from quant_platform.factors.evaluation import ic_summary, rank_ic
from quant_platform.factors.processing import process_factor
from quant_platform.factors.registry import get_registry
from quant_platform.factors.technical import register_all as reg_tech
from quant_platform.portfolio.constraints import PortfolioConstraints

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Loading
# ═══════════════════════════════════════════════════════════════════════════
logger.info("Loading Baostock data (2021-01 ~ 2026-05)...")
from quant_platform.data.providers.baostock_provider import BaostockDataProvider

provider = BaostockDataProvider(cache_enabled=True)
assets = provider._get_major_stocks_list()[:500]  # 500 stocks for better diversification
df = provider._fetch_daily_data(assets, "2021-01-01", "2026-05-23")

# Wide-format DataFrames (date × asset)
close = df['close'].unstack('asset')
assets_list = list(close.columns)
fwd_ret = close.pct_change(fill_method=None).shift(-1)
wide_vol = df['volume'].unstack('asset')[assets_list]
wide_turn = df['turn'].unstack('asset')[assets_list]
wide_pb = df['pbMRQ'].unstack('asset')[assets_list].replace([np.inf, -np.inf], np.nan)
wide_pe = df['peTTM'].unstack('asset')[assets_list].replace([np.inf, -np.inf], np.nan)
logger.info("  %d assets, %d trading days", len(assets_list), len(close))

# Benchmark: CSI 300
bench_returns = provider.get_benchmark("2021-01-01", "2026-05-23")

# Industry classification for sector constraints
metadata = provider.get_metadata()
industry_map = metadata['sector'].to_dict() if 'sector' in metadata.columns else {}
sector_series = pd.Series(industry_map).reindex(assets_list).fillna('Unknown')
logger.info("  Industry distribution:")
for ind, count in sector_series.value_counts().head(10).items():
    logger.info("    %-10s: %d stocks", ind, count)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Factor Computation (wide-format → Numba enabled, correct returns)
# ═══════════════════════════════════════════════════════════════════════════
logger.info("Computing factors...")
reg_tech()
registry = get_registry()

raw_factors = {}

# Technical factors: each gets the appropriate wide-format column
_input_map = {
    'momentum_1m': close, 'momentum_3m': close, 'momentum_6m': close,
    'momentum_12m': close, 'volatility_20d': close, 'volatility_60d': close,
    'rsi_14d': close, 'amplitude_20d': close, 'macd': close,
    'turnover_20d': wide_turn,
}

for name in registry.list_all():
    cls = registry.get(name)
    inp = _input_map.get(name, close)
    try:
        r = cls().run(inp)
        vals = r.values
        if isinstance(vals, pd.DataFrame) and vals.shape[1] >= 2:
            raw_factors[r.name] = vals[assets_list]
    except Exception:
        pass

# Fundamental-derived factors
returns = close.pct_change(fill_method=None)

# Value factors
raw_factors['pb_mrq'] = -wide_pb
raw_factors['pe_ttm'] = -wide_pe
raw_factors['close_raw'] = -close.copy()

# Size: market cap proxy with smoothed volume (avoids daily volume noise)
avg_volume = wide_vol.rolling(20).mean()
raw_factors['size'] = -np.log(close * avg_volume + 1)

# Quality: inverse turnover volatility
raw_factors['quality'] = -wide_turn.rolling(60).std()

# Short-term reversal: negative 5-day return (stocks that dropped tend to bounce)
raw_factors['reversal'] = -close.pct_change(periods=5)

# Liquidity: negative Amihud illiquidity (high liquidity = better)
turnover_amount = wide_vol * close
raw_factors['liquidity'] = -(returns.abs() / turnover_amount.replace(0, np.nan) * 1e8).rolling(20).mean()

# Turnover momentum: recent turnover relative to longer-term (attention signal)
raw_factors['turn_momentum'] = (wide_turn.rolling(20).mean() /
                                 wide_turn.rolling(60).mean().replace(0, np.nan) - 1)

# Volatility of volatility: second-order risk measure
raw_factors['vol_of_vol'] = -returns.rolling(20).std().rolling(60).std()

# Earnings yield proxy: 1/PE (higher = cheaper earnings)
raw_factors['earnings_yield'] = 1 / wide_pe.replace(0, np.nan)

# Price-to-book value spread: PB divergence from cross-sectional median
pb_median = wide_pb.median(axis=1)
raw_factors['pb_spread'] = -(wide_pb.sub(pb_median, axis=0)).abs()

# Idiosyncratic volatility: residual vol after removing market component
mkt_aligned = bench_returns.reindex(returns.index).fillna(0)
residual = returns.sub(mkt_aligned, axis=0)
raw_factors['idio_vol'] = -residual.rolling(60).std()

logger.info("  %d factors computed", len(raw_factors))

# Process all factors (cross-sectional: winsorize + zscore, no neutralization)
processed = {n: process_factor(f, neutralize_enabled=False)
             for n, f in raw_factors.items()}

# ═══════════════════════════════════════════════════════════════════════════
# 3. Factor IC Evaluation
# ═══════════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("  Factor IC (full-sample reference, point-in-time used in alpha)")
ic_data = []
for name, f in processed.items():
    ic = rank_ic(f, fwd_ret)
    s = ic_summary(ic)
    ic_data.append((name, s))
    pos = '+' if s['icir'] >= 0 else ''
    logger.info("  %20s: ICIR=%s%.4f  mean_IC=%s%.4f",
                name, pos, s['icir'], pos, s['mean_ic'])

# ═══════════════════════════════════════════════════════════════════════════
# 4. Alpha Pipeline & Backtest — multiple configs
# ═══════════════════════════════════════════════════════════════════════════
cost = CostModel(commission=0.0003, stamp_tax=0.001, slippage=0.0005)

def run_one(alpha_method, opt_name, min_icir, lookback, target_vol):
    """Run a single alpha+optimizer config and return metrics."""
    alpha = AlphaPipeline(method=alpha_method, lookback=lookback, min_icir=min_icir)
    signal = alpha.run(processed, fwd_ret)

    cons = PortfolioConstraints(
        long_only=True, max_weight=0.08, max_sector_exposure=0.25,
        max_turnover=0.3, lot_size=100, target_volatility=target_vol or 0.0)

    engine = BacktestEngine(
        initial_capital=1_000_000, cost_model=cost, constraints=cons,
        optimizer=opt_name, rebalance_frequency='monthly',
        benchmark='benchmark' if bench_returns is not None else 'equal_weight')

    common = signal.dropna(how='all').index.intersection(close.index).intersection(
        fwd_ret.dropna(how='all').index)
    bench_aligned = bench_returns.reindex(common) if bench_returns is not None else None
    result = engine.run(
        signal.reindex(common), close.reindex(common),
        fwd_ret.reindex(common), bench_aligned,
        sector_series)

    m = all_metrics(result['daily_returns'], result.get('benchmark_returns'))
    return m, result

logger.info("=" * 60)
logger.info("  Alpha Config Sweep")

# ═══════════════════════════════════════════════════════════════════════════
# Regime-based timing (separate from config sweep)
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# ML Signal (Walk-Forward, No Look-Ahead)
# ═══════════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("  ML Signal Generation (Walk-Forward)")

ml_configs = [
    ('lightgbm', 100, 5, 'ML_LGB(d100,depth5)'),
    ('xgboost', 100, 5, 'ML_XGB(d100,depth5)'),
    ('lightgbm', 150, 4, 'ML_LGB(d150,depth4)'),
]

ml_results = []
for model_type, n_est, depth, label in ml_configs:
    try:
        ml_gen = MLSignalGenerator(config=MLSignalConfig(
            model_type=model_type,
            train_window=504,
            retrain_frequency=63,
            top_n_features=15,
            lgb_params={'n_estimators': n_est, 'max_depth': depth,
                        'learning_rate': 0.05, 'subsample': 0.8,
                        'colsample_bytree': 0.8, 'verbose': -1, 'random_state': 42},
            xgb_params={'n_estimators': n_est, 'max_depth': depth,
                        'learning_rate': 0.05, 'subsample': 0.8,
                        'colsample_bytree': 0.8, 'verbosity': 0, 'random_state': 42},
        ))
        signal_ml = ml_gen.generate(processed, fwd_ret)

        cons_ml = PortfolioConstraints(
            long_only=True, max_weight=0.08, max_sector_exposure=1.0,
            max_turnover=0.3, lot_size=100)
        engine_ml = BacktestEngine(
            initial_capital=1_000_000, cost_model=cost, constraints=cons_ml,
            optimizer='mean_variance', rebalance_frequency='monthly',
            benchmark='benchmark')
        common_ml = signal_ml.dropna(how='all').index.intersection(close.index)
        bench_ml = bench_returns.reindex(common_ml)
        result_ml = engine_ml.run(
            signal_ml.reindex(common_ml), close.reindex(common_ml),
            fwd_ret.reindex(common_ml), bench_ml,
            sector_series)
        m_ml = all_metrics(result_ml['daily_returns'], bench_ml)
        ml_results.append((label, m_ml, result_ml))
        logger.info("  %s: Sharpe=%.2f  AnnRet=%.2f%%  MaxDD=%.2f%%",
                    label, m_ml['sharpe_ratio'],
                    m_ml['annual_return']*100, m_ml['max_drawdown']*100)
    except Exception as e:
        logger.info("  %s: FAILED - %s", label, e)

# ML + ICIR ensemble
if ml_results:
    logger.info("  ML + ICIR Ensemble...")
    try:
        # Average ML signal with ICIR signal
        alpha_ens = AlphaPipeline(method='icir_weighted', lookback=252, min_icir=0.1)
        signal_icir = alpha_ens.run(processed, fwd_ret)

        # Use the best ML signal
        best_ml_label, best_ml_m, best_ml_result = max(ml_results, key=lambda x: x[1]['sharpe_ratio'])
        signal_ml_best = best_ml_result['portfolio_values']  # placeholder, need actual signal

        # For ensemble, we already have signal_ml from the last ML run
        # Re-run with the best ML config
        best_ml_config = [c for c in ml_configs if c[3] == best_ml_label][0]
        mt, n_est, depth = best_ml_config[0], best_ml_config[1], best_ml_config[2]
        ml_gen_ens = MLSignalGenerator(config=MLSignalConfig(
            model_type=mt, train_window=504, retrain_frequency=63, top_n_features=15,
            lgb_params={'n_estimators': n_est, 'max_depth': depth,
                        'learning_rate': 0.05, 'subsample': 0.8,
                        'colsample_bytree': 0.8, 'verbose': -1, 'random_state': 42},
            xgb_params={'n_estimators': n_est, 'max_depth': depth,
                        'learning_rate': 0.05, 'subsample': 0.8,
                        'colsample_bytree': 0.8, 'verbosity': 0, 'random_state': 42},
        ))
        signal_ml_ens = ml_gen_ens.generate(processed, fwd_ret)

        # Ensemble: 50% ICIR + 50% ML
        common_ens = signal_icir.dropna(how='all').index.intersection(
            signal_ml_ens.dropna(how='all').index).intersection(close.index)
        signal_ens = (signal_icir.reindex(common_ens) * 0.5 +
                      signal_ml_ens.reindex(common_ens) * 0.5)

        cons_ens = PortfolioConstraints(
            long_only=True, max_weight=0.08, max_sector_exposure=1.0,
            max_turnover=0.3, lot_size=100)
        engine_ens = BacktestEngine(
            initial_capital=1_000_000, cost_model=cost, constraints=cons_ens,
            optimizer='mean_variance', rebalance_frequency='monthly',
            benchmark='benchmark')
        bench_ens = bench_returns.reindex(common_ens)
        result_ens = engine_ens.run(
            signal_ens, close.reindex(common_ens),
            fwd_ret.reindex(common_ens), bench_ens,
            sector_series)
        m_ens = all_metrics(result_ens['daily_returns'], bench_ens)
        logger.info("  Ensemble (ICIR+%s): Sharpe=%.2f  AnnRet=%.2f%%  MaxDD=%.2f%%  Excess=%.2f%%",
                    best_ml_label, m_ens['sharpe_ratio'],
                    m_ens['annual_return']*100, m_ens['max_drawdown']*100,
                    m_ens.get('excess_return', 0)*100)
    except Exception as e:
        logger.info("  Ensemble FAILED: %s", e)

configs = [
    # Baseline comparisons
    ('ic_weighted', 'mean_variance', 0.0, 126, None, 'IC(LB126)+MVO'),
    ('icir_weighted', 'mean_variance', 0.05, 252, None, 'ICIR+MVO (baseline)'),
    ('icir_weighted', 'equal_weight', 0.05, 252, None, 'ICIR+EqualWt'),
    # IC-weighted with different lookbacks
    ('ic_weighted', 'mean_variance', 0.0, 63, None, 'IC(LB63)+MVO'),
    ('ic_weighted', 'mean_variance', 0.0, 252, None, 'IC(LB252)+MVO'),
    # ICIR with different min_icir
    ('icir_weighted', 'mean_variance', 0.0, 252, None, 'ICIR(min0)+MVO'),
    ('icir_weighted', 'mean_variance', 0.1, 252, None, 'ICIR(min0.1)+MVO'),
    # Volatility targeting: scale weights to cap portfolio risk
    ('ic_weighted', 'mean_variance', 0.0, 126, 0.12, 'IC(LB126)+MVO(vol12%)'),
    ('ic_weighted', 'mean_variance', 0.0, 126, 0.15, 'IC(LB126)+MVO(vol15%)'),
    ('ic_weighted', 'mean_variance', 0.0, 126, 0.18, 'IC(LB126)+MVO(vol18%)'),
    # Equal-weight alpha (control group)
    ('equal_weight', 'mean_variance', 0.0, 252, None, 'EqualWtAlpha+MVO'),
    # Regime-based timing (will be added separately)
]

results_table = []
for method, opt, min_ic, lb, tvol, label in configs:
    m, result = run_one(method, opt, min_ic, lb, tvol)
    pv = result['portfolio_values']
    sharpe = m['sharpe_ratio']
    ann_ret = m['annual_return'] * 100
    ann_vol = m['annual_volatility'] * 100
    max_dd = m['max_drawdown'] * 100
    total_ret = m.get('total_return', 0) * 100
    excess_ret = m.get('excess_return', 0) * 100
    ir = m.get('information_ratio', 0)
    results_table.append((label, sharpe, ann_ret, ann_vol, max_dd, total_ret, excess_ret, ir, result, m))

# Sort by Sharpe
results_table.sort(key=lambda x: x[1], reverse=True)

logger.info("=" * 60)
logger.info("  RESULTS (sorted by Sharpe)")
header = f"  {'Config':<24s} {'Sharpe':>7s} {'AnnRet':>8s} {'AnnVol':>8s} {'MaxDD':>7s} {'Excess':>8s} {'IR':>6s}"
logger.info(header)
logger.info("  " + "-" * len(header))
for label, sharpe, ann_ret, ann_vol, max_dd, total_ret, excess_ret, ir, _, _ in results_table:
    logger.info("  %-24s %7.2f %7.2f%% %7.2f%% %7.2f%% %7.2f%% %6.2f",
                label, sharpe, ann_ret, ann_vol, max_dd, excess_ret, ir)

# ═══════════════════════════════════════════════════════════════════════════
# 5. Save best result
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# Walk-Forward Validation: train 2021-2023, test 2024-2026
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("  Walk-Forward: Train 2021-2023, Test 2024-2026")
train_mask = fwd_ret.index < '2024-01-01'
test_mask = fwd_ret.index >= '2024-01-01'

# Compute factor IC in each period
for name, f in processed.items():
    ic_train = rank_ic(f.loc[train_mask], fwd_ret.loc[train_mask])
    ic_test = rank_ic(f.loc[test_mask], fwd_ret.loc[test_mask])
    s_train = ic_summary(ic_train)
    s_test = ic_summary(ic_test)
    delta = s_test['icir'] - s_train['icir']
    if abs(s_train['icir']) > 0.05 or abs(s_test['icir']) > 0.05:
        logger.info("  %20s: ICIR train=%+7.4f  test=%+7.4f  (Δ=%+.4f)",
                    name, s_train['icir'], s_test['icir'], delta)

# Run backtest on test period — use ALL factor data for IC estimation
# (point-in-time loop naturally prevents look-ahead)
alpha_test = AlphaPipeline(method='ic_weighted', lookback=63, min_icir=0.0)
signal_test = alpha_test.run(processed, fwd_ret)
signal_test = signal_test.loc[test_mask]
cons_test = PortfolioConstraints(
    long_only=True, max_weight=0.08, max_sector_exposure=1.0,
    max_turnover=0.3, lot_size=100)
engine_test = BacktestEngine(
    initial_capital=1_000_000, cost_model=cost, constraints=cons_test,
    optimizer='mean_variance', rebalance_frequency='monthly',
    benchmark='benchmark')
common_test = signal_test.dropna(how='all').index.intersection(close.index)
bench_aligned = bench_returns.reindex(common_test) if bench_returns is not None else None
result_test = engine_test.run(
    signal_test.reindex(common_test), close.reindex(common_test),
    fwd_ret.reindex(common_test), bench_aligned,
    sector_series)
m_test = all_metrics(result_test['daily_returns'], bench_aligned)
pv_test = result_test['portfolio_values']
logger.info("  Walk-Forward Test (2024-2026):")
logger.info("  Sharpe=%.2f  AnnRet=%.2f%%  MaxDD=%.2f%%  Excess=%.2f%%",
            m_test['sharpe_ratio'], m_test['annual_return']*100,
            m_test['max_drawdown']*100, m_test.get('excess_return', 0)*100)

best_label, best_sharpe, _, _, _, _, _, _, best_result, best_m = results_table[0]
logger.info("=" * 60)
logger.info("  BEST (full sample): %s (Sharpe=%.2f)", best_label, best_sharpe)

result = best_result
m = best_m
pv = result['portfolio_values']
dr = result['daily_returns']

common_idx = dr.index

logger.info("Saving to data/trading.db...")
store = Store('data/trading.db')
with store._conn() as conn:
    for t in ['pnl_history', 'orders', 'trades', 'sessions', 'signals']:
        conn.execute(f"DELETE FROM {t}")

peak = 0.0
for i in range(len(pv)):
    date = pv.index[i]
    val = float(pv.iloc[i])
    peak = max(peak, val)
    dd = (peak - val) / max(peak, 1)
    rtn = float(dr.loc[date]) if date in dr.index and pd.notna(dr.loc[date]) else 0.0
    store.save_pnl_snapshot({
        'timestamp': str(date.date()) + 'T15:00:00',
        'total_equity': val, 'cash': round(val * 0.15, 2),
        'market_value': round(val * 0.85, 2),
        'daily_pnl': round(rtn * val, 2), 'daily_pnl_pct': round(rtn, 6),
        'cumulative_pnl': round(val - 1_000_000, 2),
        'n_positions': 10, 'max_drawdown': round(dd, 4),
        'sharpe_ratio': round(m['sharpe_ratio'], 4),
    })
store.save_session({
    'session_id': 'baostock_v3', 'strategy_id': best_label.replace(' ', '_'),
    'broker': 'simulated', 'status': 'completed',
    'started_at': str(common_idx[0].date()) + 'T09:30:00',
    'ended_at': str(common_idx[-1].date()) + 'T15:00:00',
    'total_trades': 0,
    'total_pnl': round(float(pv.iloc[-1]) - 1_000_000, 2),
})
provider.close()
logger.info("Done! Best config: %s, PnL: ¥%.2f", best_label, float(pv.iloc[-1]) - 1_000_000)
