"""Regression tests for the statistics used by the research scripts.

Covers the corrections made in:
- tools/run_market_structure_v3.py: Newey-West HAC t-statistics for
  overlapping forward windows, and same-day price-proxy size grouping.
- tools/run_rq1_cluster_attribution.py: per-date cross-sectional cluster
  correlation on the asset intersection (instead of correlating daily means).
- tools/run_rq3_stability.py: contemporaneous regime labels and %Time over
  actual labelled trading days (instead of calendar days / forward-filled labels).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_tool(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ms3():
    return _load_tool("_executor_g_market_structure_v3", "run_market_structure_v3.py")


@pytest.fixture(scope="module")
def rq1():
    return _load_tool("_executor_g_rq1_cluster_attribution", "run_rq1_cluster_attribution.py")


@pytest.fixture(scope="module")
def rq3():
    return _load_tool("_executor_g_rq3_stability", "run_rq3_stability.py")


# ---------------------------------------------------------------------------
# run_market_structure_v3: HAC t-statistics
# ---------------------------------------------------------------------------


def test_newey_west_tstat_is_smaller_than_naive_on_overlapping_series(ms3):
    """Positively autocorrelated (overlapping-window) ICs must lose significance."""
    rng = np.random.default_rng(20261020)
    n = 400
    shocks = rng.normal(0.0, 1.0, n)
    # MA(1) with strong positive persistence, the shape induced when
    # consecutive H-day forward windows share return days.
    x = 0.5 + shocks
    x[1:] += 0.8 * shocks[:-1]

    t_naive = ms3.naive_tstat(x)
    t_hac = ms3.newey_west_tstat(x, lag=3)

    assert np.isfinite(t_naive) and np.isfinite(t_hac)
    assert abs(t_hac) < abs(t_naive)


def test_hac_with_zero_lag_reduces_to_naive_tstat(ms3):
    rng = np.random.default_rng(11)
    x = rng.normal(size=200)
    assert ms3.newey_west_tstat(x, lag=0) == pytest.approx(ms3.naive_tstat(x))


def test_compute_horizon_ic_reports_naive_and_hac_columns(ms3):
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=320)
    assets = [f"S{i:02d}" for i in range(30)]
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )

    result = ms3.compute_horizon_ic(returns, horizon=60, sample_interval=21)

    assert result["n_obs"] > 0
    assert result["hac_lag"] == 3  # ceil(60 / 21)
    assert result["t_stat"] == result["t_stat_naive"]
    assert np.isfinite(result["t_stat_naive"])
    assert result["t_stat_hac"] == pytest.approx(
        ms3.newey_west_tstat(result["ic_series"], result["hac_lag"])
    )


# ---------------------------------------------------------------------------
# run_market_structure_v3: same-day price-proxy size groups (no look-ahead)
# ---------------------------------------------------------------------------


def _price_panel(n_dates=25, n_assets=12, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    assets = [f"S{i:02d}" for i in range(n_assets)]
    prices = pd.DataFrame(
        rng.lognormal(mean=4.0, sigma=0.4, size=(n_dates, n_assets)),
        index=dates,
        columns=assets,
    )
    return prices, dates


def test_price_groups_are_same_day_cross_sectional_terciles(ms3):
    prices, dates = _price_panel()
    date = dates[7]
    groups = ms3.assign_price_groups(prices, date)

    assert set(groups.unique()) == {"Small", "Mid", "Large"}
    counts = groups.value_counts()
    assert counts.max() - counts.min() <= 1

    day_prices = prices.loc[date]
    assert day_prices[groups == "Small"].max() <= day_prices[groups == "Large"].min()


def test_price_groups_ignore_future_prices(ms3):
    """Changing future prices must not change a past date's group membership."""
    prices, dates = _price_panel()
    date = dates[10]

    before = ms3.assign_price_groups(prices, date)

    shocked = prices.copy()
    shocked.loc[dates[11:]] *= 1000.0
    after = ms3.assign_price_groups(shocked, date)

    pd.testing.assert_series_equal(before, after)


def test_price_groups_change_when_same_day_prices_change(ms3):
    """Groups must come from the sampled date's cross-section, not a sample mean."""
    prices, dates = _price_panel()
    date = dates[6]

    before = ms3.assign_price_groups(prices, date)

    reshaped = prices.copy()
    # Reciprocal prices reverse the cross-sectional ordering deterministically.
    reshaped.loc[date] = 1.0 / prices.loc[date]
    after = ms3.assign_price_groups(reshaped, date)

    assert not before.equals(after)
    cheapest = prices.loc[date].idxmin()
    assert before[cheapest] == "Small"
    assert after[cheapest] == "Large"


def test_group_ic_does_not_require_a_full_sample_price_mean(ms3):
    """The group IC path must run off per-date groups alone, with no avg-price split."""
    rng = np.random.default_rng(5)
    prices, _ = _price_panel(n_dates=60, seed=5)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=prices.shape),
        index=prices.index,
        columns=prices.columns,
    )

    by_group = ms3.compute_horizon_ic_by_price_group(
        returns, prices, horizon=5, sample_interval=21, min_names=2
    )

    assert set(by_group) == {"Small", "Mid", "Large"}
    for summary in by_group.values():
        assert summary["hac_lag"] == 1  # ceil(5 / 21)
        assert summary["n_obs"] > 0


def test_group_ic_is_invariant_to_post_sample_prices_end_to_end(ms3):
    """Shocking prices after the last sampled date must leave every group IC unchanged."""
    rng = np.random.default_rng(12)
    prices, _ = _price_panel(n_dates=60, seed=12)
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=prices.shape),
        index=prices.index,
        columns=prices.columns,
    )
    kwargs = {"horizon": 5, "sample_interval": 21, "min_names": 3}

    baseline = ms3.compute_horizon_ic_by_price_group(returns, prices, **kwargs)
    sampled = ms3.sample_dates_for_horizon(prices.index, 5, 21)
    assert len(sampled) > 0

    shocked = prices.copy()
    shocked.loc[shocked.index > sampled[-1]] *= 1000.0
    after = ms3.compute_horizon_ic_by_price_group(returns, shocked, **kwargs)

    for label in baseline:
        assert baseline[label]["ic_series"] == after[label]["ic_series"]
        assert baseline[label]["mean_ic"] == pytest.approx(after[label]["mean_ic"])


def test_summarize_ic_series_accepts_numpy_arrays(ms3):
    values = np.array([0.1, -0.2, 0.3, 0.05])
    summary = ms3.summarize_ic_series(values, hac_lag=1)
    assert summary["n_obs"] == 4
    assert summary["t_stat_naive"] == pytest.approx(ms3.naive_tstat(values))


def test_short_history_returns_an_empty_ic_summary(ms3):
    dates = pd.bdate_range("2021-01-04", periods=10)
    returns = pd.DataFrame(0.001, index=dates, columns=["A", "B"])

    result = ms3.compute_horizon_ic(returns, horizon=60, sample_interval=21)

    assert result["n_obs"] == 0
    assert np.isnan(result["mean_ic"])


# ---------------------------------------------------------------------------
# run_rq1_cluster_attribution: asset-level cross-sectional correlation
# ---------------------------------------------------------------------------


def test_cross_sectional_corr_uses_only_common_assets(rq1):
    """Assets present in only one cluster must not enter the correlation."""
    dates = pd.date_range("2021-01-04", periods=5, freq="B")
    signal_a = pd.DataFrame(
        {
            "x1": [1.0, 1.0, 1.0, 1.0, 1.0],
            "x2": [2.0, 2.0, 2.0, 2.0, 2.0],
            "a_only": [-50.0, 40.0, -30.0, 20.0, -10.0],
        },
        index=dates,
    )
    signal_b = pd.DataFrame(
        {
            "x1": [5.0, 5.0, 5.0, 5.0, 5.0],
            "x2": [3.0, 3.0, 3.0, 3.0, 3.0],
            "b_only": [10.0, -20.0, 30.0, -40.0, 50.0],
        },
        index=dates,
    )

    corr, n_assets = rq1.per_date_cross_sectional_corr(
        signal_a, signal_b, min_assets=2
    )

    assert len(corr) == len(dates)
    assert np.allclose(corr.to_numpy(), -1.0)
    assert (n_assets == 2).all()


def test_cross_sectional_corr_handles_nan_asset_drops(rq1):
    dates = pd.date_range("2021-01-04", periods=3, freq="B")
    signal_a = pd.DataFrame(
        {"x1": [1.0, 2.0, 3.0], "x2": [2.0, 1.0, 2.0], "x3": [3.0, 3.0, 1.0]},
        index=dates,
    )
    signal_b = pd.DataFrame(
        {"x1": [3.0, 1.0, 1.0], "x2": [1.0, 2.0, 3.0], "x3": [2.0, 3.0, 2.0]},
        index=dates,
    )
    signal_b.loc[dates[0], "x3"] = np.nan

    _, n_assets = rq1.per_date_cross_sectional_corr(signal_a, signal_b, min_assets=2)

    assert n_assets.loc[dates[0]] == 2
    assert n_assets.loc[dates[1]] == 3


def test_correlation_summary_retains_renamed_daily_mean_diagnostic(rq1):
    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2021-01-04", periods=40)
    assets = [f"S{i:02d}" for i in range(20)]
    signal_a = pd.DataFrame(rng.normal(size=(len(dates), len(assets))), index=dates, columns=assets)
    signal_b = pd.DataFrame(rng.normal(size=(len(dates), len(assets))), index=dates, columns=assets)

    summary = rq1.compute_cross_sectional_correlation_summary(
        {"A": signal_a, "B": signal_b}, min_assets=5
    )
    row = summary.iloc[0]

    assert row["n_dates"] == len(dates)
    assert row["n_assets_mean"] == len(assets)
    assert np.isfinite(row["xs_pearson_mean"])
    # The legacy statistic is kept, but explicitly renamed as a diagnostic.
    assert np.isfinite(row["ts_pearson_of_daily_means"])
    assert "coverage" in row["ts_note"]


def test_cross_sectional_corr_matches_the_old_daily_mean_only_by_coincidence(rq1):
    """The two statistics answer different questions and must not be conflated."""
    dates = pd.bdate_range("2021-01-04", periods=4)
    pattern = np.array([0.0, 1.0, 2.0, 3.0])
    scale_a = np.array([1.0, 2.0, 3.0, 4.0])
    # scale_b is positively correlated with nothing: its centred version is
    # orthogonal to centred scale_a, so the daily-mean series are uncorrelated.
    scale_b = np.array([1.5, 3.5, 0.5, 2.5])
    assets = [f"S{i}" for i in range(4)]

    signal_a = pd.DataFrame([s * pattern for s in scale_a], index=dates, columns=assets)
    signal_b = pd.DataFrame([s * pattern for s in scale_b], index=dates, columns=assets)

    xs_corr, _ = rq1.per_date_cross_sectional_corr(signal_a, signal_b, min_assets=2)
    ts_corr = rq1.daily_mean_timeseries_corr(signal_a, signal_b)

    # Every date ranks the assets identically, so asset-level correlation is 1.
    assert np.allclose(xs_corr.to_numpy(), 1.0)
    # The daily means, however, are orthogonal by construction.
    assert ts_corr == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# run_rq3_stability: contemporaneous regimes and trading-day %Time
# ---------------------------------------------------------------------------


def _benchmark_with_monthly_targets(trading_days, monthly_targets):
    """Daily benchmark returns compounding to the target in each covered month."""
    values = {}
    for date in trading_days:
        target = monthly_targets.get(date.month)
        if target is None:
            continue  # month intentionally absent from the benchmark
        n_days = int((trading_days.month == date.month).sum())
        values[date] = (1.0 + target) ** (1.0 / n_days) - 1.0
    return pd.Series(values, name="benchmark")


def test_regime_label_is_contemporaneous_not_forward_filled(rq3):
    trading_days = pd.bdate_range("2021-01-01", "2021-06-30")
    targets = {1: 0.05, 2: -0.05, 3: 0.00, 4: 0.04, 5: -0.03, 6: 0.01}
    benchmark = _benchmark_with_monthly_targets(trading_days, targets)

    daily_regime = rq3.classify_regimes(benchmark, trading_days=trading_days)

    expected = {
        1: "Bull", 2: "Bear", 3: "Sideways",
        4: "Bull", 5: "Bear", 6: "Sideways",
    }
    for month, label in expected.items():
        labels = set(daily_regime[daily_regime.index.month == month].unique())
        assert labels == {label}, f"month {month} got {labels}, expected {label}"

    # The old ffill put January's Bull label on February; make that explicit.
    assert "Bull" not in set(daily_regime[daily_regime.index.month == 2].unique())

    shares = rq3.regime_shares(daily_regime)
    assert shares.sum() == pytest.approx(100.0)
    assert set(shares.index) == {"Bull", "Bear", "Sideways"}


def test_regime_shares_exclude_unlabelled_days_from_denominator(rq3):
    trading_days = pd.bdate_range("2021-01-01", "2021-06-30")
    targets = {3: 0.05, 4: -0.05, 5: 0.00, 6: 0.03}
    benchmark = _benchmark_with_monthly_targets(trading_days, targets)

    daily_regime = rq3.classify_regimes(benchmark, trading_days=trading_days)
    before_benchmark = daily_regime.loc[:"2021-02-28"]

    assert before_benchmark.isna().all()
    assert daily_regime.loc["2021-03-01":].notna().all()
    # %Time uses labelled trading days: the shares still sum to 100%, and the
    # skipped first months are visible as NaN rather than silently labelled.
    assert rq3.regime_shares(daily_regime).sum() == pytest.approx(100.0)


def test_regime_pct_time_uses_labelled_trading_days(rq3):
    trading_days = pd.bdate_range("2021-01-01", "2021-06-30")
    targets = {3: 0.05, 4: -0.05, 5: 0.00, 6: 0.03}
    benchmark = _benchmark_with_monthly_targets(trading_days, targets)
    daily_regime = rq3.classify_regimes(benchmark, trading_days=trading_days)

    labelled = int(daily_regime.notna().sum())
    total = 0.0
    for regime in ("Bull", "Bear", "Sideways"):
        dates = daily_regime.index[daily_regime == regime]
        share = rq3.regime_pct_time(dates, daily_regime)
        assert share == pytest.approx(len(dates) / labelled * 100.0)
        total += share
    assert total == pytest.approx(100.0)


def test_regime_label_thresholds_match_own_month_return(rq3):
    trading_days = pd.bdate_range("2021-01-01", "2021-04-30")
    targets = {1: 0.021, 2: 0.019, 3: -0.021, 4: -0.019}
    benchmark = _benchmark_with_monthly_targets(trading_days, targets)

    daily_regime = rq3.classify_regimes(benchmark, trading_days=trading_days)

    assert set(daily_regime[daily_regime.index.month == 1].unique()) == {"Bull"}
    assert set(daily_regime[daily_regime.index.month == 2].unique()) == {"Sideways"}
    assert set(daily_regime[daily_regime.index.month == 3].unique()) == {"Bear"}
    assert set(daily_regime[daily_regime.index.month == 4].unique()) == {"Sideways"}
