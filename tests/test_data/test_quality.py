"""Tests for data.quality — Data quality monitoring."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.data.quality import DataQualityMonitor


@pytest.fixture
def clean_prices():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    data = {}
    for i in range(50):
        base = 100 + i
        returns = np.random.normal(0.0005, 0.02, 252)
        data[f"stock_{i}"] = base * np.cumprod(1 + returns)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def dirty_prices(clean_prices):
    df = clean_prices.copy()
    df.iloc[10, 0] = np.nan
    df.iloc[20, 1] = df.iloc[19, 1] * 2
    df.iloc[30:40, 2] = df.iloc[29, 2]
    return df


class TestDataQualityMonitor:
    def test_clean_data_passes(self, clean_prices):
        monitor = DataQualityMonitor()
        results = monitor.check_prices(clean_prices)
        passed = all(c.passed for c in results if c.severity in ("error", "critical"))
        assert passed

    def test_detects_missing_data(self, dirty_prices):
        monitor = DataQualityMonitor()
        results = monitor.check_prices(dirty_prices)
        names = [c.name for c in results]
        assert "missing_data" in names

    def test_detects_price_anomalies(self, dirty_prices):
        monitor = DataQualityMonitor()
        results = monitor.check_prices(dirty_prices)
        anomaly_check = [c for c in results if c.name == "price_anomalies"]
        assert len(anomaly_check) > 0

    def test_detects_stale_prices(self, dirty_prices):
        monitor = DataQualityMonitor(max_stale_days=5)
        results = monitor.check_prices(dirty_prices)
        names = [c.name for c in results]
        assert "stale_prices" in names

    def test_report_generation(self, clean_prices):
        monitor = DataQualityMonitor()
        monitor.check_prices(clean_prices)
        report = monitor.get_report()
        assert report["total_checks"] > 0
        assert report["overall_status"] in ["PASS", "FAIL"]

    def test_check_returns(self, clean_prices):
        monitor = DataQualityMonitor()
        returns = clean_prices.pct_change(fill_method=None).dropna()
        results = monitor.check_returns(returns)
        assert len(results) > 0

    def test_negative_prices_detected(self, clean_prices):
        df = clean_prices.copy()
        df.iloc[0, 0] = -100
        monitor = DataQualityMonitor()
        results = monitor.check_prices(df)
        monotonicity = [c for c in results if c.name == "price_monotonicity"]
        assert len(monotonicity) > 0
        assert not monotonicity[0].passed


class TestMultiIndexPanel:
    """`/api/data/quality` without a run_id feeds the raw provider panel with
    a (date, asset) MultiIndex. The checks assumed a date x asset wide frame:
    pct_change compared one stock with the next (tens of thousands of fake
    anomalies) and `index[-1]` was a tuple, so freshness never failed."""

    @pytest.fixture
    def panel(self):
        from quant_platform.data.providers.synthetic import SyntheticDataProvider

        provider = SyntheticDataProvider(
            n_stocks=30, start_date="2023-01-01", end_date="2023-12-31", seed=5,
        )
        return provider.get_prices("2023-01-01", "2023-12-31")

    def test_no_fabricated_anomaly_flood(self, panel):
        from quant_platform.data.quality import PRICE_FIELDS

        monitor = DataQualityMonitor()
        results = monitor.check_prices(panel)
        check = next(c for c in results if c.name == "price_anomalies")

        price_cells = panel.shape[0] * sum(
            1 for field in PRICE_FIELDS if field in panel.columns
        )
        n_anomalies = check.details["n_anomalies"]
        # Cross-asset differencing flagged almost every cell (~price_cells).
        assert n_anomalies < 0.05 * price_cells, (
            f"{n_anomalies} anomalies out of {price_cells} price cells -- "
            "cross-asset pct_change is back"
        )

    def test_old_panel_is_not_reported_fresh(self, panel):
        monitor = DataQualityMonitor()
        monitor.check_prices(panel)
        freshness = next(c for c in monitor.checks if c.name == "freshness")

        assert not freshness.passed
        assert freshness.details["days_old"] > 3
        assert freshness.details["last_date"].startswith("2023-12-29")

    def test_all_checks_still_run_on_a_panel(self, panel):
        monitor = DataQualityMonitor()
        names = {c.name for c in monitor.check_prices(panel)}
        assert {
            "missing_data", "nan_rate", "price_anomalies",
            "stale_prices", "freshness", "price_monotonicity",
        } <= names
        assert monitor.get_report()["total_checks"] >= 6

    def test_multiindex_returns_panel_is_accepted(self, panel):
        close = panel["close"].unstack("asset")
        returns = (
            close.pct_change(fill_method=None)
            .stack("asset", future_stack=True)
            .to_frame("ret")
        )

        monitor = DataQualityMonitor()
        results = monitor.check_returns(returns)

        names = [c.name for c in results]
        assert "extreme_returns" in names
        assert "return_distribution" in names
