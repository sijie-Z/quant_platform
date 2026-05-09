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
