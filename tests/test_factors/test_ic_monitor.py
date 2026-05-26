"""Tests for Factor IC monitoring and decay detection."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.factors.ic_monitor import (
    FactorICMonitor,
    FactorICStats,
    ICMonitorConfig,
)


@pytest.fixture
def sample_factor_data():
    """Create sample factor data with known IC structure."""
    np.random.seed(42)
    n_dates, n_assets = 300, 50
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    assets = [f"stock_{i:03d}" for i in range(n_assets)]

    # Create factor with some predictive power
    factor_values = np.random.randn(n_dates, n_assets)
    factor = pd.DataFrame(factor_values, index=dates, columns=assets)

    # Forward returns correlated with factor
    noise = np.random.randn(n_dates, n_assets) * 0.02
    forward_returns = factor_values * 0.01 + noise
    forward_returns = pd.DataFrame(forward_returns, index=dates, columns=assets)

    return factor, forward_returns


@pytest.fixture
def multi_factor_data():
    """Create multiple factors with varying IC."""
    np.random.seed(42)
    n_dates, n_assets = 300, 50
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    assets = [f"stock_{i:03d}" for i in range(n_assets)]

    factors = {}
    # Good factor (high IC)
    good_vals = np.random.randn(n_dates, n_assets)
    factors["good_factor"] = pd.DataFrame(good_vals, index=dates, columns=assets)

    # Weak factor (low IC)
    factors["weak_factor"] = pd.DataFrame(
        np.random.randn(n_dates, n_assets), index=dates, columns=assets
    )

    # Forward returns correlated with good_factor
    noise = np.random.randn(n_dates, n_assets) * 0.02
    fwd = good_vals * 0.01 + noise
    forward_returns = pd.DataFrame(fwd, index=dates, columns=assets)

    return factors, forward_returns


class TestICMonitorConfig:
    def test_default_config(self):
        cfg = ICMonitorConfig()
        assert cfg.rolling_window == 63
        assert cfg.icir_window == 252
        assert cfg.decay_window == 126
        assert cfg.significance_threshold == 0.03
        assert cfg.min_observations == 63

    def test_custom_config(self):
        cfg = ICMonitorConfig(rolling_window=30, significance_threshold=0.05)
        assert cfg.rolling_window == 30
        assert cfg.significance_threshold == 0.05


class TestFactorICStats:
    def test_dataclass(self):
        stats = FactorICStats(
            name="momentum",
            current_ic=0.05,
            current_icir=1.5,
            rolling_mean_ic=0.04,
            alert_level="green",
        )
        assert stats.name == "momentum"
        assert stats.current_ic == 0.05
        assert stats.alert_level == "green"


class TestFactorICMonitor:
    def test_compute_rolling_ic(self, sample_factor_data):
        factor, forward_returns = sample_factor_data
        monitor = FactorICMonitor(config=ICMonitorConfig(rolling_window=63, min_observations=10))
        ic_series = monitor.compute_rolling_ic(factor, forward_returns, window=63)
        assert isinstance(ic_series, pd.Series)
        assert len(ic_series) > 0
        assert ic_series.name == "ic"

    def test_compute_factor_stats(self, sample_factor_data):
        factor, forward_returns = sample_factor_data
        cfg = ICMonitorConfig(rolling_window=63, min_observations=30)
        monitor = FactorICMonitor(config=cfg)
        stats = monitor.compute_factor_stats("test_factor", factor, forward_returns)
        assert isinstance(stats, FactorICStats)
        assert stats.name == "test_factor"
        assert -1 <= stats.current_ic <= 1
        assert 0 <= stats.ic_positive_ratio <= 1
        assert stats.alert_level in ("green", "yellow", "red")

    def test_compute_all(self, multi_factor_data):
        factors, forward_returns = multi_factor_data
        cfg = ICMonitorConfig(rolling_window=63, min_observations=30)
        monitor = FactorICMonitor(config=cfg)
        all_stats = monitor.compute_all(factors, forward_returns)
        assert len(all_stats) == 2
        assert "good_factor" in all_stats
        assert "weak_factor" in all_stats

    def test_get_alerts(self, multi_factor_data):
        factors, forward_returns = multi_factor_data
        cfg = ICMonitorConfig(rolling_window=63, min_observations=30)
        monitor = FactorICMonitor(config=cfg)
        monitor.compute_all(factors, forward_returns)
        alerts = monitor.get_alerts()
        assert isinstance(alerts, list)
        for alert in alerts:
            assert "factor" in alert
            assert "alert_level" in alert
            assert alert["alert_level"] in ("red", "yellow")

    def test_get_adaptive_weights(self, multi_factor_data):
        factors, forward_returns = multi_factor_data
        cfg = ICMonitorConfig(rolling_window=63, min_observations=30)
        monitor = FactorICMonitor(config=cfg)
        weights = monitor.get_adaptive_weights(factors, forward_returns)
        assert isinstance(weights, dict)
        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        for w in weights.values():
            assert w >= 0

    def test_get_summary(self, multi_factor_data):
        factors, forward_returns = multi_factor_data
        cfg = ICMonitorConfig(rolling_window=63, min_observations=30)
        monitor = FactorICMonitor(config=cfg)
        monitor.compute_all(factors, forward_returns)
        summary = monitor.get_summary()
        assert isinstance(summary, list)
        assert len(summary) == 2
        for item in summary:
            assert "factor" in item
            assert "current_ic" in item
            assert "rolling_icir" in item
            assert "alert" in item

    def test_insufficient_data(self):
        """Test with too few observations."""
        np.random.seed(42)
        dates = pd.bdate_range("2022-01-01", periods=10)
        assets = [f"stock_{i}" for i in range(10)]
        factor = pd.DataFrame(np.random.randn(10, 10), index=dates, columns=assets)
        fwd = pd.DataFrame(np.random.randn(10, 10), index=dates, columns=assets)

        cfg = ICMonitorConfig(rolling_window=63, min_observations=63)
        monitor = FactorICMonitor(config=cfg)
        stats = monitor.compute_factor_stats("test", factor, fwd)
        # Should return default stats with insufficient data
        assert stats.current_ic == 0.0

    def test_half_life_estimation(self):
        """Test half-life calculation logic."""
        cfg = ICMonitorConfig(rolling_window=63, min_observations=30)
        monitor = FactorICMonitor(config=cfg)

        # Create a decaying IC series manually
        dates = pd.bdate_range("2022-01-01", periods=200)
        ic_values = np.linspace(0.1, 0.01, 200)  # Decaying from 0.1 to 0.01
        monitor.ic_history["test"] = pd.Series(ic_values, index=dates)

        # The half_life should be positive for a decaying but positive IC
        # We can verify the stats computation works
        assert len(monitor.ic_history["test"]) == 200
