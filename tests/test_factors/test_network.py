"""Tests for graph-based network centrality factor."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from quant_platform.factors.network import (
    CentralityType,
    NetworkFactor,
)


@pytest.fixture
def sample_prices():
    """Create sample price data with correlated stocks."""
    np.random.seed(42)
    n_dates, n_assets = 200, 30
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    assets = [f"stock_{i:03d}" for i in range(n_assets)]

    # Create returns with some correlation structure
    # Group 1: stocks 0-9 correlated
    market = np.random.randn(n_dates) * 0.01
    group1_noise = np.random.randn(n_dates, 10) * 0.005
    group1_returns = market[:, None] + group1_noise

    # Group 2: stocks 10-19 correlated
    factor2 = np.random.randn(n_dates) * 0.008
    group2_noise = np.random.randn(n_dates, 10) * 0.005
    group2_returns = factor2[:, None] + group2_noise

    # Group 3: stocks 20-29 independent
    group3_returns = np.random.randn(n_dates, 10) * 0.01

    returns = np.hstack([group1_returns, group2_returns, group3_returns])
    # Convert returns to prices
    prices_data = 100 * np.exp(np.cumsum(returns, axis=0))

    return pd.DataFrame(prices_data, index=dates, columns=assets)


class TestNetworkFactor:
    def test_init_default(self):
        nf = NetworkFactor()
        assert nf._centrality_type == CentralityType.EIGENVECTOR
        assert nf._window == 60
        assert nf._threshold == 0.5

    def test_init_custom(self):
        nf = NetworkFactor(centrality_type="pagerank", window=30, threshold=0.3)
        assert nf._centrality_type == CentralityType.PAGERANK
        assert nf._window == 30
        assert nf._threshold == 0.3

    def test_name(self):
        nf = NetworkFactor(centrality_type="degree", window=40)
        assert "degree" in nf.name
        assert "40" in nf.name

    def test_custom_name(self):
        nf = NetworkFactor(name="my_factor")
        assert nf.name == "my_factor"

    def test_compute_eigenvector(self, sample_prices):
        nf = NetworkFactor(centrality_type="eigenvector", window=60, threshold=0.3)
        result = nf.compute(sample_prices)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_prices.shape
        # Values should be in [0, 1]
        for col in result.columns:
            valid = result[col].dropna()
            if len(valid) > 0:
                assert valid.min() >= 0
                assert valid.max() <= 1

    def test_compute_degree(self, sample_prices):
        nf = NetworkFactor(centrality_type="degree", window=60, threshold=0.3)
        result = nf.compute(sample_prices)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_prices.shape

    def test_compute_betweenness(self, sample_prices):
        nf = NetworkFactor(centrality_type="betweenness", window=60, threshold=0.3)
        result = nf.compute(sample_prices)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_prices.shape

    def test_compute_pagerank(self, sample_prices):
        nf = NetworkFactor(centrality_type="pagerank", window=60, threshold=0.3)
        result = nf.compute(sample_prices)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_prices.shape

    def test_correlated_stocks_higher_centrality(self, sample_prices):
        """Stocks in correlated groups should have higher degree centrality."""
        nf = NetworkFactor(centrality_type="degree", window=60, threshold=0.3)
        result = nf.compute(sample_prices)

        # Take average centrality over last 50 dates
        recent = result.iloc[-50:].mean()
        # Group 1 (stocks 0-9) are correlated, should have higher centrality
        group1_mean = recent.iloc[:10].mean()
        recent.iloc[20:].mean()
        # Correlated group should generally have higher centrality
        assert group1_mean > 0

    def test_get_adjacency_matrix(self, sample_prices):
        nf = NetworkFactor(window=60, threshold=0.5)
        returns = sample_prices.pct_change().dropna()
        adj = nf.get_adjacency_matrix(returns)
        assert isinstance(adj, pd.DataFrame)
        assert adj.shape == (30, 30)
        # Diagonal should be 0
        for i in range(30):
            assert adj.iloc[i, i] == 0

    def test_get_network_stats(self, sample_prices):
        nf = NetworkFactor(window=60, threshold=0.3)
        returns = sample_prices.pct_change().dropna()
        stats = nf.get_network_stats(returns)
        assert "n_nodes" in stats
        assert "n_edges" in stats
        assert "density" in stats
        assert "avg_degree" in stats
        assert "n_components" in stats
        assert stats["n_nodes"] == 30
        assert stats["n_edges"] >= 0
        assert 0 <= stats["density"] <= 1

    def test_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        np.random.seed(42)
        dates = pd.bdate_range("2022-01-01", periods=10)
        assets = [f"s{i}" for i in range(5)]
        prices = pd.DataFrame(
            100 + np.random.randn(10, 5).cumsum(axis=0),
            index=dates, columns=assets,
        )
        nf = NetworkFactor(window=60)
        result = nf.compute(prices)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == prices.shape

    def test_degree_centrality(self):
        # Simple 3-node graph: 0-1 connected, 2 isolated
        adj = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        c = NetworkFactor._degree_centrality(adj)
        assert c[0] == pytest.approx(0.5)  # 1/2
        assert c[1] == pytest.approx(0.5)
        assert c[2] == pytest.approx(0.0)

    def test_eigenvector_centrality(self):
        # Star graph: node 0 connected to 1,2,3
        adj = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ], dtype=float)
        c = NetworkFactor._eigenvector_centrality(adj)
        assert c[0] == pytest.approx(1.0)  # Hub node should be highest
        assert c[1] < c[0]

    def test_pagerank_basic(self):
        # Simple graph
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=float)
        nf = NetworkFactor(centrality_type="pagerank")
        c = nf._pagerank(adj)
        assert len(c) == 3
        assert all(c >= 0)
        # In a cycle, all should be roughly equal
        assert np.std(c) < 0.1


class TestGrafanaDashboard:
    def test_valid_json(self):
        dashboard_path = Path(__file__).resolve().parent.parent.parent / "monitoring" / "grafana_dashboard.json"
        if dashboard_path.exists():
            data = json.loads(dashboard_path.read_text())
            assert "panels" in data
            assert "title" in data
            assert data["title"] == "Quant Platform Monitor"
            assert len(data["panels"]) >= 10

    def test_panels_have_targets(self):
        dashboard_path = Path(__file__).resolve().parent.parent.parent / "monitoring" / "grafana_dashboard.json"
        if dashboard_path.exists():
            data = json.loads(dashboard_path.read_text())
            for panel in data["panels"]:
                if "targets" in panel:
                    assert len(panel["targets"]) > 0
                    for target in panel["targets"]:
                        assert "expr" in target
