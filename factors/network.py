"""Graph-based network factor using stock correlation networks.

Builds a stock correlation/return network and extracts centrality measures
as alpha signals. Stocks that are more central in the correlation network
tend to have different risk/return characteristics.

Network construction:
1. Build correlation matrix from rolling returns
2. Threshold edges (|corr| > threshold) to create sparse graph
3. Compute centrality measures per stock per date
4. Use centrality as a factor value

Centrality measures:
- Degree centrality: number of strong correlations
- Eigenvector centrality: connected to other well-connected stocks
- Betweenness centrality: bridges between clusters (sector connectors)
- PageRank: importance in the network

Economic intuition:
- High degree centrality → "market representative" stocks (large-cap, blue chip)
- High betweenness → sector connectors, may lead sector rotations
- High PageRank → systemically important stocks

Reference: "Network Centrality and Stock Returns" (Fricke & Fushch, 2015)

Usage:
    from quant_platform.factors.network import NetworkFactor

    nf = NetworkFactor(centrality_type="eigenvector", window=60, threshold=0.5)
    factor_values = nf.compute(prices)  # Returns (date x asset) DataFrame
"""

from __future__ import annotations

import warnings
from enum import Enum

import numpy as np
import pandas as pd

from quant_platform.factors.base import BaseFactor, FactorCategory
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class CentralityType(str, Enum):
    DEGREE = "degree"
    EIGENVECTOR = "eigenvector"
    BETWEENNESS = "betweenness"
    PAGERANK = "pagerank"


class NetworkFactor(BaseFactor):
    """Graph-based network centrality factor.

    Constructs a stock correlation network from rolling returns and
    computes centrality measures as alpha signals.

    Args:
        centrality_type: Which centrality measure to use
        window: Rolling window for correlation estimation (days)
        threshold: Minimum |correlation| to create an edge
        damping: Damping factor for PageRank (default 0.85)
        name: Factor name
    """

    category = FactorCategory.CUSTOM

    def __init__(
        self,
        centrality_type: str = "eigenvector",
        window: int = 60,
        threshold: float = 0.5,
        damping: float = 0.85,
        name: str | None = None,
    ):
        self._centrality_type = CentralityType(centrality_type)
        self._window = window
        self._threshold = threshold
        self._damping = damping
        self._name = name or f"network_{centrality_type}_{window}d"

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        prices: pd.DataFrame,
        financials: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute network centrality factor.

        Args:
            prices: (date x asset) price DataFrame
            financials: unused, kept for interface compatibility

        Returns:
            (date x asset) DataFrame of centrality values
        """
        # Compute daily returns
        returns = prices.pct_change().dropna(how="all")

        if len(returns) < self._window + 10:
            logger.warning("Not enough data for network factor: %d < %d", len(returns), self._window)
            return pd.DataFrame(0, index=prices.index, columns=prices.columns)

        assets = returns.columns
        dates = returns.index
        n_dates = len(dates)
        n_assets = len(assets)

        result = np.full((n_dates, n_assets), np.nan)

        for i in range(self._window, n_dates):
            window_returns = returns.iloc[i - self._window:i]

            # Compute correlation matrix
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr = window_returns.corr()

            # Build adjacency matrix (threshold)
            adj = corr.abs().values.copy()
            np.fill_diagonal(adj, 0)
            adj[adj < self._threshold] = 0
            adj[adj >= self._threshold] = 1

            # Compute centrality
            centrality = self._compute_centrality(adj)

            # Map back to asset order
            for j, asset in enumerate(assets):
                if j < len(centrality):
                    result[i, j] = centrality[j]

        # Forward fill initial NaN rows
        df = pd.DataFrame(result, index=dates, columns=assets)
        df = df.ffill().fillna(0)

        # Align to original prices index
        df = df.reindex(prices.index).ffill().fillna(0)

        return df

    def _compute_centrality(self, adj: np.ndarray) -> np.ndarray:
        """Compute centrality measure for the adjacency matrix."""
        n = adj.shape[0]
        if n == 0:
            return np.array([])

        if self._centrality_type == CentralityType.DEGREE:
            return self._degree_centrality(adj)
        elif self._centrality_type == CentralityType.EIGENVECTOR:
            return self._eigenvector_centrality(adj)
        elif self._centrality_type == CentralityType.BETWEENNESS:
            return self._betweenness_centrality(adj)
        elif self._centrality_type == CentralityType.PAGERANK:
            return self._pagerank(adj)
        else:
            return self._degree_centrality(adj)

    @staticmethod
    def _degree_centrality(adj: np.ndarray) -> np.ndarray:
        """Degree centrality: fraction of nodes connected to."""
        n = adj.shape[0]
        degrees = adj.sum(axis=1)
        return degrees / max(n - 1, 1)

    @staticmethod
    def _eigenvector_centrality(adj: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Eigenvector centrality via power iteration.

        The leading eigenvector of the adjacency matrix.
        """
        n = adj.shape[0]
        if n == 0:
            return np.array([])

        # Power iteration
        v = np.ones(n) / n
        for _ in range(max_iter):
            v_new = adj @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-10:
                return np.ones(n) / n
            v_new = v_new / norm
            if np.linalg.norm(v_new - v) < tol:
                break
            v = v_new

        # Normalize to [0, 1]
        v = np.abs(v)
        vmax = v.max()
        if vmax > 0:
            v = v / vmax
        return v

    @staticmethod
    def _betweenness_centrality(adj: np.ndarray) -> np.ndarray:
        """Approximate betweenness centrality using BFS.

        Betweenness = fraction of shortest paths passing through each node.
        Uses BFS from each node (unweighted graph).
        """
        n = adj.shape[0]
        if n <= 2:
            return np.ones(n) / n

        centrality = np.zeros(n)

        for s in range(n):
            # BFS from node s
            stack = []
            predecessors = [[] for _ in range(n)]
            sigma = np.zeros(n)
            sigma[s] = 1
            distance = np.full(n, -1)
            distance[s] = 0
            queue = [s]

            while queue:
                v = queue.pop(0)
                stack.append(v)
                for w in range(n):
                    if adj[v, w] > 0:
                        if distance[w] < 0:
                            distance[w] = distance[v] + 1
                            queue.append(w)
                        if distance[w] == distance[v] + 1:
                            sigma[w] += sigma[v]
                            predecessors[w].append(v)

            # Back-propagation
            delta = np.zeros(n)
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    centrality[w] += delta[w]

        # Normalize
        norm = (n - 1) * (n - 2) / 2
        if norm > 0:
            centrality = centrality / norm
        return centrality

    def _pagerank(self, adj: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """PageRank centrality.

        Google PageRank adapted for stock networks.
        """
        n = adj.shape[0]
        if n == 0:
            return np.array([])

        # Row-normalize adjacency
        row_sums = adj.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        M = adj / row_sums[:, np.newaxis]

        d = self._damping
        pr = np.ones(n) / n

        for _ in range(max_iter):
            pr_new = (1 - d) / n + d * M.T @ pr
            if np.linalg.norm(pr_new - pr) < tol:
                break
            pr = pr_new

        # Normalize to [0, 1]
        prmax = pr.max()
        if prmax > 0:
            pr = pr / prmax
        return pr

    def get_adjacency_matrix(
        self,
        returns: pd.DataFrame,
        date: str | None = None,
    ) -> pd.DataFrame:
        """Get the correlation adjacency matrix for a specific date.

        Useful for visualization and debugging.

        Args:
            returns: (date x asset) return DataFrame
            date: specific date (default: last)

        Returns:
            (asset x asset) binary adjacency DataFrame
        """
        if date:
            idx = returns.index.get_loc(date)
            window = returns.iloc[max(0, idx - self._window):idx]
        else:
            window = returns.iloc[-self._window:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = window.corr()

        adj = corr.abs().copy()
        for i in range(len(adj)):
            adj.iloc[i, i] = 0
        adj[adj < self._threshold] = 0
        adj[adj >= self._threshold] = 1

        return adj

    def get_network_stats(self, returns: pd.DataFrame) -> dict:
        """Get network statistics for the latest date.

        Returns:
            dict with n_edges, density, avg_degree, n_components
        """
        adj_df = self.get_adjacency_matrix(returns)
        adj = adj_df.values
        n = adj.shape[0]
        n_edges = int(adj.sum() / 2)  # Undirected
        max_edges = n * (n - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0
        avg_degree = adj.sum(axis=1).mean()

        # Count connected components (BFS)
        visited = set()
        n_components = 0
        for i in range(n):
            if i not in visited:
                n_components += 1
                queue = [i]
                while queue:
                    v = queue.pop(0)
                    if v in visited:
                        continue
                    visited.add(v)
                    for j in range(n):
                        if adj[v, j] > 0 and j not in visited:
                            queue.append(j)

        return {
            "n_nodes": n,
            "n_edges": n_edges,
            "density": round(density, 4),
            "avg_degree": round(float(avg_degree), 2),
            "n_components": n_components,
            "threshold": self._threshold,
        }
