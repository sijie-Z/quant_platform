"""Synthetic data provider for A-share market.

Generates realistic synthetic data for ~500 stocks over 5 years, including:
- Daily OHLCV with correlated returns (market + sector + idiosyncratic)
- Quarterly financial statements
- Stock metadata (sector, market cap group, ST status)
- Suspension periods and price limits

All randomness is seeded for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.data.providers.base import DataProvider
from quant_platform.data.schema import SECTORS, SECTOR_WEIGHTS
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class SyntheticDataProvider(DataProvider):
    """Generates a complete synthetic A-share dataset."""

    def __init__(
        self,
        n_stocks: int = 500,
        start_date: str = "2021-01-01",
        end_date: str = "2025-12-31",
        seed: int = 42,
        market_drift: float = 0.0003,    # ~8% annual return
        market_vol: float = 0.012,        # ~19% annual vol
        sector_vol_scale: float = 0.3,
        idio_vol_scale: float = 0.7,
    ):
        self.n_stocks = n_stocks
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.market_drift = market_drift
        self.market_vol = market_vol
        self.sector_vol_scale = sector_vol_scale
        self.idio_vol_scale = idio_vol_scale

        # Internal caches
        self._prices: pd.DataFrame | None = None
        self._financials: pd.DataFrame | None = None
        self._benchmark: pd.Series | None = None
        self._metadata: pd.DataFrame | None = None

        # Derived data
        self._dates: pd.DatetimeIndex | None = None
        self._assets: list[str] | None = None
        self._sector_map: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Public interface (DataProvider contract)
    # ------------------------------------------------------------------

    def get_prices(
        self,
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        if self._prices is None:
            self._generate_all()
        df = self._prices.loc[
            pd.Timestamp(start_date):pd.Timestamp(end_date)
        ]
        if fields:
            df = df[fields]
        return df

    def get_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if self._financials is None:
            self._generate_all()
        return self._financials.loc[
            pd.Timestamp(start_date):pd.Timestamp(end_date)
        ]

    def get_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        if self._benchmark is None:
            self._generate_all()
        return self._benchmark.loc[
            pd.Timestamp(start_date):pd.Timestamp(end_date)
        ]

    def get_metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            self._generate_all()
        return self._metadata

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_all(self) -> None:
        logger.info("Generating synthetic A-share data: %d stocks, %s to %s",
                     self.n_stocks, self.start_date.date(), self.end_date.date())

        self._dates = pd.bdate_range(self.start_date, self.end_date, name="date")
        self._assets = [f"{i:06d}.SH" if i % 3 != 0 else f"{i:06d}.SZ"
                        for i in range(1, self.n_stocks + 1)]

        self._metadata = self._generate_metadata()
        self._sector_map = self._metadata["sector"].to_dict()

        returns = self._generate_returns()
        self._prices = self._build_price_data(returns)
        self._benchmark = self._generate_benchmark()
        self._financials = self._generate_financials()

        logger.info("Synthetic data generation complete: %d dates, %d assets",
                     len(self._dates), len(self._assets))

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _generate_metadata(self) -> pd.DataFrame:
        """Generate stock metadata: sector, listing date, ST status, etc."""
        n = self.n_stocks
        sectors_list = list(SECTOR_WEIGHTS.keys())
        weights = list(SECTOR_WEIGHTS.values())
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        sectors = self.rng.choice(sectors_list, size=n, p=weights)

        # Market cap groups: 10% large, 30% mid, 60% small
        cap_groups = self.rng.choice(
            ["large", "mid", "small"], size=n,
            p=[0.10, 0.30, 0.60],
        )

        # Listing dates: some stocks list during the period
        listing_dates = pd.Series(self.start_date, index=range(n))
        late_listers = self.rng.choice(n, size=int(n * 0.15), replace=False)
        for i in late_listers:
            offset = self.rng.integers(30, len(self._dates) // 2)
            listing_dates.iloc[i] = self._dates[offset]

        # ST status (~3% of stocks, random assignment)
        is_st = self.rng.choice([True, False], size=n, p=[0.03, 0.97])

        return pd.DataFrame({
            "sector": sectors,
            "market_cap_group": cap_groups,
            "is_st": is_st,
            "listing_date": listing_dates.values,
            "delisting_date": pd.NaT,
        }, index=pd.Index(self._assets, name="asset"))

    # ------------------------------------------------------------------
    # Returns generation (factor model)
    # ------------------------------------------------------------------

    def _generate_returns(self) -> pd.DataFrame:
        """Generate daily returns using a 3-factor decomposition with embedded alpha.

        r_i,t = market_t + sector_sector(t),t + idio_i,t + alpha_i,t

        The alpha component embeds realistic cross-sectional return predictability:
        - Short-term momentum (t-21:t-1 return)
        - Long-term reversal (t-252:t-21 return)
        - Value (inverse PB ratio)
        - Size (inverse market cap)

        Effects are calibrated to realistic A-share IC levels (~0.02-0.04).
        This ensures the factor evaluation and backtest show meaningful results
        rather than near-zero ICs from pure noise.
        """
        n_dates = len(self._dates)
        n_assets = len(self._assets)

        # 1) Market factor: random walk with drift
        market_shocks = self.rng.normal(0, 1, size=n_dates)
        market_returns = self.market_drift + self.market_vol * market_shocks

        # 2) Sector factors (one per sector)
        unique_sectors = list(SECTORS)
        n_sectors = len(unique_sectors)
        sector_shocks = self.rng.normal(0, 1, size=(n_dates, n_sectors))
        sector_returns_raw = self.sector_vol_scale * self.market_vol * sector_shocks

        sector_idx = {s: i for i, s in enumerate(unique_sectors)}
        asset_sector_idx = np.array([
            sector_idx[self._sector_map[a]] for a in self._assets
        ])

        # 3) Idiosyncratic returns
        idio_vol = self.idio_vol_scale * self.market_vol * np.sqrt(2)
        idio_returns = self.rng.normal(0, idio_vol, size=(n_dates, n_assets))

        # Combine base returns
        sector_returns_per_asset = sector_returns_raw[:, asset_sector_idx]
        base_returns = (
            market_returns[:, np.newaxis]
            + sector_returns_per_asset
            + idio_returns
        )

        # --- Embedded alpha: create predictive structure ---
        # Asset-level characteristics (static for simplicity)
        base_pb = np.clip(self.rng.lognormal(np.log(2.0), 0.6, size=n_assets), 0.5, 10.0)
        base_mcap = self.rng.lognormal(np.log(5e9), 0.8, size=n_assets)

        # Value signal: low PB = higher expected return (inverse-PB rank)
        value_signal = -np.log(base_pb)  # higher for cheap stocks
        value_signal = (value_signal - value_signal.mean()) / value_signal.std()

        # Size signal: small cap = higher expected return
        size_signal = -np.log(base_mcap)
        size_signal = (size_signal - size_signal.mean()) / size_signal.std()

        # Build cumulative returns iteratively, adding momentum alpha
        # We simulate prices alongside returns to compute rolling momentum
        cumulative_base = np.zeros((n_dates, n_assets))
        cumulative_base[0] = base_returns[0]
        for t in range(1, n_dates):
            cumulative_base[t] = cumulative_base[t - 1] + base_returns[t]

        # Compute past 21-day return (momentum) at each date
        momentum_1m = np.zeros((n_dates, n_assets))
        for t in range(21, n_dates):
            momentum_1m[t] = cumulative_base[t - 1] - cumulative_base[t - 21]

        # Normalize momentum cross-sectionally
        for t in range(21, n_dates):
            row = momentum_1m[t]
            valid = ~np.isnan(row)
            if valid.sum() < 30:
                continue
            mu, std = row[valid].mean(), row[valid].std()
            if std > 1e-10:
                momentum_1m[t] = (row - mu) / std

        # Alpha return = weighted combination of signals
        # Target IC ~0.015-0.02: realistic A-share levels.
        # Momentum is the dominant signal; value/size are weaker.
        # Signal-to-noise ~1:2 — real markets have more noise than signal.
        alpha_return = np.zeros((n_dates, n_assets))

        for t in range(21, n_dates):
            # Momentum alpha (moderate, realistic A-share level)
            alpha_return[t] += 0.015 * momentum_1m[t]

            # Value alpha 
            value_weight = min(1.0, t / 63)
            alpha_return[t] += 0.008 * value_signal * value_weight

            # Size alpha
            alpha_return[t] += 0.005 * size_signal * value_weight

        # Scale to ~0.0008/day cross-sectional std → IC ~0.04-0.05
        raw_std = np.nanstd(alpha_return[21:])
        if raw_std > 1e-10:
            alpha_return *= 0.0003 / raw_std

        # Add modest noise (signal-to-noise ~2:1)
        alpha_noise = self.rng.normal(0, 0.0006, size=(n_dates, n_assets))
        alpha_return += alpha_noise

        total_returns = base_returns + alpha_return

        # Apply price limits (±10%) — truncate returns
        total_returns = np.clip(total_returns, -0.10, 0.10)

        return pd.DataFrame(
            total_returns,
            index=self._dates,
            columns=self._assets,
        )

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    def _build_price_data(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Build full OHLCV data from returns.

        Each stock starts at a random price between 5 and 200 CNY.
        Close = prior_close * (1 + return)
        Open, High, Low derived with intraday randomness.
        Volume scales with absolute return (higher vol -> more volume).
        """
        n_dates = len(self._dates)
        n_assets = len(self._assets)

        # Initial prices: log-uniform between 5 and 200
        init_prices = np.exp(self.rng.uniform(np.log(5), np.log(200), size=n_assets))

        # Close prices: cumulative product of (1 + r)
        close_prices = np.zeros((n_dates, n_assets))
        close_prices[0] = init_prices * (1 + returns.iloc[0].values)

        for t in range(1, n_dates):
            # Handle listing: stocks not yet listed have NaN
            mask = self._dates[t] >= self._metadata["listing_date"].values
            close_prices[t] = close_prices[t - 1] * (1 + returns.iloc[t].values)
            close_prices[t, ~mask] = np.nan

        # Open, High, Low: intraday randomness
        intraday_range = self.rng.uniform(0.005, 0.03, size=(n_dates, n_assets))
        open_ratio = 1 + self.rng.normal(0, 0.005, size=(n_dates, n_assets))

        open_prices = close_prices * open_ratio
        # For t>0, open is based on prior close with gap
        for t in range(1, n_dates):
            open_prices[t] = close_prices[t - 1] * open_ratio[t]

        high_prices = np.maximum(open_prices, close_prices) * (1 + intraday_range / 2)
        low_prices = np.minimum(open_prices, close_prices) * (1 - intraday_range / 2)

        # Volume: base + extra volume on volatile days
        base_volume = self.rng.lognormal(15, 0.8, size=n_assets)  # ~3M shares
        abs_ret = np.abs(returns.values)
        vol_multiplier = 1 + 5 * abs_ret
        volume = base_volume[np.newaxis, :] * vol_multiplier * self.rng.uniform(0.5, 1.5, size=(n_dates, n_assets))
        volume = np.round(volume).astype(int)

        # Turnover rate: volume / shares_outstanding
        shares_outstanding = base_volume * self.rng.uniform(100, 500)
        turnover = volume / shares_outstanding[np.newaxis, :]
        turnover = np.clip(turnover, 0.001, 0.15)

        # VWAP: average of OHLC
        vwap = (open_prices + high_prices + low_prices + close_prices) / 4

        # Adj factor (random corporate actions: ~5% of stocks per year)
        adj_factor = np.ones((n_dates, n_assets))
        for i in range(n_assets):
            n_actions = self.rng.poisson(0.05 * n_dates / 252)
            if n_actions > 0:
                action_dates = self.rng.choice(n_dates, size=n_actions, replace=False)
                action_dates.sort()
                factor = 1.0
                for d in range(n_dates):
                    if d in action_dates:
                        factor *= self.rng.uniform(0.8, 1.2)
                    adj_factor[d, i] = factor

        # Build MultiIndex DataFrame
        arrays = []
        for i, asset in enumerate(self._assets):
            for d in range(n_dates):
                arrays.append([
                    self._dates[d], asset,
                    open_prices[d, i], high_prices[d, i],
                    low_prices[d, i], close_prices[d, i],
                    volume[d, i], turnover[d, i],
                    adj_factor[d, i], vwap[d, i],
                ])

        df = pd.DataFrame(
            arrays,
            columns=["date", "asset", "open", "high", "low", "close",
                     "volume", "turnover", "adj_factor", "vwap"],
        )
        df = df.set_index(["date", "asset"]).sort_index()

        # Apply suspension: ~2% of stock-days are suspended (NaN prices)
        n_total = n_dates * n_assets
        suspended = self.rng.choice(n_total, size=int(n_total * 0.02), replace=False)
        # We do this via the underlying arrays for efficiency
        flat_idx = np.array(suspended)
        for col in ["open", "high", "low", "close", "volume", "turnover", "vwap"]:
            df.iloc[flat_idx, df.columns.get_loc(col)] = np.nan

        # Forward-fill suspension gaps (max 30 days)
        df = df.groupby("asset").ffill(limit=30)

        return df

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def _generate_benchmark(self) -> pd.Series:
        """Generate benchmark returns (market-cap-weighted index proxy)."""
        n_dates = len(self._dates)
        # Smoother than individual stocks — ~10% annual return, ~16% vol
        bench_returns = self.rng.normal(0.0004, 0.010, size=n_dates)
        return pd.Series(bench_returns, index=self._dates, name="benchmark")

    # ------------------------------------------------------------------
    # Financial data
    # ------------------------------------------------------------------

    def _generate_financials(self) -> pd.DataFrame:
        """Generate quarterly financial statement data.

        Reported at quarter-end dates and forward-filled to daily frequency.
        Each stock gets realistic financial ratios.
        """
        n_assets = len(self._assets)

        # Find quarter-end dates in our date range
        all_dates = pd.DatetimeIndex(self._dates)
        quarter_ends = all_dates[all_dates.is_quarter_end]
        n_quarters = len(quarter_ends)

        # Base financial characteristics (cross-sectional)
        # Market cap: log-normal, 1B to 500B CNY
        base_market_cap = self.rng.lognormal(np.log(5e9), 0.8, size=n_assets)

        # PB ratio: 0.5 to 10
        base_pb = self.rng.lognormal(np.log(2.0), 0.6, size=n_assets)
        base_pb = np.clip(base_pb, 0.5, 10.0)

        # PE ratio: 5 to 100
        base_pe = self.rng.lognormal(np.log(25), 0.7, size=n_assets)
        base_pe = np.clip(base_pe, 5, 100)

        # ROE: -20% to 30%
        base_roe = self.rng.normal(0.08, 0.08, size=n_assets)
        base_roe = np.clip(base_roe, -0.20, 0.30)

        # Asset growth: -10% to 30% annually
        base_asset_growth = self.rng.normal(0.10, 0.10, size=n_assets)
        base_asset_growth = np.clip(base_asset_growth, -0.10, 0.30)

        rows = []
        for qi, qdate in enumerate(quarter_ends):
            # Time-varying component: mean-reverting random walk around base
            t_factor = qi / max(n_quarters - 1, 1)
            cycle_factor = 0.05 * np.sin(t_factor * 2 * np.pi * 2)  # 2 cycles

            market_cap_t = base_market_cap * (
                1 + cycle_factor + self.rng.normal(0, 0.05, size=n_assets)
            )
            market_cap_t = np.maximum(market_cap_t, 5e8)

            pb_t = base_pb * (1 + self.rng.normal(0, 0.03, size=n_assets))
            pb_t = np.clip(pb_t, 0.3, 15)

            pe_t = base_pe * (1 + self.rng.normal(0, 0.05, size=n_assets))
            pe_t = np.clip(pe_t, 3, 200)

            roe_t = base_roe + self.rng.normal(0, 0.02, size=n_assets)
            roe_t = np.clip(roe_t, -0.30, 0.40)

            asset_growth_t = base_asset_growth + self.rng.normal(0, 0.03, size=n_assets)
            asset_growth_t = np.clip(asset_growth_t, -0.15, 0.35)

            total_assets = market_cap_t / np.clip(pb_t, 0.1, None)
            net_assets = total_assets * self.rng.uniform(0.3, 0.7, size=n_assets)
            revenue = market_cap_t * self.rng.uniform(0.1, 1.0, size=n_assets)
            net_profit = roe_t * net_assets

            for i, asset in enumerate(self._assets):
                rows.append([
                    qdate, asset,
                    market_cap_t[i], total_assets[i], net_assets[i],
                    revenue[i], net_profit[i], roe_t[i],
                    pb_t[i], pe_t[i], asset_growth_t[i],
                ])

        df = pd.DataFrame(rows, columns=[
            "date", "asset",
            "market_cap", "total_assets", "net_assets",
            "revenue", "net_profit", "roe",
            "pb_ratio", "pe_ratio", "asset_growth",
        ])
        df = df.set_index(["date", "asset"]).sort_index()

        # Forward-fill to daily frequency by joining with all dates
        # Create empty frame with all date-asset combos
        full_idx = pd.MultiIndex.from_product(
            [all_dates, self._assets], names=["date", "asset"]
        )
        df_full = df.reindex(full_idx)
        df_full = df_full.groupby("asset").ffill()
        df_full = df_full.groupby("asset").bfill()  # Fill initial NaN

        return df_full
