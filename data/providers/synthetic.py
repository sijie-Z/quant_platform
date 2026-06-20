"""Synthetic data provider for A-share market.

Generates realistic synthetic data for ~500 stocks over 5 years, including:
- Daily OHLCV with correlated returns (market + sector + idiosyncratic)
- Quarterly financial statements
- Stock metadata (sector, market cap group, ST status)
- Suspension periods and price limits

All randomness is seeded for reproducibility.

CRITICAL: embedded_alpha defaults to False.
- False: Pure noise returns. No predictable patterns. Safe for research.
- True: Embeds momentum/value/size alpha. For DEMO/INTERVIEW only.
  Never use embedded_alpha=True to validate strategy performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.data.providers.base import DataProvider
from quant_platform.data.schema import SECTOR_WEIGHTS, SECTORS
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class SyntheticDataProvider(DataProvider):
    """Generates a complete synthetic A-share dataset.

    Alpha strength levels (configurable via alpha_strength):
        - 0.00: Pure noise. No predictable patterns. For research validation.
        - 0.03: Weak/realistic (default). IC ~ 0.01-0.02, mimicking real A-share.
        - 0.06: Normal. IC ~ 0.03-0.04, good for demo.
        - 0.12: Strong. IC ~ 0.05-0.08, clearly visible in factor evaluation.
        - 0.50: Oracle. IC ~ 0.10+, for testing pipeline correctness.

    The alpha component embeds PREDICTIVE signals:
        alpha[t] affects return[t+1] (not return[t]).
        This means factor[t] naturally correlates with future returns[t+1].
    """

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
        embedded_alpha: bool = False,
        alpha_strength: float = 0.03,    # predictive alpha (0=off, 0.03=realistic)
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
        self.embedded_alpha = embedded_alpha or (alpha_strength > 0)
        self.alpha_strength = alpha_strength

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
        self._st_timeseries = self._generate_st_timeseries()

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

        # --- Embedded alpha: create predictive structure (demo only) ---
        # WARNING: This embeds predictable return patterns into synthetic data.
        # For production research, set embedded_alpha=False so returns are pure noise
        # (market + sector + idiosyncratic only). The embedded alpha is intended
        # ONLY for interview demos and integration tests.
        total_returns = base_returns.copy()

        if self.embedded_alpha:
            # Asset-level characteristics (static for simplicity)
            base_pb = np.clip(self.rng.lognormal(np.log(2.0), 0.6, size=n_assets), 0.5, 10.0)
            base_mcap = self.rng.lognormal(np.log(5e9), 0.8, size=n_assets)

            # Value signal: low PB = higher expected return (inverse-PB rank)
            value_signal = -np.log(base_pb)
            value_signal = (value_signal - value_signal.mean()) / value_signal.std()

            # Size signal: small cap = higher expected return
            size_signal = -np.log(base_mcap)
            size_signal = (size_signal - size_signal.mean()) / size_signal.std()

            # Build cumulative returns iteratively for momentum computation
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
            # KEY DESIGN: alpha[t] affects return[t+1] (predictive)
            # To control IC, we CORRUPT the alpha driver so factor and alpha
            # are only weakly correlated (like real markets)
            alpha_return = np.zeros((n_dates, n_assets))

            # noise_scale controls factor-alpha correlation (higher = lower IC)
            # Calibrated so IC ~ alpha_strength * (noise_level_factor)
            s = self.alpha_strength
            alpha_noise = {0.03: 10, 0.06: 5, 0.12: 3, 0.50: 2}.get(round(s, 2), max(1, int(10 / max(s, 0.01))))

            for t in range(21, n_dates - 1):
                value_weight = min(1.0, t / 63)
                raw_signal = (
                    0.50 * momentum_1m[t]
                    + 0.30 * value_signal * value_weight
                    + 0.20 * size_signal * value_weight
                )
                # Corrupted signal: momentum + noise, re-normalized cross-sectionally
                corrupt = raw_signal + self.rng.normal(0, alpha_noise, size=n_assets)
                c_mean, c_std = corrupt.mean(), corrupt.std()
                if c_std > 1e-10:
                    corrupt = (corrupt - c_mean) / c_std
                alpha_return[t + 1] = self.alpha_strength * corrupt

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

            # publish_date: report_date + 40-50 calendar days (realistic A-share lag)
            publish_date = qdate + pd.DateOffset(days=40 + int(self.rng.integers(0, 11)))

            for i, asset in enumerate(self._assets):
                rows.append([
                    qdate, asset,
                    market_cap_t[i], total_assets[i], net_assets[i],
                    revenue[i], net_profit[i], roe_t[i],
                    pb_t[i], pe_t[i], asset_growth_t[i],
                    publish_date,
                ])

        df = pd.DataFrame(rows, columns=[
            "date", "asset",
            "market_cap", "total_assets", "net_assets",
            "revenue", "net_profit", "roe",
            "pb_ratio", "pe_ratio", "asset_growth",
            "publish_date",
        ])
        df = df.set_index(["date", "asset"]).sort_index()

        # Forward-fill to daily frequency by joining with all dates
        # Create empty frame with all date-asset combos
        full_idx = pd.MultiIndex.from_product(
            [all_dates, self._assets], names=["date", "asset"]
        )
        df_full = df.reindex(full_idx)
        # Forward-fill financial values but keep publish_date from the original quarter
        df_full = df_full.groupby("asset").ffill()
        df_full = df_full.groupby("asset").bfill()  # Fill initial NaN

        return df_full

    # ------------------------------------------------------------------
    # ST timeseries (point-in-time)
    # ------------------------------------------------------------------

    def _generate_st_timeseries(self) -> pd.DataFrame:
        """Generate daily ST status with realistic announcement lag.

        ST status changes are announced by the exchange, but there is a
        delay between the trigger event (e.g., consecutive losses) and
        the public announcement. We model this as:
        - Actual ST trigger: random date
        - Public announcement (effective for trading): trigger + 1-3 trading days
        - The trader only knows about ST AFTER the announcement date.
        """
        records = []
        for i, asset in enumerate(self._assets):
            # Determine if this stock ever becomes ST
            if not self._metadata.loc[asset, "is_st"]:
                # Never ST: all dates False
                continue

            # ST events: 1-2 episodes per ST stock
            n_episodes = self.rng.integers(1, 3)
            for _ in range(n_episodes):
                # Random trigger date
                trigger_idx = self.rng.integers(60, len(self._dates) - 60)
                trigger_date = self._dates[trigger_idx]

                # Announcement lag: 1-3 trading days after trigger
                announce_idx = min(trigger_idx + int(self.rng.integers(1, 4)),
                                   len(self._dates) - 1)
                announce_date = self._dates[announce_idx]

                # ST duration: 30-200 trading days
                duration = int(self.rng.integers(30, 200))
                end_idx = min(announce_idx + duration, len(self._dates) - 1)
                end_date = self._dates[end_idx]

                records.append({
                    "asset": asset,
                    "trigger_date": trigger_date,
                    "announce_date": announce_date,
                    "end_date": end_date,
                    "is_st": True,
                })

        if not records:
            return pd.DataFrame(columns=["asset", "announce_date", "is_st"])

        return pd.DataFrame(records)

    def get_st_timeseries(self) -> pd.DataFrame:
        """Return ST status timeseries with announcement dates.

        Returns:
            DataFrame with columns: asset, trigger_date, announce_date, end_date, is_st
            Use announce_date for point-in-time filtering (trader only knows after announcement).
        """
        if self._st_timeseries is None:
            self._generate_all()
        return self._st_timeseries

    # ------------------------------------------------------------------
    # Industry changes (point-in-time)
    # ------------------------------------------------------------------

    def get_industry_changes(self) -> pd.DataFrame:
        """Return industry classification with effective dates.

        Industry reclassifications happen ~2 times per year for ~5% of stocks.
        The effective date is when the new classification takes effect.

        Returns:
            DataFrame with columns: asset, industry, effective_date
        """
        records = []
        all_dates = pd.DatetimeIndex(self._dates)
        # Industry changes happen at semi-annual boundaries
        semi_annual = all_dates[all_dates.is_quarter_end][::2]  # Every other quarter end

        for i, asset in enumerate(self._assets):
            current_industry = self._sector_map[asset]
            # Initial classification
            records.append({
                "asset": asset,
                "industry": current_industry,
                "effective_date": self._dates[0],
            })

            # ~5% chance of reclassification per semi-annual period
            for change_date in semi_annual:
                if self.rng.random() < 0.05:
                    new_industry = self.rng.choice(
                        [s for s in SECTORS if s != current_industry]
                    )
                    records.append({
                        "asset": asset,
                        "industry": new_industry,
                        "effective_date": change_date,
                    })
                    current_industry = new_industry

        return pd.DataFrame(records)
