"""Asset-day eligibility regressions (P1-P5).

The pipeline used to carry only an ST mask: fundamental factors bypassed it
(financials were passed through unmasked), mid-sample delistings were only
used to filter the static universe, suspended names kept receiving fresh
signals and target weights, pre-listing assets entered fundamental/turnover
cross-sections, and suspension masking depended on an optional provider
column.  These tests pin the single (date x asset) tradability mask that now
covers all four cases, and the fact that it reaches factors, signals and
optimizer weights.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.base import DataProvider
from quant_platform.factors.fundamental import ROE, LogMarketCap
from quant_platform.factors.processing import process_factor
from quant_platform.factors.technical import Momentum1M
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.portfolio.optimizers import EqualWeightOptimizer

N_DATES = 60
DATES = pd.bdate_range("2024-01-02", periods=N_DATES, name="date")
ASSETS = [f"{i:06d}.SZ" for i in range(1, 13)]


def _random_close(seed: int = 7, drift: float = 0.0004, vol: float = 0.008):
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, vol, (N_DATES, len(ASSETS)))
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(steps, axis=0)), index=DATES, columns=ASSETS
    )


def _alpha_config(method: str = "equal_weight"):
    return types.SimpleNamespace(
        alpha=types.SimpleNamespace(
            method=method,
            lookback=252,
            min_icir=0.0,
            tradability_gate=False,
            min_tradability=0.3,
        )
    )


def _generate_signal(processed_factors, returns, prices):
    from quant_platform import main as cli

    return cli._generate_signal(
        _alpha_config(), processed_factors, returns, prices=prices
    )


class FixedProvider(DataProvider):
    """Deterministic provider with explicit listing/delisting/suspension data.

    Prices are a smooth per-asset path; volume is a constant (never zero) so
    suspension derivation cannot lean on the zero-volume heuristic, and the
    ``is_suspended`` column is optional to model providers that omit it.
    """

    def __init__(
        self,
        close: pd.DataFrame,
        *,
        listing_date: dict | None = None,
        delisting_date: dict | None = None,
        st_episodes=(),
        suspended: pd.DataFrame | None = None,
        flag_suspension: bool = False,
        ffill_limit: int | None = None,
        drop_remaining: bool = False,
    ):
        self._close = close.reindex(index=DATES, columns=ASSETS).astype(float)
        self._suspended = (
            suspended.reindex(index=DATES, columns=ASSETS).fillna(False).astype(bool)
            if suspended is not None
            else pd.DataFrame(False, index=DATES, columns=ASSETS)
        )
        self._listing = listing_date or {}
        self._delisting = delisting_date or {}
        self._st_episodes = list(st_episodes)
        self._flag = flag_suspension
        self._ffill_limit = ffill_limit
        self._drop_remaining = drop_remaining

    # -- provider API ---------------------------------------------------
    def get_metadata(self) -> pd.DataFrame:
        ever_st = {ep["asset"] for ep in self._st_episodes}
        return pd.DataFrame(
            {
                "sector": ["Tech"] * len(ASSETS),
                "market_cap_group": ["mid"] * len(ASSETS),
                "is_st": [a in ever_st for a in ASSETS],
                "listing_date": [
                    pd.Timestamp(self._listing.get(a, pd.Timestamp("2000-01-01")))
                    for a in ASSETS
                ],
                "delisting_date": [
                    pd.Timestamp(self._delisting[a])
                    if a in self._delisting
                    else pd.NaT
                    for a in ASSETS
                ],
            },
            index=pd.Index(ASSETS, name="asset"),
        )

    def get_prices(self, start_date, end_date, fields=None) -> pd.DataFrame:
        close = self._close.mask(self._suspended)
        if self._ffill_limit is not None:
            close = close.ffill(limit=self._ffill_limit)
        if self._drop_remaining:
            still_missing = self._suspended & close.isna()
        else:
            still_missing = pd.DataFrame(False, index=DATES, columns=ASSETS)

        index = pd.MultiIndex.from_product([DATES, ASSETS], names=["date", "asset"])
        values = close.to_numpy()
        df = pd.DataFrame(index=index)
        for col, factor in (
            ("open", 1.0),
            ("high", 1.001),
            ("low", 0.999),
            ("close", 1.0),
            ("vwap", 1.0),
        ):
            df[col] = (values * factor).ravel()
        volume = np.where(np.isnan(values), np.nan, 1_000_000.0)
        df["volume"] = volume.ravel()
        df["turnover"] = np.where(np.isnan(values), np.nan, 0.02).ravel()
        df["adj_factor"] = 1.0
        if self._flag:
            df["is_suspended"] = self._suspended.to_numpy().ravel()
        if self._drop_remaining:
            drop = still_missing.to_numpy().ravel()
            df = df[~drop]
        df = df.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
        if fields:
            df = df[fields]
        return df

    def get_financials(self, start_date, end_date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product([DATES, ASSETS], names=["date", "asset"])
        base_cap = np.linspace(1.0e9, 3.0e9, len(ASSETS))
        out = pd.DataFrame(index=index)
        out["market_cap"] = np.tile(base_cap, N_DATES)
        out["total_assets"] = out["market_cap"] * 2
        out["net_assets"] = out["market_cap"]
        out["revenue"] = out["market_cap"] * 0.4
        out["net_profit"] = out["market_cap"] * 0.05
        out["roe"] = np.tile(np.linspace(0.30, 0.02, len(ASSETS)), N_DATES)
        out["pb_ratio"] = 2.0
        out["pe_ratio"] = 20.0
        out["asset_growth"] = 0.10
        out["publish_date"] = DATES[0]
        out["fiscal_period_end"] = DATES[0]
        return out.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]

    def get_benchmark(self, start_date, end_date) -> pd.Series:
        return pd.Series(0.0, index=DATES, name="benchmark").loc[
            pd.Timestamp(start_date):pd.Timestamp(end_date)
        ]

    def get_st_timeseries(self) -> pd.DataFrame:
        if not self._st_episodes:
            return pd.DataFrame(
                columns=["asset", "announce_date", "end_date", "is_st"]
            )
        return pd.DataFrame(self._st_episodes)


def _run_pipeline(provider, **kwargs) -> DataPipeline:
    pipe = DataPipeline(
        provider,
        str(DATES[0].date()),
        str(DATES[-1].date()),
        **kwargs,
    )
    pipe.run()
    return pipe


def _optimizer() -> EqualWeightOptimizer:
    return EqualWeightOptimizer(PortfolioConstraints())


class TestP1FundamentalFactorsRespectTheMask:
    """ST names must not get a fundamental factor, signal or weight."""

    @staticmethod
    def _pipeline():
        st_asset = ASSETS[0]
        st = [
            {
                "asset": st_asset,
                "announce_date": DATES[20],
                "end_date": DATES[30],
                "is_st": True,
            }
        ]
        pipe = _run_pipeline(FixedProvider(_random_close(), st_episodes=st))
        return pipe, st_asset, DATES[20], DATES[30]

    def test_fundamental_factor_is_nan_on_st_days(self):
        pipe, asset, announce, end = self._pipeline()
        prices = pipe.get_factor_prices()
        fin = pipe.financials.unstack("asset")

        values = ROE().run(prices, financials=fin).values
        st_days = DATES[(announce <= DATES) & (end >= DATES)]

        assert values.loc[st_days, asset].isna().all()
        # Before the episode the value is live; nothing else was masked.
        assert values.loc[DATES[19], asset] == pytest.approx(0.30)
        assert values["000002.SZ"].notna().all()

        # Regression guard: with the raw financials (no mask) the factor is
        # non-NaN on exactly these days — this is what the old code fed into
        # the signal.
        unmasked = ROE().compute(prices.ffill(), financials=fin)
        assert unmasked.loc[st_days, asset].notna().all()

    def test_signal_and_weight_are_masked_on_st_days(self):
        pipe, asset, announce, end = self._pipeline()
        prices = pipe.get_factor_prices()
        fin = pipe.financials.unstack("asset")

        roe = ROE().run(prices, financials=fin).values
        processed = process_factor(
            roe, neutralize_enabled=False, winsorize_enabled=False,
            standardize_enabled=False,
        )
        st_days = DATES[(announce <= DATES) & (end >= DATES)]
        assert processed.loc[st_days, asset].isna().all()

        signal = _generate_signal({"roe": processed}, pipe.returns, prices)
        assert signal.loc[st_days, asset].isna().all()
        assert signal.loc[DATES[35], asset:].notna().any()

        st_day = DATES[25]
        weights = _optimizer().optimize(signal.loc[st_day])
        assert float(weights.get(asset, 0.0)) == 0.0
        # Non-vacuous: the cross-section does produce positions.
        assert float(weights.sum()) > 0.0

    def test_compute_factors_masks_financial_inputs_end_to_end(self):
        """main._compute_factors must not pass unfiltered financials on."""
        from quant_platform import main as cli

        pipe, asset, announce, end = self._pipeline()
        prices = pipe.get_factor_prices()

        processed, _ic, _sector, fin_masked, raw = cli._compute_factors(
            prices, pipe.returns, pipe.financials, pipe.metadata,
            pipe.get_turnover(), config=None, return_raw=True,
        )
        st_days = DATES[(announce <= DATES) & (end >= DATES)]

        assert raw["roe"].loc[st_days, asset].isna().all()
        assert processed["roe"].loc[st_days, asset].isna().all()
        assert fin_masked["roe"][asset].loc[st_days].isna().all()
        assert fin_masked["roe"][asset].loc[DATES[19]] == pytest.approx(0.30)


class TestP2DelistingIsEnforced:
    """Both provider shapes: prices stop, or prices continue past delisting."""

    ASSET = ASSETS[6]
    DELIST = DATES[40]

    def _pipeline(self, keep_pricing: bool):
        close = _random_close(seed=9)
        if not keep_pricing:
            close.loc[close.index >= self.DELIST, self.ASSET] = np.nan
        provider = FixedProvider(close, delisting_date={self.ASSET: self.DELIST})
        return _run_pipeline(provider, exclude_st=False)

    @pytest.mark.parametrize("keep_pricing", [False, True])
    def test_factor_and_signal_are_nan_from_delisting(self, keep_pricing):
        pipe = self._pipeline(keep_pricing)
        prices = pipe.get_factor_prices()

        momentum = Momentum1M(period=5).run(prices).values
        after = DATES[DATES >= self.DELIST]
        assert momentum.loc[after, self.ASSET].isna().all()
        # The day before delisting is still live.
        assert pd.notna(momentum.loc[DATES[39], self.ASSET])

        signal = _generate_signal({"momentum_1m": momentum}, pipe.returns, prices)
        assert signal.loc[after, self.ASSET].isna().all()

    @pytest.mark.parametrize("keep_pricing", [False, True])
    def test_optimizer_cannot_open_a_new_position(self, keep_pricing):
        pipe = self._pipeline(keep_pricing)
        prices = pipe.get_factor_prices()
        momentum = Momentum1M(period=5).run(prices).values
        signal = _generate_signal({"momentum_1m": momentum}, pipe.returns, prices)

        weights = _optimizer().optimize(signal.loc[DATES[45]])
        assert float(weights.get(self.ASSET, 0.0)) == 0.0

    @pytest.mark.parametrize("keep_pricing", [False, True])
    def test_prices_and_returns_are_blanked_from_delisting(self, keep_pricing):
        pipe = self._pipeline(keep_pricing)
        after = DATES[DATES >= self.DELIST]

        # Variant B keeps producing prices; the pipeline must still blank
        # them, or a stale/fabricated mark drives returns and factors.
        assert pipe.get_close().loc[after, self.ASSET].isna().all()
        assert pipe.returns.loc[after, self.ASSET].isna().all()
        # The last pre-delisting return is NaN too: it would span into the
        # delisting.
        assert pd.isna(pipe.returns.loc[DATES[39], self.ASSET])

        mask = pipe.get_tradable_mask()
        assert not mask.loc[after, self.ASSET].any()

    def test_delisting_is_inferred_when_the_provider_stops_pricing(self):
        """No delisting_date metadata: a terminal price gap can be a delisting.

        Justification for the change: inference now uses
        ``delisting_gap_days``, a threshold deliberately separate from
        ``max_suspension_days`` (D31).  This sample's terminal gap is 39
        sessions; the default 60-session threshold would classify it as an
        end-of-sample suspension (see the companion test below), so this test
        opts into a 35-session inference threshold explicitly.  The
        provider-stated delisting_date path is unchanged and covered above.
        """
        close = _random_close(seed=9)
        close.loc[close.index >= DATES[20], self.ASSET] = np.nan
        provider = FixedProvider(close)  # no delisting_date in metadata
        pipe = _run_pipeline(provider, exclude_st=False, delisting_gap_days=35)

        after = DATES[DATES[20] <= DATES]
        assert not pipe.get_tradable_mask().loc[after, self.ASSET].any()
        momentum = Momentum1M(period=5).run(pipe.get_factor_prices()).values
        assert momentum.loc[after, self.ASSET].isna().all()
        assert pd.notna(momentum.loc[DATES[19], self.ASSET])

        signal = _generate_signal({"momentum_1m": momentum}, pipe.returns, pipe.get_factor_prices())
        assert signal.loc[after, self.ASSET].isna().all()

    def test_end_of_sample_suspension_is_not_inferred_as_delisting(self):
        """A terminal gap shorter than delisting_gap_days stays a suspension.

        Same 39-session gap as above but with the default thresholds
        (``max_suspension_days`` 30, ``delisting_gap_days`` 60): the gap
        exceeds the suspension limit but not the delisting threshold, so the
        asset must not be silently reclassified as delisted (D31).
        """
        close = _random_close(seed=9)
        close.loc[close.index >= DATES[20], self.ASSET] = np.nan
        pipe = _run_pipeline(FixedProvider(close), exclude_st=False)

        assert self.ASSET not in pipe._inferred_delistings
        assert pd.isna(pipe._delisting_dates[self.ASSET])
        # The missing rows are still untradable — as a suspension, via the
        # close-is-missing rule — so the distinction is semantic, not a
        # tradability loophole.
        after = DATES[DATES[20] <= DATES]
        assert not pipe.get_tradable_mask().loc[after, self.ASSET].any()


class TestP3SuspensionBlocksNewWeights:
    """A top-ranked but suspended name must not receive a target weight."""

    ASSET = ASSETS[2]
    SUSP_DAY = DATES[30]

    @staticmethod
    def _pipeline():
        close = _random_close(seed=13)
        # Strong steady uptrend: unambiguously the top momentum name.
        close[TestP3SuspensionBlocksNewWeights.ASSET] = (
            10.0 * np.cumprod(np.full(N_DATES, 1.02))
        )
        suspended = pd.DataFrame(False, index=DATES, columns=ASSETS)
        suspended.loc[TestP3SuspensionBlocksNewWeights.SUSP_DAY,
                      TestP3SuspensionBlocksNewWeights.ASSET] = True
        provider = FixedProvider(
            close, suspended=suspended, flag_suspension=True, ffill_limit=5,
        )
        return _run_pipeline(provider, exclude_st=False)

    def test_suspended_day_has_no_factor_signal_or_weight(self):
        pipe = self._pipeline()
        prices = pipe.get_factor_prices()
        asset, day = self.ASSET, self.SUSP_DAY

        momentum = Momentum1M(period=5).run(prices).values
        assert pd.isna(momentum.loc[day, asset])
        assert momentum.loc[DATES[29], asset] == pytest.approx(1.02**5 - 1)

        signal = _generate_signal({"momentum_1m": momentum}, pipe.returns, prices)
        assert pd.isna(signal.loc[day, asset])

        weights = _optimizer().optimize(signal.loc[day])
        assert float(weights.get(asset, 0.0)) == 0.0

        # Non-vacuous: one day earlier the same name is selected.
        weights_ok = _optimizer().optimize(signal.loc[DATES[29]])
        assert float(weights_ok.get(asset, 0.0)) > 0.0

    def test_suspension_is_in_the_tradability_mask(self):
        pipe = self._pipeline()
        mask = pipe.get_tradable_mask()
        assert not bool(mask.loc[self.SUSP_DAY, self.ASSET])
        assert bool(mask.loc[DATES[29], self.ASSET])
        # Valuation keeps the stale mark even though the mask excludes it.
        assert pd.notna(pipe.get_close().loc[self.SUSP_DAY, self.ASSET])


class TestP4PreListingIsGated:
    """No fundamental/turnover factor may exist before listing."""

    ASSET = ASSETS[8]
    LISTING = DATES[20]

    def _pipeline(self):
        close = _random_close(seed=5)
        close.loc[close.index < self.LISTING, self.ASSET] = np.nan
        provider = FixedProvider(close, listing_date={self.ASSET: self.LISTING})
        return _run_pipeline(provider, exclude_st=False)

    def test_fundamental_factor_is_nan_before_listing(self):
        pipe = self._pipeline()
        fin = pipe.financials.unstack("asset")
        values = LogMarketCap().run(pipe.get_factor_prices(), financials=fin).values

        pre = DATES[DATES < self.LISTING]
        assert values.loc[pre, self.ASSET].isna().all()
        assert values.loc[DATES[21], self.ASSET] > 0

    def test_volume_and_turnover_are_nan_before_listing(self):
        pipe = self._pipeline()
        pre = DATES[DATES < self.LISTING]

        assert pipe.get_volume().loc[pre, self.ASSET].isna().all()
        assert pipe.get_turnover().loc[pre, self.ASSET].isna().all()
        assert pipe.get_turnover().loc[DATES[21], self.ASSET] > 0

    def test_signal_is_nan_before_listing(self):
        pipe = self._pipeline()
        fin = pipe.financials.unstack("asset")
        values = LogMarketCap().run(pipe.get_factor_prices(), financials=fin).values
        processed = process_factor(
            values, neutralize_enabled=False, winsorize_enabled=False,
            standardize_enabled=False,
        )
        pre = DATES[DATES < self.LISTING]

        signal = _generate_signal(
            {"log_market_cap": processed}, pipe.returns, pipe.get_factor_prices()
        )
        assert signal.loc[pre, self.ASSET].isna().all()
        assert pd.notna(signal.loc[DATES[25], self.ASSET])

    def test_as_of_accessor_hides_pre_listing_fundamentals(self):
        pipe = self._pipeline()
        as_of = pipe.get_financials_as_of(DATES[10])
        assert self.ASSET not in set(as_of.index.get_level_values("asset"))

        after = pipe.get_financials_as_of(DATES[25])
        assert self.ASSET in set(after.index.get_level_values("asset"))


class TestMaskCausality:
    """The eligibility mask must not read the future of the panel it masks.

    Running the pipeline on a prefix of the sample must reproduce the full
    run's masks, factor inputs and factor values on that prefix.  Only the
    final forward-return cell may differ (it has no next day in the prefix).
    """

    CUT = 45

    @staticmethod
    def _provider():
        close = _random_close(seed=31)
        close.loc[close.index < DATES[20], ASSETS[8]] = np.nan
        suspended = pd.DataFrame(False, index=DATES, columns=ASSETS)
        suspended.loc[DATES[32], ASSETS[2]] = True
        st = [
            {
                "asset": ASSETS[0],
                "announce_date": DATES[18],
                "end_date": DATES[28],
                "is_st": True,
            }
        ]
        return FixedProvider(
            close,
            st_episodes=st,
            suspended=suspended,
            flag_suspension=True,
            ffill_limit=5,
            listing_date={ASSETS[8]: DATES[20]},
            # Delisting after the cut: it must not move pre-cut values.
            delisting_date={ASSETS[6]: DATES[52]},
        )

    def _run(self, end):
        pipe = DataPipeline(
            self._provider(),
            str(DATES[0].date()),
            str(pd.Timestamp(end).date()),
            exclude_st=True,
        )
        pipe.run()
        return pipe

    def test_prefix_run_matches_full_run_on_the_prefix(self):
        full = self._run(DATES[-1])
        cut = self._run(DATES[self.CUT - 1])

        for name, left, right in (
            ("factor_prices", cut.get_factor_prices(), full.get_factor_prices()),
            (
                "excluded_mask",
                cut.get_factor_prices().attrs["excluded_mask"],
                full.get_factor_prices().attrs["excluded_mask"],
            ),
            ("tradable_mask", cut.get_tradable_mask(), full.get_tradable_mask()),
            ("close", cut.get_close(), full.get_close()),
            ("volume", cut.get_volume(), full.get_volume()),
            ("turnover", cut.get_turnover(), full.get_turnover()),
        ):
            pd.testing.assert_frame_equal(
                left, right.reindex(index=left.index, columns=left.columns),
                obj=name,
            )

        # Forward returns only differ in the final cell (no next day).
        pd.testing.assert_frame_equal(
            cut.returns.iloc[:-1],
            full.returns.reindex_like(cut.returns).iloc[:-1],
        )

        prices_cut = cut.get_factor_prices()
        prices_full = full.get_factor_prices()
        pd.testing.assert_frame_equal(
            Momentum1M(period=5).run(prices_cut).values,
            Momentum1M(period=5).run(prices_full).values.reindex_like(
                Momentum1M(period=5).run(prices_cut).values
            ),
        )

        roe_cut = ROE().run(
            prices_cut, financials=cut.financials.unstack("asset")
        ).values
        roe_full = ROE().run(
            prices_full, financials=full.financials.unstack("asset")
        ).values
        pd.testing.assert_frame_equal(roe_cut, roe_full.reindex_like(roe_cut))


class TestP5SuspensionDerivedWithoutFlag:
    """Providers that ffill without `is_suspended` must still be masked."""

    ASSET = ASSETS[2]

    @staticmethod
    def _suspended(start, stop):
        suspended = pd.DataFrame(False, index=DATES, columns=ASSETS)
        suspended.loc[start:stop, TestP5SuspensionDerivedWithoutFlag.ASSET] = True
        return suspended

    def test_ffilled_gap_yields_nan_returns_and_no_fake_zeros(self):
        suspended = self._suspended(DATES[20], DATES[24])
        provider = FixedProvider(
            _random_close(seed=21), suspended=suspended,
            flag_suspension=False, ffill_limit=5,
        )
        pipe = _run_pipeline(provider, exclude_st=False)
        asset = self.ASSET

        # No provider column, but the fill is detected.
        assert "is_suspended" in pipe.prices.columns
        assert pipe._suspended.loc[DATES[20]:DATES[24], asset].all()

        # Every contaminated return is NaN: the two fill boundaries and the
        # whole stale run.
        window = pipe.returns.loc[DATES[19]:DATES[24], asset]
        assert window.isna().all()

        # Under the old code the same rows carried fake zeros:
        raw = pipe.prices["close_adj"].unstack("asset")[asset]
        fake = raw.pct_change(fill_method=None)
        assert (fake.loc[DATES[20]:DATES[24]] == 0.0).all()

        # The first genuine one-day return after the resume survives.
        resume = DATES[25]
        expected = (
            pipe.prices["close_adj"].unstack("asset")[asset].loc[DATES[26]]
            / pipe.prices["close_adj"].unstack("asset")[asset].loc[resume]
            - 1.0
        )
        assert pipe.returns.loc[resume, asset] == pytest.approx(expected)
        assert pd.notna(pipe.returns.loc[resume, asset])

    def test_ffill_then_dropped_rows_are_masked(self):
        """Tushare shape: ffill 5 days, drop the rest of the gap."""
        suspended = self._suspended(DATES[20], DATES[27])
        provider = FixedProvider(
            _random_close(seed=23), suspended=suspended,
            flag_suspension=False, ffill_limit=5, drop_remaining=True,
        )
        pipe = _run_pipeline(provider, exclude_st=False)
        asset = self.ASSET

        assert pipe._suspended.loc[DATES[20]:DATES[27], asset].all()
        assert pipe.returns.loc[DATES[19]:DATES[27], asset].isna().all()

        resume = DATES[28]
        assert pd.notna(pipe.returns.loc[resume, asset])
        expected = (
            pipe.prices["close_adj"].unstack("asset")[asset].loc[DATES[29]]
            / pipe.prices["close_adj"].unstack("asset")[asset].loc[resume]
            - 1.0
        )
        assert pipe.returns.loc[resume, asset] == pytest.approx(expected)
