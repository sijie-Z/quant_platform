"""Provider-contract validation and data-quality regressions (D25-D31, D44, D45).

Each test crafts the offending provider frame and asserts the pipeline's
documented, actionable behaviour:

* D45 -- provider output is validated at the boundary and errors name the
  provider plus the violated column/index;
* D25 -- NaT listing dates are kept with a warning/count, a missing
  ``listing_date`` column is a contract error;
* D26 -- duplicate ``(date, asset)`` rows are deduplicated keep="last" with a
  warning, never a raw pandas unstack error;
* D27 -- ``adj_factor`` is validated; invalid factors are per-asset
  forward-filled with a count, unrepairable ones raise;
* D28 -- empty financials no longer kill the run;
* D29 -- a close-only forward-fill is detected as a suspension;
* D30 -- board price limits use normalized symbols (prefix/suffix/Beijing);
* D31 -- delisting inference uses a threshold separate from
  ``max_suspension_days`` and warns with the inferred dates;
* D44 -- ``get_financials_as_of`` refuses non-PIT panels and filters delisted
  names.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from quant_platform.data.pipeline import (
    DataPipeline,
    normalize_symbol,
    price_limit_threshold,
)
from quant_platform.data.providers.base import DataProvider
from quant_platform.data.schema import (
    DataContractError,
    DataQualityError,
    validate_financials,
)

N_DATES = 40
DATES = pd.bdate_range("2024-01-02", periods=N_DATES, name="date")
ASSETS = ["600001.SH", "000001.SZ", "300001.SZ"]


# ---------------------------------------------------------------------------
# Crafted provider frames
# ---------------------------------------------------------------------------


def make_close(dates=None, assets=None, base: float = 10.0) -> pd.DataFrame:
    dates = DATES if dates is None else dates
    assets = ASSETS if assets is None else assets
    values = np.linspace(base, base * 1.2, len(dates))
    grid = np.repeat(values, len(assets)).reshape(len(dates), len(assets))
    return pd.DataFrame(grid, index=pd.DatetimeIndex(dates, name="date"),
                        columns=pd.Index(assets, name="asset"))


def make_prices(close: pd.DataFrame | None = None) -> pd.DataFrame:
    close = make_close() if close is None else close
    dates = close.index
    assets = close.columns
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    flat = close.to_numpy().ravel()
    df = pd.DataFrame(index=index)
    df["open"] = flat * 0.995
    df["high"] = flat * 1.01
    df["low"] = flat * 0.99
    df["close"] = flat
    df["volume"] = 1_000_000.0
    df["turnover"] = 0.02
    df["adj_factor"] = 1.0
    # vwap intentionally omitted: it is optional in the provider contract.
    return df


def make_financials(publish_date=None) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([DATES, ASSETS], names=["date", "asset"])
    n = len(index)
    data = {
        "market_cap": np.full(n, 1.0e9),
        "total_assets": np.full(n, 2.0e9),
        "net_assets": np.full(n, 1.0e9),
        "revenue": np.full(n, 1.0e8),
        "net_profit": np.full(n, 1.0e7),
        "roe": np.full(n, 0.10),
        "pb_ratio": np.full(n, 1.5),
        "pe_ratio": np.full(n, 15.0),
        "asset_growth": np.full(n, 0.05),
    }
    if publish_date is not None:
        data["publish_date"] = publish_date
    return pd.DataFrame(data, index=index)


def make_metadata(*, include_is_st: bool = True) -> pd.DataFrame:
    meta = pd.DataFrame(
        {
            "sector": ["Tech"] * len(ASSETS),
            "market_cap_group": ["mid"] * len(ASSETS),
            "listing_date": [pd.Timestamp("2020-01-01")] * len(ASSETS),
            "delisting_date": [pd.NaT] * len(ASSETS),
        },
        index=pd.Index(ASSETS, name="asset"),
    )
    if include_is_st:
        meta["is_st"] = False
    return meta


class CraftedProvider(DataProvider):
    """Static provider returning crafted frames, sliced by the requested range."""

    def __init__(
        self,
        prices: pd.DataFrame | None = None,
        financials: pd.DataFrame | None = None,
        metadata: pd.DataFrame | None = None,
        benchmark: pd.Series | None = None,
        name: str = "crafted",
    ):
        self._prices = prices
        self._financials = financials
        self._metadata = metadata
        self._benchmark = benchmark
        self.name = name

    def get_prices(self, start_date, end_date, fields=None):
        if self._prices is None:
            return pd.DataFrame()
        out = self._prices.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
        return out[fields] if fields else out

    def get_financials(self, start_date, end_date):
        if self._financials is None:
            return pd.DataFrame()
        return self._financials.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]

    def get_benchmark(self, start_date, end_date):
        if self._benchmark is None:
            return pd.Series(0.0, index=DATES)
        return self._benchmark.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]

    def get_metadata(self):
        return self._metadata


def run_pipeline(provider: CraftedProvider, **kwargs) -> DataPipeline:
    kwargs.setdefault("exclude_st", False)
    kwargs.setdefault("exclude_suspended", False)
    pipe = DataPipeline(
        provider,
        str(DATES[0].date()),
        str(DATES[-1].date()),
        **kwargs,
    )
    pipe.run()
    return pipe


def default_provider(**overrides) -> CraftedProvider:
    frames = {
        "prices": make_prices(),
        "financials": make_financials(),
        "metadata": make_metadata(),
    }
    frames.update(overrides)
    return CraftedProvider(**frames)


# ---------------------------------------------------------------------------
# D45: contract validation at the provider boundary
# ---------------------------------------------------------------------------


class TestProviderContractValidation:
    def test_missing_price_column_names_provider_and_column(self):
        provider = default_provider(prices=make_prices().drop(columns=["volume"]))
        with pytest.raises(DataContractError) as exc:
            run_pipeline(provider)
        message = str(exc.value)
        assert "CraftedProvider" in message
        assert "volume" in message

    def test_price_index_levels_must_be_named_date_asset(self):
        prices = make_prices()
        prices.index = prices.index.set_names([None, None])
        provider = default_provider(prices=prices)
        with pytest.raises(DataContractError, match="index levels must be named"):
            run_pipeline(provider)

    def test_unsorted_price_index_is_rejected_with_actionable_message(self):
        from quant_platform.data.schema import validate_prices

        prices = make_prices().sort_index().iloc[::-1]
        with pytest.raises(DataContractError, match="not sorted"):
            validate_prices(prices, provider=CraftedProvider())

    def test_price_frame_without_vwap_is_accepted(self):
        """vwap is optional: Baostock/Postgres do not ship it."""
        pipe = run_pipeline(default_provider())
        assert "vwap" not in pipe.prices.columns

    def test_missing_financial_column_names_provider_and_column(self):
        financials = make_financials().drop(columns=["asset_growth"])
        provider = default_provider(financials=financials)
        with pytest.raises(DataContractError) as exc:
            run_pipeline(provider)
        message = str(exc.value)
        assert "CraftedProvider" in message
        assert "asset_growth" in message

    def test_missing_metadata_column_names_provider_and_column(self):
        metadata = make_metadata().drop(columns=["listing_date"])
        provider = default_provider(metadata=metadata)
        with pytest.raises(DataContractError) as exc:
            run_pipeline(provider)
        message = str(exc.value)
        assert "CraftedProvider" in message
        assert "listing_date" in message

    def test_metadata_index_must_be_named_asset(self):
        metadata = make_metadata()
        metadata.index = metadata.index.rename("code")
        provider = default_provider(metadata=metadata)
        with pytest.raises(DataContractError, match="index must be named 'asset'"):
            run_pipeline(provider)

    def test_empty_metadata_is_a_contract_error(self):
        metadata = pd.DataFrame(
            columns=["sector", "listing_date", "delisting_date"],
            index=pd.Index([], name="asset"),
        )
        provider = default_provider(metadata=metadata)
        with pytest.raises(DataContractError, match="metadata is empty"):
            run_pipeline(provider)

    def test_empty_financials_frame_is_accepted_by_validator(self):
        validate_financials(pd.DataFrame())  # must not raise (D28)

    def test_is_st_absent_is_not_treated_as_false(self, caplog):
        """Absence of ST history must warn, not silently mean 'not ST'."""
        provider = default_provider(metadata=make_metadata(include_is_st=False))
        with caplog.at_level(logging.WARNING):
            pipe = run_pipeline(provider, exclude_st=True)
        assert "is_st" not in pipe.metadata.columns
        assert "ST screening is unavailable" in caplog.text

    def test_nullable_is_st_does_not_crash(self, caplog):
        """SQL NULL flags (all-NA) must not raise TypeError in the filter."""
        metadata = make_metadata()
        metadata["is_st"] = pd.array([pd.NA] * len(ASSETS), dtype="boolean")
        provider = default_provider(metadata=metadata)
        with caplog.at_level(logging.WARNING):
            pipe = run_pipeline(provider, exclude_st=True)
        assert "is_st column with no values" in caplog.text
        assert len(pipe.metadata) == len(ASSETS)


# ---------------------------------------------------------------------------
# D25: unknown listing dates
# ---------------------------------------------------------------------------


class TestUnknownListingDates:
    def test_nat_listing_date_is_kept_with_warning_and_count(self, caplog):
        metadata = make_metadata()
        metadata.loc[ASSETS[0], "listing_date"] = pd.NaT
        provider = default_provider(metadata=metadata)

        with caplog.at_level(logging.WARNING):
            pipe = run_pipeline(provider)

        assert ASSETS[0] in pipe.valid_assets
        assert pipe.unknown_listing_assets == 1
        assert "unknown (NaT) listing_date" in caplog.text
        # Kept names keep their price history from the start of the sample.
        assert pipe.get_close()[ASSETS[0]].notna().any()

    def test_known_future_listing_is_still_filtered(self):
        metadata = make_metadata()
        late = DATES[-1] + pd.Timedelta(days=30)
        metadata.loc[ASSETS[1], "listing_date"] = late
        pipe = run_pipeline(default_provider(metadata=metadata))
        assert ASSETS[1] not in pipe.valid_assets


# ---------------------------------------------------------------------------
# D26: duplicate (date, asset) rows
# ---------------------------------------------------------------------------


class TestDuplicateRows:
    def test_duplicate_price_rows_are_deduplicated_keep_last(self, caplog):
        prices = make_prices()
        key = (DATES[5], ASSETS[0])
        duplicate = prices.loc[[key]].copy()
        duplicate[["open", "high", "low", "close"]] = 999.0
        prices = pd.concat([prices, duplicate]).sort_index()

        with caplog.at_level(logging.WARNING):
            pipe = run_pipeline(default_provider(prices=prices))

        assert pipe.duplicate_rows_dropped["prices"] == 1
        assert pipe.prices.loc[key, "close"] == 999.0
        assert "duplicate" in caplog.text
        # The run completes: no raw "Index contains duplicate entries".
        assert pipe.returns.shape[1] == len(ASSETS)

    def test_duplicate_financial_rows_are_deduplicated_keep_last(self):
        financials = make_financials()
        key = (DATES[3], ASSETS[1])
        duplicate = financials.loc[[key]].copy()
        duplicate["market_cap"] = 5.0e9
        financials = pd.concat([financials, duplicate]).sort_index()

        pipe = run_pipeline(default_provider(financials=financials))

        assert pipe.duplicate_rows_dropped["financials"] == 1
        assert pipe.financials.loc[key, "market_cap"] == 5.0e9


# ---------------------------------------------------------------------------
# D27: adj_factor validation / repair
# ---------------------------------------------------------------------------


class TestAdjFactorValidation:
    def test_zero_factor_is_repaired_from_last_valid_with_warning(self, caplog):
        prices = make_prices()
        key = (DATES[10], ASSETS[0])
        prices.loc[key, "adj_factor"] = 0.0

        with caplog.at_level(logging.WARNING):
            pipe = run_pipeline(default_provider(prices=prices))

        assert pipe.adj_factor_repairs == 1
        assert pipe.prices.loc[key, "adj_factor"] == 1.0
        close_adj = pipe.prices.loc[key, "close_adj"]
        assert np.isfinite(close_adj)
        assert close_adj == pytest.approx(pipe.prices.loc[key, "close"])
        assert "adj_factor" in caplog.text

    def test_negative_and_infinite_factors_are_treated_as_invalid(self):
        prices = make_prices()
        prices.loc[(DATES[10], ASSETS[0]), "adj_factor"] = -2.0
        prices.loc[(DATES[11], ASSETS[0]), "adj_factor"] = np.inf

        pipe = run_pipeline(default_provider(prices=prices))

        assert pipe.adj_factor_repairs == 2
        assert np.isfinite(
            pipe.prices.loc[(DATES[10], ASSETS[0]), "close_adj"]
        )
        assert np.isfinite(
            pipe.prices.loc[(DATES[11], ASSETS[0]), "close_adj"]
        )

    def test_unrepairable_factor_raises_actionable_error(self):
        prices = make_prices()
        prices.loc[(DATES[0], ASSETS[0]), "adj_factor"] = 0.0

        with pytest.raises(DataQualityError) as exc:
            run_pipeline(default_provider(prices=prices))

        message = str(exc.value)
        assert "CraftedProvider" in message
        assert "adj_factor" in message
        assert ASSETS[0] in message

    def test_nan_factor_on_missing_close_is_not_fatal(self):
        prices = make_prices()
        key = (DATES[10], ASSETS[1])
        for column in ("open", "high", "low", "close"):
            prices.loc[key, column] = np.nan
        prices.loc[key, "adj_factor"] = np.nan

        pipe = run_pipeline(default_provider(prices=prices))

        assert np.isnan(pipe.prices.loc[key, "close_adj"])
        assert bool(pipe.prices.loc[key, "is_suspended"])


# ---------------------------------------------------------------------------
# D28: empty financials
# ---------------------------------------------------------------------------


class TestEmptyFinancials:
    def test_run_completes_with_documented_multiindex(self):
        provider = default_provider(financials=None)
        pipe = run_pipeline(provider)

        assert isinstance(pipe.financials.index, pd.MultiIndex)
        assert pipe.financials.index.names == ["date", "asset"]
        assert pipe.financials.empty
        assert pipe.financials_pit_available is False

    def test_market_cap_and_as_of_degrade_to_empty_panels(self):
        pipe = run_pipeline(default_provider(financials=None))

        market_cap = pipe.get_market_cap()
        assert list(market_cap.columns) == ASSETS
        assert market_cap.isna().all().all()

        assert pipe.get_financials_as_of(DATES[5]).empty

    def test_empty_frame_with_columns_keeps_the_column_set(self):
        financials = make_financials().iloc[0:0]
        pipe = run_pipeline(default_provider(financials=financials))
        assert pipe.financials.empty
        assert "market_cap" in pipe.financials.columns


# ---------------------------------------------------------------------------
# D29: suspension detection for close-only forward fills
# ---------------------------------------------------------------------------


class TestSuspensionDetection:
    ASSET = ASSETS[0]
    STALE = 2

    def _close(self) -> pd.DataFrame:
        close = make_close(assets=[self.ASSET])
        return close

    def _prices_with_close_only_fill(self) -> pd.DataFrame:
        close = self._close()
        # The provider forward-fills only `close` on the stale day; OHLC and
        # vwap are left missing and volume is unchanged.
        close.iloc[self.STALE] = close.iloc[self.STALE - 1]
        prices = make_prices(close)
        key = (DATES[self.STALE], self.ASSET)
        prices.loc[key, ["open", "high", "low"]] = np.nan
        return prices

    def test_close_only_forward_fill_is_detected_as_suspension(self):
        pipe = run_pipeline(
            default_provider(prices=self._prices_with_close_only_fill())
        )
        stale_key = (DATES[self.STALE], self.ASSET)
        assert bool(pipe.prices.loc[stale_key, "is_suspended"])
        # The fake zero return on the day into the stale row, and the
        # compounded resume move, are both masked.
        assert pd.isna(pipe.returns.loc[DATES[self.STALE - 1], self.ASSET])
        assert pd.isna(pipe.returns.loc[DATES[self.STALE], self.ASSET])

    def test_unchanged_close_with_intraday_change_is_not_stale(self):
        close = self._close()
        close.iloc[self.STALE] = close.iloc[self.STALE - 1]
        prices = make_prices(close)
        key = (DATES[self.STALE], self.ASSET)
        # A genuine trade: same close, different intraday range.
        prices.loc[key, "open"] = close.iloc[self.STALE, 0] + 0.5
        prices.loc[key, "high"] = close.iloc[self.STALE, 0] + 0.8
        prices.loc[key, "low"] = close.iloc[self.STALE, 0] - 0.7

        pipe = run_pipeline(default_provider(prices=prices))

        assert not bool(pipe.prices.loc[key, "is_suspended"])
        assert pipe.returns.loc[DATES[self.STALE - 1], self.ASSET] == 0.0

    def test_unchanged_close_with_volume_change_is_not_stale(self):
        close = self._close()
        close.iloc[self.STALE] = close.iloc[self.STALE - 1]
        prices = make_prices(close)
        key = (DATES[self.STALE], self.ASSET)
        prices.loc[key, "volume"] = 2_000_000.0

        pipe = run_pipeline(default_provider(prices=prices))

        assert not bool(pipe.prices.loc[key, "is_suspended"])


# ---------------------------------------------------------------------------
# D30: symbol normalization / board price limits
# ---------------------------------------------------------------------------


class TestSymbolNormalization:
    @pytest.mark.parametrize(
        ("raw", "canonical"),
        [
            ("300001.SZ", "300001.SZ"),
            ("sz300001", "300001.SZ"),
            ("SZ.300001", "300001.SZ"),
            ("sh.688001", "688001.SH"),
            ("688001.SH", "688001.SH"),
            ("920001.BJ", "920001.BJ"),
            ("bj430001", "430001.BJ"),
            ("600001", "600001.SH"),
            ("000001", "000001.SZ"),
            ("430001", "430001.BJ"),
        ],
    )
    def test_normalize_symbol(self, raw, canonical):
        assert normalize_symbol(raw) == canonical

    @pytest.mark.parametrize(
        ("symbol", "threshold"),
        [
            ("300001.SZ", 0.20),
            ("sz300001", 0.20),
            ("301001.SZ", 0.20),
            ("sh.688001", 0.20),
            ("688001.SH", 0.20),
            ("920001.BJ", 0.30),
            ("430001.BJ", 0.30),
            ("bj830001", 0.30),
            ("600001.SH", 0.10),
            ("sh600000", 0.10),
            ("000001.SZ", 0.10),
            ("900901.SH", 0.10),
        ],
    )
    def test_price_limit_threshold(self, symbol, threshold):
        assert price_limit_threshold(symbol) == pytest.approx(threshold)

    def test_limit_flags_use_normalized_board_rules(self):
        assets = [
            "300001.SZ", "sz300001", "sh.688001",
            "920001.BJ", "430001.BJ", "600001.SH",
        ]
        dates = pd.bdate_range("2024-01-02", periods=6, name="date")
        close = pd.DataFrame(10.0, index=dates, columns=assets)
        # +15% on day 1 for the 20%/10% boards; +30% for Beijing.
        close.loc[dates[1], ["300001.SZ", "sz300001", "sh.688001",
                             "600001.SH"]] = 11.5
        close.loc[dates[1], ["920001.BJ", "430001.BJ"]] = 13.0
        close.iloc[2:] = close.iloc[1] * 1.001

        index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        flat = close.to_numpy().ravel()
        prices = pd.DataFrame(index=index)
        prices["open"] = flat * 0.99
        prices["high"] = flat * 1.01
        prices["low"] = flat * 0.98
        prices["close"] = flat
        prices["volume"] = 1_000_000.0
        prices["turnover"] = 0.02
        prices["adj_factor"] = 1.0

        metadata = pd.DataFrame(
            {
                "sector": ["Tech"] * len(assets),
                "listing_date": [pd.Timestamp("2020-01-01")] * len(assets),
                "delisting_date": [pd.NaT] * len(assets),
                "is_st": [False] * len(assets),
            },
            index=pd.Index(assets, name="asset"),
        )
        provider = CraftedProvider(
            prices=prices, financials=None, metadata=metadata
        )
        pipe = DataPipeline(
            provider, str(dates[0].date()), str(dates[-1].date()),
            exclude_st=False, exclude_suspended=False,
        )
        pipe.run()

        limit_up = {
            asset: bool(pipe.prices.loc[(dates[1], asset), "is_limit_up"])
            for asset in assets
        }
        # ChiNext/STAR boards: +15% is below the 20% limit.
        for asset in ("300001.SZ", "sz300001", "sh.688001"):
            assert limit_up[asset] is False, asset
        # Main board: +15% exceeds 10%.
        assert limit_up["600001.SH"] is True
        # Beijing: +30% hits the 30% limit.
        assert limit_up["920001.BJ"] is True
        assert limit_up["430001.BJ"] is True


# ---------------------------------------------------------------------------
# D31: delisting inference threshold
# ---------------------------------------------------------------------------


class TestDelistingInference:
    ASSET = ASSETS[0]

    def _prices_with_terminal_gap(self, first_missing: int = 11) -> pd.DataFrame:
        close = make_close()
        close.loc[DATES[first_missing]:, self.ASSET] = np.nan
        return make_prices(close)

    def test_inference_threshold_is_separate_from_max_suspension(self, caplog):
        prices = self._prices_with_terminal_gap()
        provider = default_provider(prices=prices)

        with caplog.at_level(logging.WARNING):
            pipe = run_pipeline(
                provider, max_suspension_days=5, delisting_gap_days=15
            )

        assert self.ASSET in pipe._inferred_delistings
        assert pipe._inferred_delistings[self.ASSET] == DATES[11]
        assert pd.notna(pipe._delisting_dates[self.ASSET])
        assert "Inferred delisting" in caplog.text

    def test_provider_delisting_date_wins_over_gap_inference(self):
        prices = self._prices_with_terminal_gap()
        metadata = make_metadata()
        metadata.loc[self.ASSET, "delisting_date"] = DATES[25]
        provider = default_provider(prices=prices, metadata=metadata)

        pipe = run_pipeline(
            provider, max_suspension_days=5, delisting_gap_days=15
        )

        # The inferred first-missing date D+11 would be wrong; the stated
        # date is authoritative.
        assert pipe._delisting_dates[self.ASSET] == DATES[25]
        assert self.ASSET not in pipe._inferred_delistings

    def test_gap_below_inference_threshold_is_not_a_delisting(self):
        prices = self._prices_with_terminal_gap(first_missing=30)
        provider = default_provider(prices=prices)

        pipe = run_pipeline(
            provider, max_suspension_days=5, delisting_gap_days=15
        )

        # The gap is 10 sessions: a suspension, not a delisting.
        assert self.ASSET not in pipe._inferred_delistings
        assert pd.isna(pipe._delisting_dates[self.ASSET])


# ---------------------------------------------------------------------------
# D44: PIT financial accessor
# ---------------------------------------------------------------------------


class TestFinancialsAsOf:
    def test_missing_publish_date_raises_instead_of_returning_full_panel(self):
        financials = make_financials()  # no publish_date
        pipe = run_pipeline(default_provider(financials=financials))
        assert pipe.financials_pit_available is False

        with pytest.raises(DataQualityError) as exc:
            pipe.get_financials_as_of(DATES[5])
        assert "publish_date" in str(exc.value)

    def test_pit_available_flag_true_when_publish_date_present(self):
        financials = make_financials(publish_date=DATES[0])
        pipe = run_pipeline(default_provider(financials=financials))
        assert pipe.financials_pit_available is True
        assert not pipe.get_financials_as_of(DATES[5]).empty

    def test_delisted_names_are_filtered(self):
        financials = make_financials(publish_date=DATES[0])
        metadata = make_metadata()
        metadata.loc[ASSETS[0], "delisting_date"] = DATES[10]
        pipe = run_pipeline(
            default_provider(financials=financials, metadata=metadata)
        )

        before = pipe.get_financials_as_of(DATES[5])
        after = pipe.get_financials_as_of(DATES[15])

        assert ASSETS[0] in set(before.index.get_level_values("asset"))
        assert ASSETS[0] not in set(after.index.get_level_values("asset"))
        assert ASSETS[1] in set(after.index.get_level_values("asset"))

    def test_rows_published_after_as_of_are_excluded(self):
        index = pd.MultiIndex.from_product(
            [DATES[:12], [ASSETS[1]]], names=["date", "asset"]
        )
        financials = make_financials().loc[index].copy()
        financials["publish_date"] = pd.Series(
            DATES[10], index=index, dtype="datetime64[ns]"
        )
        pipe = run_pipeline(default_provider(financials=financials))

        assert pipe.get_financials_as_of(DATES[5]).empty
        visible = pipe.get_financials_as_of(DATES[10])
        assert len(visible) == 1
        assert visible["publish_date"].iloc[0] == DATES[10]
