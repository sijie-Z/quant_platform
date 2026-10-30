"""Provider price-basis declarations (Postgres / Adata / Validated).

Residual item 6 of the price-basis wave: the Postgres and Adata providers
never declared what ``close`` means, and ``ValidatedProvider`` fusion dropped
``adj_factor``/basis attrs, so ``get_raw_close()`` refused validated runs.

Controlled scenario reused from ``test_price_basis.py``: a stock trades at
raw 10 CNY, then a 2-for-1 split halves the traded price to 5 while the
divisor halves to 0.5, so the pipeline's adjusted series stays at 10.

Network-free: SQLAlchemy's engine is a ``MagicMock`` and ``pd.read_sql`` is a
dispatcher over in-memory frames; the adata SDK is never imported.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers import postgres_provider as pg
from quant_platform.data.providers.adata_provider import AdataProvider
from quant_platform.data.providers.base import DataProvider, provider_identity
from quant_platform.data.providers.postgres_provider import PostgresDataProvider
from quant_platform.data.providers.validated_provider import (
    DataDiscrepancyError,
    ValidatedProvider,
)
from quant_platform.data.schema import (
    FINANCIAL_COLUMNS,
    PRICE_COLUMNS,
    DataQualityError,
)

DATES = pd.bdate_range("2024-01-02", periods=50, name="date")
SPLIT_DAY = 43
ASSET = "600519.SH"


# ----------------------------------------------------------------------
# Shared fixtures / helpers
# ----------------------------------------------------------------------

def _split_close() -> tuple[np.ndarray, np.ndarray]:
    """Raw close 10 -> 5 on the split day; divisor 1 -> 0.5."""
    close = np.full(len(DATES), 10.0)
    factor = np.ones(len(DATES))
    close[SPLIT_DAY:] = 5.0
    factor[SPLIT_DAY:] = 0.5
    return close, factor


def _price_rows() -> pd.DataFrame:
    close, factor = _split_close()
    return pd.DataFrame({
        "date": DATES,
        "code": ASSET,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1_000_000.0 + np.arange(len(DATES)),
        "turnover": 0.01,
        "adj_factor": factor,
    })


def _financial_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=[*FINANCIAL_COLUMNS, "publish_date"])


def _benchmark_rows() -> pd.DataFrame:
    return pd.DataFrame({"date": DATES, "return_value": 0.0})


def _metadata_rows() -> pd.DataFrame:
    return pd.DataFrame([{
        "code": ASSET,
        "sector": "食品饮料",
        "market_cap_group": "large",
        "is_st": False,
        "listing_date": pd.Timestamp("2020-01-01"),
        "delisting_date": pd.NaT,
    }])


def _base_frames() -> dict[str, pd.DataFrame]:
    return {
        "prices": _price_rows(),
        "financials": _financial_rows(),
        "benchmark": _benchmark_rows(),
        "metadata": _metadata_rows(),
    }


def _install_fake_read_sql(monkeypatch, frames) -> None:
    def fake_read_sql(query, engine, params=None):
        q = " ".join(str(query).split())
        if "FROM daily_prices" in q:
            return frames["prices"].copy()
        if "FROM financials" in q:
            return frames["financials"].copy()
        if "FROM benchmark_returns" in q:
            return frames["benchmark"].copy()
        if "FROM stock_metadata" in q:
            return frames["metadata"].copy()
        raise AssertionError(f"unexpected query: {q}")

    monkeypatch.setattr(pg.pd, "read_sql", fake_read_sql)


def _make_postgres_provider(monkeypatch, frames, **kwargs) -> PostgresDataProvider:
    _install_fake_read_sql(monkeypatch, frames)
    with patch.object(pg, "create_engine") as mock_engine:
        mock_engine.return_value = MagicMock()
        return PostgresDataProvider(**kwargs)


def _run_pipeline(provider) -> DataPipeline:
    pipe = DataPipeline(
        provider,
        str(DATES[0].date()),
        str(DATES[-1].date()),
        exclude_st=False,
        exclude_suspended=False,
    )
    pipe.run()
    return pipe


def _panel(
    close: np.ndarray | None = None,
    factor: np.ndarray | None = None,
    *,
    with_factor: bool = True,
) -> pd.DataFrame:
    """A (date, asset) OHLCV panel matching the provider contract."""
    if close is None or factor is None:
        close, factor = _split_close()
    index = pd.MultiIndex.from_product(
        [DATES, [ASSET]], names=["date", "asset"],
    )
    frame = pd.DataFrame(index=index)
    frame["open"] = close
    frame["high"] = close * 1.01
    frame["low"] = close * 0.99
    frame["close"] = close
    frame["volume"] = 1_000_000.0 + np.arange(len(index))
    frame["turnover"] = 0.01
    if with_factor:
        frame["adj_factor"] = factor
    return frame


class DeclaredProvider(DataProvider):
    """Provider with an explicit price-basis declaration.

    ``declare_attrs=False`` models a legacy source that returns a bare frame
    (no attrs); with a constant placeholder factor that frame has no basis
    evidence at all and must not be fused with a declared source.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        basis: str = "raw",
        adjusted: str | None = None,
        factor_available: bool | None = None,
        fail: bool = False,
        declare_attrs: bool = True,
    ):
        self.frame = frame
        self.price_adjustment = basis
        self.adjusted_price_basis = (
            adjusted if adjusted is not None
            else ("hfq" if basis == "raw" else basis)
        )
        self.adjustment_factor_available = (
            factor_available if factor_available is not None else basis == "raw"
        )
        self.fail = fail
        self.declare_attrs = declare_attrs

    def get_prices(self, start_date="", end_date="", fields=None):
        if self.fail:
            raise ConnectionError("source unavailable")
        df = self.frame.copy()
        if fields:
            df = df[list(fields)]
        if self.declare_attrs:
            df.attrs.update({
                "price_basis": self.price_adjustment,
                "adjusted_price_basis": self.adjusted_price_basis,
                "adjustment_factor_available": self.adjustment_factor_available,
            })
        return df

    def get_financials(self, start_date="", end_date=""):
        return pd.DataFrame(
            columns=[*FINANCIAL_COLUMNS, "publish_date"],
            index=pd.MultiIndex.from_arrays([[], []], names=["date", "asset"]),
        )

    def get_benchmark(self, start_date="", end_date=""):
        return pd.Series(
            np.zeros(len(DATES)), index=DATES, name="benchmark",
        )

    def get_metadata(self):
        return _metadata_rows().set_index("code").rename_axis("asset")


# ----------------------------------------------------------------------
# PostgresDataProvider: explicit raw + divisor contract
# ----------------------------------------------------------------------

class TestPostgresPriceBasis:
    def test_class_declares_the_raw_divisor_contract(self):
        assert PostgresDataProvider.price_adjustment == "raw"
        assert PostgresDataProvider.adjusted_price_basis == "hfq"
        assert PostgresDataProvider.adjustment_factor_available is True

        provider = object.__new__(PostgresDataProvider)
        assert provider_identity(provider)["adjustment"] == "raw"

    def test_pipeline_resolves_raw_basis_and_getters(self, monkeypatch):
        provider = _make_postgres_provider(monkeypatch, _base_frames())

        pipe = _run_pipeline(provider)

        assert pipe.price_basis == "raw"
        assert pipe.adjusted_price_basis == "hfq"
        assert pipe.adjustment_factor_available is True
        assert pipe.prices.attrs["price_basis"] == "raw"

        raw = pipe.get_raw_close()
        assert raw.attrs["price_basis"] == "raw"
        assert raw.loc[DATES[SPLIT_DAY - 1], ASSET] == pytest.approx(10.0)
        assert raw.loc[DATES[SPLIT_DAY], ASSET] == pytest.approx(5.0)

        adjusted = pipe.get_adjusted_close()
        assert adjusted.attrs["price_basis"] == "hfq"
        # adjusted = raw / factor; the split is invisible to research.
        assert adjusted.loc[DATES[SPLIT_DAY - 1], ASSET] == pytest.approx(10.0)
        assert adjusted.loc[DATES[SPLIT_DAY], ASSET] == pytest.approx(10.0)
        assert pipe.returns[ASSET].abs().max() < 1e-12

    def test_frame_carries_the_basis_attrs(self, monkeypatch):
        provider = _make_postgres_provider(monkeypatch, _base_frames())

        prices = provider.get_prices("2024-01-01", "2024-01-04")

        assert prices.attrs["price_basis"] == "raw"
        assert prices.attrs["adjusted_price_basis"] == "hfq"
        assert prices.attrs["adjustment_factor_available"] is True

    def test_empty_prices_keep_the_contract_and_a_raw_basis(self, monkeypatch):
        frames = _base_frames()
        frames["prices"] = frames["prices"].iloc[0:0]
        provider = _make_postgres_provider(monkeypatch, frames)

        prices = provider.get_prices("2024-01-01", "2024-01-04")

        assert isinstance(prices.index, pd.MultiIndex)
        assert list(prices.index.names) == ["date", "asset"]
        assert prices.empty
        assert "close" in prices.columns
        assert prices.attrs["price_basis"] == "raw"
        assert prices.attrs["adjusted_price_basis"] == "hfq"
        assert prices.attrs["adjustment_factor_available"] is True

    def test_empty_prices_still_reject_unknown_fields(self, monkeypatch):
        """The fields projection contract also applies to empty results."""
        frames = _base_frames()
        frames["prices"] = frames["prices"].iloc[0:0]
        provider = _make_postgres_provider(monkeypatch, frames)

        with pytest.raises(ValueError, match="Unknown price fields"):
            provider.get_prices(
                "2024-01-01", "2024-01-04", fields=["nonexistent"],
            )

    @pytest.mark.parametrize("bad_factor", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_adj_factor_raises_data_quality_error(
        self, monkeypatch, bad_factor,
    ):
        frames = _base_frames()
        frames["prices"].loc[0, "adj_factor"] = bad_factor
        provider = _make_postgres_provider(monkeypatch, frames)

        with pytest.raises(DataQualityError, match="adj_factor"):
            provider.get_prices("2024-01-01", "2024-01-04")

    def test_adjusted_close_with_non_constant_factor_is_rejected(self, monkeypatch):
        """A DB that says 'already adjusted' must not ship a divisor too."""
        provider = _make_postgres_provider(
            monkeypatch, _base_frames(), close_adjustment="hfq",
        )
        assert provider.price_adjustment == "hfq"
        assert provider.adjusted_price_basis == "hfq"
        assert provider.adjustment_factor_available is False

        with pytest.raises(DataQualityError, match="double-adjusted"):
            provider.get_prices("2024-01-01", "2024-01-04")

    def test_adjusted_close_with_unit_factor_declares_adjusted_basis(
        self, monkeypatch,
    ):
        frames = _base_frames()
        frames["prices"]["adj_factor"] = 1.0
        provider = _make_postgres_provider(
            monkeypatch, frames, close_adjustment="qfq",
        )

        prices = provider.get_prices("2024-01-01", "2024-01-04")
        assert prices.attrs["price_basis"] == "qfq"
        assert prices.attrs["adjusted_price_basis"] == "qfq"
        assert prices.attrs["adjustment_factor_available"] is False

        pipe = _run_pipeline(provider)
        assert pipe.price_basis == "qfq"
        with pytest.raises(DataQualityError, match="no raw close"):
            pipe.get_raw_close()

    def test_unknown_close_adjustment_is_rejected(self):
        with pytest.raises(ValueError, match="close_adjustment"):
            PostgresDataProvider(close_adjustment="mystery")


# ----------------------------------------------------------------------
# AdataProvider: prices unsupported (fail closed)
# ----------------------------------------------------------------------

class TestAdataPriceBasis:
    def test_prices_fail_closed_with_actionable_error(self):
        provider = AdataProvider()

        # Must raise before any lazy import of the optional adata package.
        with pytest.raises(NotImplementedError, match="raw OHLCV"):
            provider.get_prices("2024-01-02", "2024-01-04")

    def test_provider_declares_prices_unsupported(self):
        assert AdataProvider.price_adjustment == "unsupported"
        assert AdataProvider.adjustment_factor_available is False

    def test_adata_remains_usable_as_a_validation_secondary(self):
        """A price-less secondary cannot corrupt a raw primary's panel."""
        primary_frame = _panel()
        vp = ValidatedProvider(
            {"primary": DeclaredProvider(primary_frame), "secondary": AdataProvider()},
            primary="primary",
        )

        result = vp.get_validated_prices("2024-01-02", "2024-01-04")

        # Adata's NotImplementedError is caught as a failed source; the raw
        # primary is returned untouched, with its basis intact.
        assert result.n_sources == 1
        assert result.values.attrs["price_basis"] == "raw"
        assert result.values["adj_factor"].notna().all()


# ----------------------------------------------------------------------
# ValidatedProvider: forward/fuse basis + adj_factor
# ----------------------------------------------------------------------

class TestValidatedPriceBasis:
    def test_fuses_two_raw_sources_and_keeps_factor_and_basis(self):
        vp = ValidatedProvider(
            {
                "a": DeclaredProvider(_panel()),
                "b": DeclaredProvider(_panel()),
            },
            primary="a",
        )
        assert vp.price_adjustment == "raw"
        assert vp.adjusted_price_basis == "hfq"
        assert vp.adjustment_factor_available is True

        result = vp.get_validated_prices("2024-01-02", "2024-01-04")
        fused = result.values

        assert result.n_sources == 2
        assert fused.attrs["price_basis"] == "raw"
        assert fused.attrs["adjusted_price_basis"] == "hfq"
        assert fused.attrs["adjustment_factor_available"] is True
        assert fused["adj_factor"].notna().all()
        assert fused.loc[(DATES[SPLIT_DAY], ASSET), "adj_factor"] == pytest.approx(0.5)
        assert fused.loc[(DATES[SPLIT_DAY], ASSET), "close"] == pytest.approx(5.0)

    def test_fusion_takes_the_reference_factor_when_secondary_lacks_it(self):
        primary_frame = _panel()
        secondary_frame = primary_frame.drop(columns=["adj_factor"])
        vp = ValidatedProvider(
            {
                "a": DeclaredProvider(primary_frame),
                "b": DeclaredProvider(secondary_frame),
            },
            primary="a",
        )

        fused = vp.get_prices("2024-01-02", "2024-01-04")

        # The old fusion averaged against a missing column -> NaN factors ->
        # the pipeline's repair step failed closed.
        assert "adj_factor" in fused.columns
        assert fused["adj_factor"].notna().all()
        expected = primary_frame["adj_factor"]
        np.testing.assert_allclose(fused["adj_factor"].to_numpy(), expected.to_numpy())

    def test_fusion_restricts_rows_to_the_reference_panel(self):
        """A secondary with extra symbols must not extend the fused panel.

        Unioning row sets would emit rows the reference cannot price, whose
        adj_factor is NaN (the pipeline then fails closed) and whose close
        levels would sit under the reference's basis attrs.
        """
        primary_frame = _panel()
        extra_index = pd.MultiIndex.from_product(
            [DATES[:10], ["000001.SZ"]], names=["date", "asset"],
        )
        extra = pd.DataFrame(
            {
                "open": 20.0, "high": 20.2, "low": 19.8, "close": 20.0,
                "volume": 500_000.0, "turnover": 0.01, "adj_factor": 1.0,
            },
            index=extra_index,
        )
        secondary_frame = pd.concat([primary_frame, extra]).sort_index()
        vp = ValidatedProvider(
            {
                "a": DeclaredProvider(primary_frame),
                "b": DeclaredProvider(secondary_frame),
            },
            primary="a",
        )

        fused = vp.get_prices("2024-01-02", "2024-01-04")

        assert fused.index.equals(primary_frame.index)
        assert "000001.SZ" not in fused.index.get_level_values("asset")
        assert fused["adj_factor"].notna().all()

    def test_adjusted_basis_divergence_warns_and_keeps_reference(self, caplog):
        """Raw closes with different adjustment paths: reference wins, loudly."""
        vp = ValidatedProvider(
            {
                "a": DeclaredProvider(_panel(), adjusted="hfq"),
                "b": DeclaredProvider(_panel(), adjusted="qfq"),
            },
            primary="a",
        )

        with caplog.at_level(
            logging.WARNING,
            logger="quant_platform.data.providers.validated_provider",
        ):
            fused = vp.get_prices("2024-01-02", "2024-01-04")

        assert "different adjusted bases" in caplog.text
        assert fused.attrs["price_basis"] == "raw"
        assert fused.attrs["adjusted_price_basis"] == "hfq"
        expected = _panel()["adj_factor"]
        np.testing.assert_allclose(
            fused["adj_factor"].to_numpy(), expected.to_numpy(),
        )

    def test_source_contract_violation_is_not_masked(self):
        """A corrupt source must abort the validated run, not be ignored."""
        class ContractViolatingProvider(DeclaredProvider):
            def get_prices(self, start_date="", end_date="", fields=None):
                raise DataQualityError("adj_factor would be double-adjusted")

        vp = ValidatedProvider(
            {
                "bad": ContractViolatingProvider(_panel()),
                "good": DeclaredProvider(_panel()),
            },
            primary="bad",
        )

        with pytest.raises(DataQualityError, match="double-adjusted"):
            vp.get_prices("2024-01-02", "2024-01-04")

    def test_fused_panel_works_with_data_pipeline(self):
        """The regression from residual item 6: validated runs refused raw close."""
        vp = ValidatedProvider(
            {
                "a": DeclaredProvider(_panel()),
                "b": DeclaredProvider(_panel()),
            },
            primary="a",
        )

        pipe = _run_pipeline(vp)

        assert pipe.price_basis == "raw"
        assert pipe.adjusted_price_basis == "hfq"
        assert pipe.adjustment_factor_available is True
        assert pipe.prices["adj_factor"].notna().all()

        raw = pipe.get_raw_close()
        adjusted = pipe.get_adjusted_close()
        assert raw.loc[DATES[SPLIT_DAY], ASSET] == pytest.approx(5.0)
        assert adjusted.loc[DATES[SPLIT_DAY], ASSET] == pytest.approx(10.0)
        assert pipe.returns[ASSET].abs().max() < 1e-12

    def test_single_source_fallback_keeps_its_own_basis(self):
        vp = ValidatedProvider(
            {
                "a": DeclaredProvider(_panel(), fail=True),
                "b": DeclaredProvider(_panel()),
            },
            primary="a",
        )

        prices = vp.get_prices("2024-01-02", "2024-01-04")

        assert prices.attrs["price_basis"] == "raw"
        assert prices.attrs["adjustment_factor_available"] is True
        assert prices["adj_factor"].notna().all()

    def test_refuses_fusion_when_bases_disagree(self):
        raw = DeclaredProvider(_panel())
        adjusted = DeclaredProvider(
            _panel(close=np.full(len(DATES), 10.0), factor=np.ones(len(DATES))),
            basis="hfq", adjusted="hfq", factor_available=False,
        )
        vp = ValidatedProvider({"a": raw, "b": adjusted}, primary="a")

        with pytest.raises(DataDiscrepancyError, match="incompatible close bases"):
            vp.get_prices("2024-01-02", "2024-01-04")
        # The refusal is a data-quality failure, not a generic crash.
        assert issubclass(DataDiscrepancyError, DataQualityError)

    def test_refuses_fusion_of_declared_and_undeclared_sources(self):
        raw = DeclaredProvider(_panel())
        undeclared = DeclaredProvider(
            _panel(close=np.full(len(DATES), 10.0), factor=np.ones(len(DATES))),
            declare_attrs=False,
        )
        undeclared.price_adjustment = "provider-defined"
        vp = ValidatedProvider({"a": raw, "b": undeclared}, primary="a")

        with pytest.raises(DataDiscrepancyError, match="incompatible close bases"):
            vp.get_prices("2024-01-02", "2024-01-04")

    def test_discrepancy_path_with_failed_primary_tags_survivor(self):
        """The primary may be absent from the discrepancy path (KeyError fix)."""
        base_close, base_factor = _split_close()
        wide = _panel(close=base_close * 1.5, factor=base_factor)
        vp = ValidatedProvider(
            {
                "a": DeclaredProvider(_panel(), fail=True),
                "b": DeclaredProvider(_panel()),
                "c": DeclaredProvider(wide),
            },
            primary="a",
        )

        result = vp.get_validated_prices("2024-01-02", "2024-01-04")

        assert result.n_sources == 2
        assert result.discrepancies
        assert result.values.attrs["price_basis"] == "raw"
        assert result.values.attrs["adjustment_factor_available"] is True

    def test_all_sources_failed_returns_empty_unusable(self):
        vp = ValidatedProvider(
            {"a": DeclaredProvider(_panel(), fail=True)}, primary="a",
        )

        result = vp.get_validated_prices("2024-01-02", "2024-01-04")

        assert result.n_sources == 0
        assert result.confidence == 0.0
        values = result.values
        assert isinstance(values, pd.DataFrame)
        assert values.empty
        # The empty panel still honours the price contract: a (date, asset)
        # MultiIndex, the required columns and the forwarded basis attrs.
        assert isinstance(values.index, pd.MultiIndex)
        assert list(values.index.names) == ["date", "asset"]
        assert set(PRICE_COLUMNS) <= set(values.columns)
        assert values.attrs["price_basis"] == "raw"
        assert values.attrs["adjusted_price_basis"] == "hfq"
        assert values.attrs["adjustment_factor_available"] is True

    def test_none_result_is_treated_as_a_failed_source(self):
        class NoneProvider(DeclaredProvider):
            def get_prices(self, start_date="", end_date="", fields=None):
                return None

        vp = ValidatedProvider(
            {
                "a": NoneProvider(_panel()),
                "b": DeclaredProvider(_panel()),
            },
            primary="a",
        )

        prices = vp.get_prices("2024-01-02", "2024-01-04")

        assert prices.attrs["price_basis"] == "raw"
        assert prices["adj_factor"].notna().all()
