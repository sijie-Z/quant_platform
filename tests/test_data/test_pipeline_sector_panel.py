"""Point-in-time sector panel and PIT sector neutralization.

``main._compute_factors`` used the static ``metadata["sector"]`` map: the
end-of-sample classification.  Whenever the provider ships industry changes,
that leaks the future into every earlier cross-section.  ``DataPipeline``
already had ``get_industry_map(date)``; the panel exposes it as date x asset
so ``neutralize`` can use each date's own classification.

The key property: two datasets that differ only in a *future* industry
change must produce identical neutralized factors before that change, and
must use the new classification afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.base import DataProvider
from quant_platform.factors.processing import neutralize, process_factor

N_ASSETS = 30
ASSETS = [f"{i:06d}.SH" for i in range(1, N_ASSETS + 1)]
DATES = pd.bdate_range("2024-01-02", periods=15)
SECTORS = ["Tech", "Finance", "Energy"]
STATIC_SECTORS = pd.Series(
    [SECTORS[i % len(SECTORS)] for i in range(N_ASSETS)], index=ASSETS
)
CHANGE_DATE = DATES[9]


class _IndustryProvider(DataProvider):
    """Fixed universe with an explicit industry-change table."""

    def __init__(self, changes: pd.DataFrame | None):
        self._changes = changes
        rng = np.random.default_rng(5)
        steps = rng.normal(0, 0.01, (len(DATES), len(ASSETS)))
        self._close = 10.0 * np.exp(np.cumsum(steps, axis=0))

    def get_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sector": STATIC_SECTORS.to_numpy(),
                "market_cap_group": ["mid"] * N_ASSETS,
                "is_st": [False] * N_ASSETS,
                "listing_date": [pd.Timestamp("2000-01-01")] * N_ASSETS,
                "delisting_date": [pd.NaT] * N_ASSETS,
            },
            index=pd.Index(ASSETS, name="asset"),
        )

    def get_prices(self, start_date, end_date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [DATES, ASSETS], names=["date", "asset"]
        )
        close = self._close.ravel()  # row-major: date-major, asset-minor
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": 1_000_000.0,
                "turnover": 0.02,
                "adj_factor": 1.0,
            },
            index=index,
        )

    def get_financials(self, start_date, end_date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [DATES, ASSETS], names=["date", "asset"]
        )
        return pd.DataFrame(
            {
                "market_cap": 1e10, "total_assets": 2e10, "net_assets": 1e10,
                "revenue": 1e9, "net_profit": 1e8, "roe": 0.10,
                "pb_ratio": 1.5, "pe_ratio": 15.0, "asset_growth": 0.05,
            },
            index=index,
        )

    def get_benchmark(self, start_date, end_date) -> pd.Series:
        return pd.Series(0.0005, index=DATES)

    def get_industry_changes(self) -> pd.DataFrame:
        if self._changes is None:
            return pd.DataFrame(
                columns=["asset", "industry", "effective_date"]
            )
        return self._changes


def _changes(change_date: pd.Timestamp | None) -> pd.DataFrame | None:
    """Initial classification for every asset; asset[0] may reclassify."""
    if change_date is None:
        return None
    records = [
        {
            "asset": asset,
            "industry": STATIC_SECTORS[asset],
            "effective_date": DATES[0],
        }
        for asset in ASSETS
    ]
    records.append(
        {
            "asset": ASSETS[0],
            "industry": "Energy"
            if STATIC_SECTORS[ASSETS[0]] != "Energy"
            else "Tech",
            "effective_date": change_date,
        }
    )
    return pd.DataFrame(records)


def _run(changes: pd.DataFrame | None) -> DataPipeline:
    pipeline = DataPipeline(
        provider=_IndustryProvider(changes),
        start_date=str(DATES[0].date()),
        end_date=str(DATES[-1].date()),
    )
    pipeline.run()
    return pipeline


def _factor() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    return pd.DataFrame(
        rng.normal(size=(len(DATES), len(ASSETS))),
        index=DATES,
        columns=ASSETS,
    )


class TestSectorPanel:
    def test_panel_is_date_by_asset_and_follows_the_change(self):
        pipeline = _run(_changes(CHANGE_DATE))
        panel = pipeline.get_sector_panel()

        assert panel.shape == (len(DATES), len(ASSETS))
        assert list(panel.columns) == ASSETS
        # Old classification before the change, new one on/after it.
        change_pos = DATES.get_loc(CHANGE_DATE)
        assert (
            panel.iloc[change_pos - 1][ASSETS[0]]
            == STATIC_SECTORS[ASSETS[0]]
        )
        assert panel.iloc[change_pos][ASSETS[0]] == (
            "Energy" if STATIC_SECTORS[ASSETS[0]] != "Energy" else "Tech"
        )

    def test_static_fallback_without_industry_history(self):
        """No history -> the metadata map repeated for every date, which is
        exactly what get_industry_map() answers in that case."""
        panel = _run(None).get_sector_panel()

        assert panel.shape == (len(DATES), len(ASSETS))
        for asset in ASSETS:
            assert panel[asset].nunique() == 1
        assert panel.iloc[0].to_dict() == STATIC_SECTORS.to_dict()

    def test_industry_map_is_queried_once_per_date(self, monkeypatch):
        """The panel used to call get_industry_map() once per (date, asset):
        46,800 calls / ~98s at 390x120, ~35 minutes at the default sample."""
        pipeline = _run(_changes(CHANGE_DATE))
        assert pipeline._sector_panel is None
        calls = []
        original = DataPipeline.get_industry_map

        def counted(self, as_of_date):
            calls.append(as_of_date)
            return original(self, as_of_date)

        monkeypatch.setattr(DataPipeline, "get_industry_map", counted)
        panel = pipeline.get_sector_panel()

        assert panel.shape[0] == len(DATES)
        assert len(calls) == len(DATES)

    def test_partial_history_is_not_backfilled_from_the_static_map(self):
        """With history that starts mid-sample, early dates have no
        classification. Filling them from the whole-sample static map would
        reintroduce exactly the look-ahead this panel exists to remove."""
        changes = _changes(CHANGE_DATE).copy()
        initial = changes["effective_date"] == DATES[0]
        changes.loc[initial, "effective_date"] = DATES[5]
        changes = changes.sort_values("effective_date").reset_index(drop=True)

        pipeline = _run(changes)
        panel = pipeline.get_sector_panel()
        assert panel.iloc[:5].isna().all().all()

        # The early rows skip the sector leg but keep the regression
        # intercept: they equal the demeaned input, never the static-map
        # residual (and never a no-op pass-through either).
        factor = _factor()
        early = panel.index < DATES[5]
        result = neutralize(
            factor, sector_map=STATIC_SECTORS, sector_panel=panel,
        )
        expected = factor.sub(factor.mean(axis=1), axis=0)
        np.testing.assert_allclose(
            result.loc[early].to_numpy(),
            expected.loc[early].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )
        assert not np.allclose(
            result.loc[early].to_numpy(),
            factor.loc[early].to_numpy(),
            atol=1e-8,
        )
        # Classified dates are still neutralized.
        assert not np.allclose(
            result.loc[DATES[8]].to_numpy(),
            factor.loc[DATES[8]].to_numpy(),
            atol=1e-8,
        )


def test_single_category_panel_is_intercept_only_not_ignored():
    """A one-sector panel must override a multi-sector static map.

    The panel says every asset is in the same industry, so the only
    legitimate design is intercept-only (demeaning); falling back to the
    static map would invent sector effects the PIT data does not support.
    """
    factor = _factor()
    panel = pd.DataFrame("Tech", index=DATES, columns=ASSETS, dtype=object)

    result = neutralize(
        factor, sector_map=STATIC_SECTORS, sector_panel=panel,
    )
    expected = factor.sub(factor.mean(axis=1), axis=0)
    np.testing.assert_allclose(
        result.to_numpy(), expected.to_numpy(), rtol=0.0, atol=1e-12,
    )


def test_history_outside_the_sample_still_disables_static_backfill():
    """History entries that all fall after the sample still mean "history
    exists": the all-NaN panel must suppress the static-map fallback."""
    late = pd.DataFrame(
        [
            {
                "asset": asset,
                "industry": STATIC_SECTORS[asset],
                "effective_date": DATES[-1] + pd.Timedelta(days=30),
            }
            for asset in ASSETS
        ]
    )
    pipeline = _run(late)
    assert pipeline.has_industry_history() is True
    panel = pipeline.get_sector_panel()
    assert panel.isna().all().all()

    factor = _factor()
    result = neutralize(
        factor, sector_map=STATIC_SECTORS, sector_panel=panel,
    )
    expected = factor.sub(factor.mean(axis=1), axis=0)
    np.testing.assert_allclose(
        result.to_numpy(), expected.to_numpy(), rtol=0.0, atol=1e-12,
    )


def test_sector_panel_attach_rule_distinguishes_no_history_from_out_of_sample():
    """main._load_data must not attach an all-NaN panel when the provider has
    no history at all (it would flip the no-op static path into demeaning),
    but must attach it when history exists outside the sample (to block the
    static fallback)."""
    from quant_platform.main import _sector_panel_is_usable

    all_nan = pd.DataFrame(index=DATES, columns=ASSETS, dtype=object)
    assert _sector_panel_is_usable(all_nan, has_history=False) is False
    assert _sector_panel_is_usable(all_nan, has_history=True) is True
    assert _sector_panel_is_usable(None, has_history=True) is False
    populated = all_nan.copy()
    populated.iloc[0, 0] = "Tech"
    assert _sector_panel_is_usable(populated, has_history=False) is True


class TestNoLookaheadInNeutralization:
    def test_future_reclassification_does_not_change_earlier_factors(self):
        # Same data; only the date of asset[0]'s reclassification differs.
        early = _run(_changes(CHANGE_DATE))
        late = _run(_changes(DATES[11]))

        panel_early = early.get_sector_panel()
        panel_late = late.get_sector_panel()

        before = panel_early.index < CHANGE_DATE
        pd.testing.assert_frame_equal(
            panel_early.loc[before], panel_late.loc[before]
        )

        factor = _factor()
        res_early = neutralize(
            factor, sector_map=STATIC_SECTORS, sector_panel=panel_early,
        )
        res_late = neutralize(
            factor, sector_map=STATIC_SECTORS, sector_panel=panel_late,
        )
        np.testing.assert_allclose(
            res_early.loc[before].to_numpy(),
            res_late.loc[before].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )

        # The old static-map path is the look-ahead this replaces: the
        # end-of-sample classification (asset[0] -> Energy) applied to dates
        # before the reclassification changes the past cross-sections.
        latest_map = panel_early.iloc[-1]
        res_static_latest = neutralize(factor, sector_map=latest_map)
        assert not np.allclose(
            res_early.loc[before].to_numpy(),
            res_static_latest.loc[before].to_numpy(),
            atol=1e-6,
        )

    def test_new_classification_is_used_on_and_after_the_change(self):
        pipeline = _run(_changes(CHANGE_DATE))
        panel = pipeline.get_sector_panel()
        factor = _factor()

        res_panel = neutralize(
            factor, sector_map=STATIC_SECTORS, sector_panel=panel,
        )
        # The panel row on the change date IS the classification in effect;
        # a static map built from that row must give the same residual.
        as_of_map = panel.loc[CHANGE_DATE]
        res_static = neutralize(factor, sector_map=as_of_map)

        np.testing.assert_allclose(
            res_panel.loc[CHANGE_DATE].to_numpy(),
            res_static.loc[CHANGE_DATE].to_numpy(),
            rtol=0.0,
            atol=1e-10,
        )
        # ...and it differs from the stale static map's answer.
        res_stale = neutralize(factor, sector_map=STATIC_SECTORS)
        assert not np.allclose(
            res_panel.loc[CHANGE_DATE].to_numpy(),
            res_stale.loc[CHANGE_DATE].to_numpy(),
            atol=1e-6,
        )

    def test_process_factor_forwards_the_panel(self):
        """process_factor must accept and use sector_panel (main passes it)."""
        pipeline = _run(_changes(CHANGE_DATE))
        panel = pipeline.get_sector_panel()
        factor = _factor()

        by_panel = process_factor(
            factor,
            standardize_enabled=False,
            sector_map=STATIC_SECTORS,
            sector_panel=panel,
            market_cap=None,
        )
        as_of_map = panel.loc[CHANGE_DATE]
        by_static = process_factor(
            factor,
            standardize_enabled=False,
            sector_map=as_of_map,
            market_cap=None,
        )
        np.testing.assert_allclose(
            by_panel.loc[CHANGE_DATE].to_numpy(),
            by_static.loc[CHANGE_DATE].to_numpy(),
            rtol=0.0,
            atol=1e-10,
        )


def test_compute_factors_uses_the_panel_carried_by_metadata_attrs():
    """main._load_data attaches the panel to metadata.attrs; _compute_factors
    must pick it up instead of falling back to the static map."""
    from quant_platform.factors.registry import get_registry
    from quant_platform.main import _compute_factors
    from quant_platform.utils.config import load_config

    rng = np.random.default_rng(3)
    n_dates, n_assets = 120, 40
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    assets = [f"S{i:03d}" for i in range(n_assets)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_dates, n_assets)), axis=0)),
        index=dates,
        columns=assets,
    )
    returns = prices.pct_change(fill_method=None).shift(-1)

    static = pd.Series(
        [SECTORS[i % len(SECTORS)] for i in range(n_assets)], index=assets,
    )
    panel = pd.DataFrame(
        np.tile(static.to_numpy(dtype=object), (n_dates, 1)),
        index=dates,
        columns=assets,
    )
    # asset[0] reclassifies from its static sector to "Energy" late in the
    # sample; the static map never sees it.
    new_sector = "Energy" if static[assets[0]] != "Energy" else "Tech"
    panel.loc[dates[80]:, assets[0]] = new_sector

    metadata = pd.DataFrame({"sector": static})
    metadata.attrs["sector_panel"] = panel

    config = load_config(raw={
        "factors": {
            "technical": {"momentum_1m": {"enabled": True}},
            "fundamental": {},
            "processing": {"neutralize": {"enabled": True, "by": ["sector"]}},
        },
    })
    processed, _, sector_map, _ = _compute_factors(
        prices, returns, None, metadata, config=config,
    )

    raw = get_registry().get("momentum_1m")().run(prices).values
    expected = process_factor(
        raw,
        sector_map=sector_map,
        market_cap=None,
        sector_panel=panel,
        factor_name="momentum_1m",
    )
    pd.testing.assert_frame_equal(processed["momentum_1m"], expected)

    # ...and the panel actually changed the output (the static map alone
    # would have left the future classification out).
    without_panel = process_factor(
        raw,
        sector_map=sector_map,
        market_cap=None,
        factor_name="momentum_1m",
    )
    changed = processed["momentum_1m"].loc[dates[80]:]
    stale = without_panel.loc[dates[80]:]
    assert not np.allclose(
        changed.to_numpy(), stale.to_numpy(), equal_nan=True, atol=1e-10,
    )
