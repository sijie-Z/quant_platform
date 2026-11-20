"""Tests for the shared causal composite signal (`trading/signals.py`).

The contract: one equal-weight 7-factor score on a documented [0, 1]
cross-sectional percentile scale, computed only from data at or before the
decision row, with deterministic tie handling and NaN/exclusion when history
is insufficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.trading.signals import (
    DEFAULT_MIN_SIGNAL_SCORE,
    MIN_HISTORY,
    SIGNAL_FACTORS,
    SIGNAL_HISTORY_SESSIONS,
    composite_signal,
    rank_symbols,
)


def _panel(
    n_days: int = 180,
    n_assets: int = 12,
    seed: int = 7,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """A deterministic, non-degenerate price panel (>= 10 assets)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    assets = symbols or [f"{600000 + i}.SH" for i in range(n_assets)]
    drift = rng.normal(0.0004, 0.0002, len(assets))
    steps = rng.normal(0, 0.015, (n_days, len(assets))) + drift
    return pd.DataFrame(
        100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=assets,
    )


class TestScale:
    def test_scores_are_cross_sectional_percentiles_in_unit_interval(self):
        panel = _panel()
        scores = composite_signal(panel)

        assert scores.name == "composite_score"
        assert list(scores.index) == list(panel.columns)
        assert scores.notna().all()
        assert scores.min() == pytest.approx(0.0)
        assert scores.max() == pytest.approx(1.0)
        assert ((scores >= 0.0) & (scores <= 1.0)).all()

    def test_a_flat_cross_section_maps_to_zero(self):
        """Every symbol tied -> one averaged rank -> 0.0 on the [0,1] scale,
        below any positive entry threshold (documented degenerate case)."""
        panel = pd.DataFrame(
            {f"{600000 + i}.SH": [100.0] * 40 for i in range(12)}
        )
        scores = composite_signal(panel)

        assert (scores == 0.0).all()
        assert rank_symbols(scores, min_score=DEFAULT_MIN_SIGNAL_SCORE) == []

    def test_default_threshold_selects_from_the_top_of_the_scale(self):
        scores = composite_signal(_panel())
        ranked = rank_symbols(scores, min_score=DEFAULT_MIN_SIGNAL_SCORE)

        assert ranked, "a non-degenerate cross-section must rank something"
        assert all(score >= DEFAULT_MIN_SIGNAL_SCORE for _code, score in ranked)
        # Descending score order, and the best symbol comes first.
        assert [score for _code, score in ranked] == sorted(
            (score for _code, score in ranked), reverse=True,
        )
        assert ranked[0][0] == scores.idxmax()


class TestCausality:
    def test_as_of_uses_only_rows_up_to_the_decision_date(self):
        panel = _panel()

        full_decision = composite_signal(panel, as_of=len(panel) - 1)
        truncated = composite_signal(panel.iloc[: len(panel)])

        pd.testing.assert_series_equal(full_decision, truncated)

        earlier = composite_signal(panel, as_of=150)
        prefix = composite_signal(panel.iloc[:151])
        pd.testing.assert_series_equal(earlier, prefix)

    def test_future_rows_cannot_change_an_earlier_score(self):
        """Poison every row after the decision date; the score must not move."""
        panel = _panel()
        poisoned = panel.copy()
        poisoned.iloc[151:] = poisoned.iloc[151:] * 100.0

        clean = composite_signal(panel, as_of=150)
        dirty = composite_signal(poisoned, as_of=150)

        pd.testing.assert_series_equal(
            clean, dirty, check_names=True, rtol=1e-12, atol=1e-12,
        )

    def test_appending_future_rows_does_not_change_the_last_score(self):
        """The common live pattern: score today, then append tomorrow."""
        panel = _panel()
        base = composite_signal(panel)
        extended = pd.concat([panel, panel.iloc[[-1]] * 2])
        # With as_of the appended rows must be irrelevant.
        again = composite_signal(extended, as_of=len(panel) - 1)
        pd.testing.assert_series_equal(base, again, rtol=1e-12, atol=1e-12)


class TestInsufficientHistory:
    def test_short_panel_returns_nan_for_every_symbol(self):
        panel = _panel(n_days=MIN_HISTORY - 1)
        scores = composite_signal(panel)

        assert scores.isna().all()
        assert rank_symbols(scores) == []
        assert rank_symbols(scores, limit=10, min_score=0.0) == []

    def test_mapping_excludes_short_symbols_without_affecting_the_rest(self):
        panel = _panel()
        history = {code: panel[code].to_numpy() for code in panel.columns}
        short = "600099.SH"
        history[short] = [1.0] * (MIN_HISTORY - 1)

        scores = composite_signal(history)
        assert np.isnan(scores[short])
        assert scores.drop(short).notna().all()

        # The short symbol must not change the others' scores: it never enters
        # the cross-section.
        reference = composite_signal({
            code: values for code, values in history.items() if code != short
        })
        pd.testing.assert_series_equal(
            scores.drop(short), reference, rtol=1e-12, atol=1e-12,
        )

    def test_mapping_and_padded_dataframe_are_equivalent(self):
        """Unequal histories: the mapping is left-padded onto the same row
        axis, exactly like an equivalently NaN-padded DataFrame."""
        panel = _panel()
        short_code = panel.columns[0]
        history = {code: panel[code].to_numpy() for code in panel.columns}
        history[short_code] = history[short_code][-40:]

        padded = panel.copy()
        padded.loc[padded.index[:-40], short_code] = np.nan

        pd.testing.assert_series_equal(
            composite_signal(history), composite_signal(padded),
            rtol=1e-12, atol=1e-12,
        )

    def test_symbol_history_is_a_trailing_window_only(self):
        """Longer histories are truncated to the shared signal window -- no
        old rows leak into the cross-section."""
        panel = _panel()
        full = {code: panel[code].to_numpy() for code in panel.columns}
        tail = {
            code: values[-SIGNAL_HISTORY_SESSIONS:]
            for code, values in full.items()
        }

        pd.testing.assert_series_equal(
            composite_signal(full), composite_signal(tail),
            rtol=1e-12, atol=1e-12,
        )


class TestDeterminism:
    def test_ties_break_on_symbol_ascending(self):
        scores = pd.Series(
            {"600003.SH": 0.5, "600001.SH": 0.5, "600002.SH": 0.9,
             "600004.SH": float("nan")},
        )
        ranked = rank_symbols(scores)

        assert ranked == [
            ("600002.SH", 0.9), ("600001.SH", 0.5), ("600003.SH", 0.5),
        ]

    def test_column_order_does_not_change_scores(self):
        panel = _panel()
        permuted = panel[list(reversed(panel.columns))]

        original = composite_signal(panel)
        shuffled = composite_signal(permuted)

        pd.testing.assert_series_equal(
            original.sort_index(), shuffled.sort_index(),
            rtol=1e-12, atol=1e-12,
        )

    def test_identical_columns_receive_identical_scores(self):
        panel = _panel(n_assets=11)
        panel["600099.SH"] = panel["600000.SH"].to_numpy()

        scores = composite_signal(panel)
        assert scores["600099.SH"] == scores["600000.SH"]

        ranked = rank_symbols(scores)
        first = ranked.index(next(
            item for item in ranked if item[0] == "600000.SH"
        ))
        second = ranked.index(next(
            item for item in ranked if item[0] == "600099.SH"
        ))
        assert first < second, "a tied pair must be ordered by symbol"

    def test_rank_symbols_applies_threshold_then_limit(self):
        scores = pd.Series({"A": 0.9, "B": 0.5, "C": 0.3, "D": 0.1})

        assert [code for code, _ in rank_symbols(scores, min_score=0.3)] == [
            "A", "B", "C",
        ]
        assert [code for code, _ in rank_symbols(scores, limit=2)] == ["A", "B"]
        assert [
            code for code, _ in rank_symbols(scores, limit=2, min_score=0.4)
        ] == ["A", "B"]


class TestSharedDefinition:
    def test_seven_factors_are_named(self):
        assert SIGNAL_FACTORS == (
            "momentum_1m", "momentum_3m", "momentum_6m",
            "volatility_20d", "volatility_60d", "rsi_14d", "macd",
        )

    def test_equal_weight_generator_delegates_to_the_shared_function(self):
        from quant_platform.trading.signal_generator import LiveSignalGenerator

        panel = _panel()
        generator = LiveSignalGenerator(alpha_method="equal_weight")
        generator.update_prices(panel)
        delegated = generator.generate()

        expected = composite_signal(panel).dropna().to_dict()
        assert delegated == pytest.approx(expected)

    def test_generator_custom_factors_still_resolve_through_the_registry(self):
        """The shared function must not narrow LiveSignalGenerator to its
        seven defaults: any registered price-only factor still works."""
        from quant_platform.trading.signal_generator import LiveSignalGenerator

        generator = LiveSignalGenerator(
            factor_names=["amplitude_20d", "efficiency_ratio"],
            alpha_method="equal_weight",
        )
        generator.update_prices(_panel())

        scores = generator.generate()
        assert scores, "custom registered factors produced no scores"

    def test_generator_forwards_extra_factor_inputs(self):
        """Factors that need a non-price panel (turnover_20d) must receive it
        on the equal-weight delegation too, not only on the IC branches."""
        from quant_platform.trading.signal_generator import LiveSignalGenerator

        panel = _panel()
        generator = LiveSignalGenerator(
            factor_names=["turnover_20d"], alpha_method="equal_weight",
        )
        generator.update_prices(panel)
        generator.update_turnover(panel.rank(axis=1))

        scores = generator.generate()
        assert scores, "turnover_20d was dropped: its panel was not forwarded"
        assert max(scores.values()) > 0.0

    def test_the_default_path_self_registers_its_factors(self):
        """The registry is a process-wide singleton; a cleared or partially
        populated registry must not silently drop the default factors."""
        from quant_platform.factors.registry import get_registry

        registry = get_registry()
        saved = dict(registry._factors)
        try:
            registry.clear()
            scores = composite_signal(_panel())
            assert scores.notna().all()
        finally:
            registry._factors.clear()
            registry._factors.update(saved)

    def test_the_full_signal_window_is_long_enough_for_six_month_momentum(self):
        # Six-month momentum consumes 126 daily returns -> 127 closes.
        assert SIGNAL_HISTORY_SESSIONS >= 127
