"""Strategy-gate boundary semantics: thresholds are inclusive (``>=``).

Every existing gate test uses values strictly above or below the threshold,
so changing ``>=`` to ``>`` went unnoticed: a factor exactly at
``minimum_ic`` stopped counting as passing.
"""

import pytest

from quant_platform.strategy.gates import (
    FAIL,
    PASS,
    WARNING,
    GateConfig,
    check_ic_quality,
)

THRESHOLD = 0.02


def _ic(**factors: dict) -> dict:
    return {name: stats for name, stats in factors.items()}


class TestIcBoundary:
    def test_factor_exactly_at_the_ic_threshold_passes(self):
        ic = {f"f{i}": {"mean_ic": THRESHOLD, "icir": 0.8} for i in range(3)}
        config = GateConfig(
            minimum_ic=THRESHOLD, minimum_icir=0.0,
            minimum_factors_passing_ic=3,
        )

        result = check_ic_quality(ic, config)

        assert result.status == PASS
        assert result.details["n_above_minimum_ic"] == 3

    def test_negative_ic_exactly_at_the_threshold_also_passes(self):
        """The comparison is on absolute value; a stable -0.02 counts."""
        ic = {"short": {"mean_ic": -THRESHOLD, "icir": -0.8}}
        config = GateConfig(
            minimum_ic=THRESHOLD, minimum_icir=0.0,
            minimum_factors_passing_ic=1,
        )

        result = check_ic_quality(ic, config)

        assert result.status == PASS
        assert result.details["n_above_minimum_ic"] == 1

    def test_one_below_the_ic_threshold_does_not_pass(self):
        ic = {"just_below": {"mean_ic": THRESHOLD - 1e-9, "icir": 0.8}}
        config = GateConfig(
            minimum_ic=THRESHOLD, minimum_icir=0.0,
            minimum_factors_passing_ic=1,
        )

        result = check_ic_quality(ic, config)

        assert result.status == WARNING
        assert result.details["n_above_minimum_ic"] == 0


class TestIcirBoundary:
    def test_factor_exactly_at_the_icir_threshold_passes(self):
        ic = {"edge": {"mean_ic": 0.05, "icir": 0.2}}
        config = GateConfig(
            minimum_ic=THRESHOLD, minimum_icir=0.2,
            minimum_factors_passing_ic=1,
        )

        result = check_ic_quality(ic, config)

        assert result.status == PASS
        assert result.details["icir_gate_enabled"] is True
        assert result.details["n_above_minimum_icir"] == 1

    def test_one_below_the_icir_threshold_fails_the_gate(self):
        ic = {"just_below": {"mean_ic": 0.05, "icir": 0.2 - 1e-9}}
        config = GateConfig(
            minimum_ic=THRESHOLD, minimum_icir=0.2,
            minimum_factors_passing_ic=1,
        )

        result = check_ic_quality(ic, config)

        assert result.status == FAIL
        assert result.details["n_above_minimum_icir"] == 0


def test_mixed_factors_count_only_the_ones_at_or_above_threshold():
    ic = _ic(
        at={"mean_ic": 0.02, "icir": 0.9},
        above={"mean_ic": 0.03, "icir": 0.9},
        below={"mean_ic": 0.0199, "icir": 0.9},
    )
    config = GateConfig(
        minimum_ic=THRESHOLD, minimum_icir=0.0,
        minimum_factors_passing_ic=3,
    )

    result = check_ic_quality(ic, config)

    assert result.status == WARNING
    assert result.details["n_above_minimum_ic"] == 2
