"""R8: cost-aware constraint fields and their validation."""

import pytest
from quant_platform.config.schema import PortfolioConstraintsConfig
from quant_platform.portfolio.constraints import PortfolioConstraints


def test_new_cost_fields_default_to_zero():
    c = PortfolioConstraints()
    assert c.transaction_cost_bps == 0.0
    assert c.turnover_penalty == 0.0
    assert c.max_participation == 0.0


@pytest.mark.parametrize("field", ["transaction_cost_bps", "turnover_penalty"])
def test_negative_cost_coefficients_are_rejected(field):
    with pytest.raises(ValueError, match=field):
        PortfolioConstraints(**{field: -0.1})


@pytest.mark.parametrize("field", ["transaction_cost_bps", "turnover_penalty"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_cost_coefficients_are_rejected(field, value):
    """NaN silently disables a ``> 0`` branch; reject it instead."""
    with pytest.raises(ValueError, match=field):
        PortfolioConstraints(**{field: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_max_participation_is_rejected(value):
    with pytest.raises(ValueError, match="max_participation"):
        PortfolioConstraints(max_participation=value)


@pytest.mark.parametrize("value", [-0.01, 1.01, 2.0])
def test_max_participation_outside_the_legal_range_is_rejected(value):
    with pytest.raises(ValueError, match="max_participation"):
        PortfolioConstraints(max_participation=value)


@pytest.mark.parametrize("value", [0.0, 0.01, 0.1, 1.0])
def test_max_participation_accepts_disabled_and_fractions(value):
    assert PortfolioConstraints(max_participation=value).max_participation == value


def test_from_config_forwards_the_cost_fields():
    cfg = PortfolioConstraintsConfig(
        transaction_cost_bps=20.0,
        turnover_penalty=0.5,
        max_participation=0.1,
    )
    c = PortfolioConstraints.from_config(cfg)
    assert c.transaction_cost_bps == 20.0
    assert c.turnover_penalty == 0.5
    assert c.max_participation == 0.1


def test_from_config_defaults_old_style_config_objects_to_disabled():
    """Config objects that predate R8 must still construct (0 = disabled)."""

    class OldConfig:
        long_only = True
        max_weight = 0.05
        max_sector_exposure = 0.30
        max_turnover = 0.30
        lot_size = 100

    c = PortfolioConstraints.from_config(OldConfig())
    assert c.transaction_cost_bps == 0.0
    assert c.turnover_penalty == 0.0
    assert c.max_participation == 0.0
