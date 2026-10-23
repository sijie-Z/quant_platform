"""The portfolio config knobs must reach the objects that use them.

`portfolio.risk_aversion`, `portfolio.covariance.ewma_half_life` and
`risk.var.confidence` were declared in `config/schema.py` and
`config/default.yaml` but no caller read them: changing them in YAML produced
an identical backtest.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from quant_platform.main import _run_backtest
from quant_platform.utils.config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def test_run_backtest_forwards_portfolio_config(monkeypatch):
    import quant_platform.backtest.engine as engine_mod

    config = load_config(str(DEFAULT_CONFIG))
    config.portfolio.risk_aversion = 3.5
    config.portfolio.covariance.method = "ewma"
    config.portfolio.covariance.ewma_half_life = 21

    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            return {"summary": {}, "daily_returns": None, "weights_history": {}}

    monkeypatch.setattr(engine_mod, "BacktestEngine", FakeEngine)

    _run_backtest(config, None, None, None, None, None, None)

    assert captured["constraints"].risk_aversion == 3.5
    assert captured["covariance_method"] == "ewma"
    assert captured["covariance_ewma_half_life"] == 21


def test_run_backtest_forwards_both_execution_masks(monkeypatch):
    """R6b: main must hand ``tradable_mask`` and ``universe_mask`` to the
    engine unchanged (buys check both, sells check tradability only)."""
    import quant_platform.backtest.engine as engine_mod

    config = load_config(str(DEFAULT_CONFIG))
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            captured.update(kwargs)
            return {"summary": {}, "daily_returns": None, "weights_history": {}}

    monkeypatch.setattr(engine_mod, "BacktestEngine", FakeEngine)

    dates = pd.bdate_range("2024-01-02", periods=3)
    tradable = pd.DataFrame(True, index=dates, columns=["A", "B"])
    universe = pd.DataFrame(True, index=dates, columns=["A", "B"])
    universe.loc[dates[-1], "B"] = False

    _run_backtest(
        config, None, None, None, None, None, None,
        gates={"tradable_mask": tradable, "universe_mask": universe},
    )

    assert captured["tradable_mask"] is tradable
    assert captured["universe_mask"] is universe


def test_default_yaml_ships_the_cost_knobs_disabled():
    """R8 defaults must stay 0 in the shipped config (otherwise the
    'default path unchanged' guarantee silently breaks)."""
    config = load_config(str(DEFAULT_CONFIG))
    constraints = config.portfolio.constraints
    assert constraints.transaction_cost_bps == 0.0
    assert constraints.turnover_penalty == 0.0
    assert constraints.max_participation == 0.0


def test_run_backtest_forwards_cost_aware_constraint_config(monkeypatch):
    """R8: the three cost-aware knobs must reach PortfolioConstraints."""
    import quant_platform.backtest.engine as engine_mod

    config = load_config(str(DEFAULT_CONFIG))
    config.portfolio.constraints.transaction_cost_bps = 20.0
    config.portfolio.constraints.turnover_penalty = 0.75
    config.portfolio.constraints.max_participation = 0.1

    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            return {"summary": {}, "daily_returns": None, "weights_history": {}}

    monkeypatch.setattr(engine_mod, "BacktestEngine", FakeEngine)

    _run_backtest(config, None, None, None, None, None, None)

    constraints = captured["constraints"]
    assert constraints.transaction_cost_bps == 20.0
    assert constraints.turnover_penalty == 0.75
    assert constraints.max_participation == 0.1


def test_engine_forwards_ewma_half_life_to_covariance(monkeypatch):
    import quant_platform.backtest.engine as engine_mod
    from quant_platform.backtest.engine import BacktestEngine

    captured = {}

    def fake_estimate(ret, method="sample", lookback=252, ewma_half_life=63):
        captured["method"] = method
        captured["ewma_half_life"] = ewma_half_life
        return None

    monkeypatch.setattr(engine_mod, "estimate_covariance", fake_estimate)

    dates = pd.bdate_range("2023-01-02", periods=90)
    assets = [f"S{i:02d}" for i in range(12)]
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (len(dates), len(assets))),
        index=dates, columns=assets,
    )
    prices = 100 * (1 + returns).cumprod()
    signal = pd.DataFrame(1.0, index=dates, columns=assets)

    engine = BacktestEngine(
        initial_capital=1_000_000,
        rebalance_frequency="monthly",
        optimizer="equal_weight",
        covariance_method="ewma",
        covariance_lookback=20,
        covariance_ewma_half_life=17,
    )
    engine.run(
        signal=signal, prices=prices, returns=returns,
        benchmark_returns=returns.mean(axis=1), sector_map=None,
    )

    assert captured["method"] == "ewma"
    assert captured["ewma_half_life"] == 17


def _failing_engine(monkeypatch):
    import quant_platform.backtest.engine as engine_mod

    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            captured.update(kwargs)
            return {"summary": {}, "daily_returns": None, "weights_history": {}}

    monkeypatch.setattr(engine_mod, "BacktestEngine", FakeEngine)
    return captured


def test_run_backtest_forwards_volume_and_raw_close_from_metadata(monkeypatch):
    """R9: the volume (shares) and raw close mounted by `_load_data` must reach
    the engine, which converts shares to CNY; the CostModel also gets the
    configured impact model."""
    config = load_config(str(DEFAULT_CONFIG))
    captured = _failing_engine(monkeypatch)

    dates = pd.bdate_range("2024-01-02", periods=3)
    volume = pd.DataFrame(10_000.0, index=dates, columns=["A", "B"])
    raw_close = pd.DataFrame(12.5, index=dates, columns=["A", "B"])
    metadata = pd.DataFrame({"note": ["provenance"]})
    metadata.attrs["daily_volume"] = volume
    metadata.attrs["raw_prices"] = {"close": raw_close, "price_basis": "raw"}

    _run_backtest(
        config, None, None, None, None, None, None, metadata=metadata,
    )

    assert captured["daily_volume"] is volume
    assert captured["raw_prices"] is raw_close
    assert captured["cost_model"]._impact_model_name == config.costs.impact_model
    # The shipped default really is the impact model.
    assert config.costs.slippage_model == "impact"


def test_run_backtest_without_market_panels_fabricates_nothing(monkeypatch):
    """Missing panels must stay missing: the engine then warns and the cost
    model falls back to fixed slippage."""
    config = load_config(str(DEFAULT_CONFIG))
    captured = _failing_engine(monkeypatch)

    metadata = pd.DataFrame({"note": ["no volume"]})
    _run_backtest(
        config, None, None, None, None, None, None, metadata=metadata,
    )

    assert "daily_volume" not in captured
    assert "raw_prices" not in captured
