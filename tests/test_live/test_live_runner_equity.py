"""Regression tests for LiveRunner equity accounting (BUG-02).

`Position.market_value` is written by exactly one method --
`SimulatedBroker.update_price()` via `update_market_prices()` -- and the runner
never called it. Every position therefore kept `market_value == 0`, so
`get_account()["total_equity"]` was just cash, and a cycle that spent 160k on
stock reported a 160k *loss*.

Compounding it, `_generate_signals` emitted `buy` for the whole top-N on every
cycle without looking at the book, so cash drained by ~4% x N per cycle and no
position could ever be exited.
"""

import numpy as np
import pytest
import quant_platform.trading.live_runner as live_runner_module
from quant_platform.core.store import Store
from quant_platform.trading.live_runner import LiveRunner

CODES = ["600519", "000858", "000001", "601318", "600036"]


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """A runner whose Store writes to tmp_path, not the real trading.db.

    `LiveRunner.__init__` calls `Store()` with the default path, so without
    this every test would write sessions and P&L snapshots into the live
    registry.
    """
    monkeypatch.setattr(
        live_runner_module, "Store", lambda *a, **k: Store(str(tmp_path / "live.db"))
    )
    r = LiveRunner(broker_type="simulated", initial_cash=1_000_000, dual_track=False)
    r.set_universe(CODES)
    return r


def _prices(seed=3, n_cycles=5):
    """Deterministic price walk, one dict per cycle."""
    rng = np.random.default_rng(seed)
    base = {c: 100.0 + i * 10 for i, c in enumerate(CODES)}
    out = []
    for _ in range(n_cycles):
        base = {c: p * (1 + rng.normal(0, 0.01)) for c, p in base.items()}
        out.append(dict(base))
    return out


class TestEquityIncludesPositions:
    def test_market_value_is_not_zero_after_buying(self, runner):
        prices = _prices()
        runner.run_once(date="2026-01-05", prices=prices[0])

        acct = runner._broker.get_account()
        assert acct["n_positions"] > 0, "expected the cycle to open positions"
        assert acct["market_value"] > 0, (
            "positions are not marked to market -- total_equity is only cash"
        )

    def test_total_equity_is_cash_plus_positions(self, runner):
        for i, px in enumerate(_prices()):
            runner.run_once(date=f"2026-01-{i + 5:02d}", prices=px)

        acct = runner._broker.get_account()
        assert acct["total_equity"] == pytest.approx(
            acct["cash"] + acct["market_value"], rel=1e-9
        )

    def test_equity_does_not_track_cash_downwards(self, runner):
        """Buying stock is not a loss. Before the fix, equity equalled cash, so
        each cycle's purchases showed up as a drawdown."""
        prices = _prices()
        runner.run_once(date="2026-01-05", prices=prices[0])
        acct = runner._broker.get_account()

        assert acct["total_equity"] > acct["cash"], (
            "equity equals cash -- the purchase is being reported as a loss"
        )
        # Market noise aside, the book should still be worth about what it was.
        assert acct["total_equity"] == pytest.approx(1_000_000, rel=0.02)


class TestRebalanceDoesNotReBuy:
    """The re-buy defect shows up in *concentration*, not in cash.

    Cash turns out to be a weak discriminator: both implementations spend down
    over the first few cycles and then settle. What separates them is that the
    old one keeps topping the same names up to 4% of equity *on every cycle*
    while ignoring what is already held. Measured over 12 cycles:

        fixed    book 13.7% of equity, single names 3.0-3.8%  (target is 4%)
        old      book 45.6% of equity, largest name 25.3%

    A 5-name basket at a 4% target is 20%; anything far above that is re-buying.
    """

    N_CYCLES = 12

    def _run(self, runner, n_cycles):
        for i, px in enumerate(_prices(n_cycles=n_cycles)):
            runner.run_once(date=f"2026-01-{i + 5:02d}", prices=px)

    def test_book_does_not_grow_far_past_the_target_basket(self, runner):
        self._run(runner, self.N_CYCLES)
        acct = runner._broker.get_account()
        exposure = acct["market_value"] / acct["total_equity"]
        assert exposure < 0.30, (
            f"book is {exposure:.1%} of equity; a 5-name basket at 4% each is 20%, "
            f"so names are being re-bought"
        )

    def test_no_single_name_far_exceeds_its_target(self, runner):
        self._run(runner, self.N_CYCLES)
        equity = runner._broker.get_account()["total_equity"]
        for pos in runner._broker.get_positions():
            weight = pos.market_value / equity
            assert weight < 0.10, (
                f"{pos.code} is {weight:.1%} of the book; the target is 4%"
            )
