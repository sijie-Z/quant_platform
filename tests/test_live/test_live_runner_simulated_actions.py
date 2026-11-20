"""``main.py trade --broker simulated`` path: load/apply a corporate-action feed.

The CLI simulated run historically started ``LiveRunner`` without any feed,
so a paper session never applied a split or dividend.  ``run`` now accepts
``simulate_corporate_actions`` (passed by ``cmd_trade`` for simulated/paper
brokers): a deterministic demo feed is generated from a dedicated RNG
stream, the synthetic price path is adjusted on each ex-date (so the equity
curve and the factor history stay continuous), and the session report
carries ``corporate_actions_applied`` provenance.  Real brokers never get a
local feed.
"""

from __future__ import annotations

import pytest
from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import BrokerInterface
from quant_platform.trading.live_runner import LiveRunner


def _runner(tmp_path, broker=None, dual_track=False, db="runner.db") -> LiveRunner:
    kwargs = {"broker": broker} if broker is not None else {
        "broker_type": "simulated",
    }
    return LiveRunner(
        dual_track=dual_track,
        initial_cash=1_000_000,
        risk_monitor=RiskMonitor(),
        store=Store(str(tmp_path / db)),
        **kwargs,
    )


def test_simulated_run_applies_the_generated_feed(tmp_path):
    runner = _runner(tmp_path)
    runner.set_universe([f"{600000 + i}" for i in range(10)])

    report = runner.run(
        days=30, seed=42, simulate_corporate_actions=True,
    )
    data = report.to_dict()

    assert data["status"] == "completed"
    provenance = data["corporate_actions_applied"]
    assert provenance["source"] == "synthetic"
    assert provenance["count"] >= 1, (
        "the simulated trading path never applied a split/dividend"
    )
    assert provenance["last_ex_date"]
    # Applied events are persisted for reconciliation, and the feed is
    # visible in the runner state used by the CLI/report.  At least one
    # event must have hit a held position -- zero-holding events are
    # recorded too (entitlement is fixed on the ex-date), so a count alone
    # would not prove the book was adjusted.
    events = runner._store.get_events(topic="corporate_action")
    assert events
    assert any(
        int((event.get("data") or {}).get("quantity_before", 0) or 0) > 0
        for event in events
    ), "no corporate action was applied to a held position"
    state = runner.get_state()
    assert state["corporate_actions_applied"]["count"] == provenance["count"]


def test_simulated_rerun_with_the_same_store_still_applies_actions(tmp_path):
    """A second fresh-book run must not be suppressed by stale markers.

    `LiveRunner` never restores positions, so each run is a new account.
    Persisted `(asset, ex_date)` markers from a previous run (same default
    Store) used to suppress every action while the synthetic price path was
    still adjusted, silently corrupting the demo book.
    """
    counts = []
    for _ in range(2):
        runner = _runner(tmp_path, db="shared_runner.db")
        runner.set_universe([f"{600000 + i}" for i in range(10)])
        report = runner.run(
            days=30, seed=42, simulate_corporate_actions=True,
        )
        counts.append(
            report.to_dict()["corporate_actions_applied"]["count"]
        )
        runner._store.close()

    assert counts[0] >= 1
    assert counts[1] == counts[0], (
        "the second fresh-book run skipped actions because markers from the "
        "first run were persisted under the shared Store key"
    )


def test_runner_without_the_flag_stays_unchanged(tmp_path):
    runner = _runner(tmp_path)
    runner.set_universe([f"{600000 + i}" for i in range(10)])

    report = runner.run(days=5, seed=42)
    data = report.to_dict()

    assert data["corporate_actions_applied"] == {
        "count": 0, "last_ex_date": None, "source": "",
    }


class _RealBrokerStub(BrokerInterface):
    supports_corporate_actions = False

    def __init__(self):
        self.corporate_calls = 0

    def connect(self):
        return True

    def place_order(self, order):
        return order

    def cancel_order(self, order_id):
        return True

    def get_positions(self):
        return []

    def get_orders(self, status=None):
        return []

    def get_account(self):
        return {"cash": 0.0, "total_equity": 0.0, "n_positions": 0}

    def set_corporate_actions(self, actions):  # pragma: no cover - must not run
        self.corporate_calls += 1


def test_real_broker_path_never_applies_actions(tmp_path):
    broker = _RealBrokerStub()
    runner = _runner(tmp_path, broker=broker)
    runner.set_universe(["600519"])

    assert runner.set_corporate_actions(
        [{
            "ex_date": "2024-01-05", "asset": "600519.SH", "kind": "split",
            "cash_per_share": 0.0, "split_ratio": 2.0,
        }],
    ) is False
    runner._advance_exchange_day("2024-01-05")

    assert broker.corporate_calls == 0
    assert runner.get_corporate_actions_applied()["count"] == 0
