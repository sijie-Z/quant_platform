"""Ops wave 2 regressions — daemon graceful shutdown (S16).

The daemon caught only KeyboardInterrupt and never closed the store; a
SIGTERM (docker stop) abandoned the snapshot and the WAL.
"""

from __future__ import annotations

import json
from pathlib import Path

from quant_platform.core.store import Store
from quant_platform.daemon.runner import Daemon


def _daemon(tmp_path) -> Daemon:
    state_path = tmp_path / "daemon_state.json"
    return Daemon(
        store=Store(str(tmp_path / "daemon.db")),
        state_path=state_path,
    )


def test_close_persists_state_and_closes_the_store(tmp_path):
    daemon = _daemon(tmp_path)
    daemon.last_run_date = "2026-10-19"

    daemon.close()

    state_path = Path(daemon._state_path)
    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["last_run_date"] == "2026-10-19"
    # close() closed this thread's connection; a new connection still works.
    assert daemon.store.get_stats()["backend"] == "sqlite"


def test_run_forever_closes_on_interrupt(tmp_path):
    daemon = _daemon(tmp_path)

    def _interrupt():
        raise KeyboardInterrupt

    daemon.should_run_today = _interrupt

    daemon.run_forever(interval_seconds=0)

    assert Path(daemon._state_path).exists()


def test_close_is_idempotent(tmp_path):
    daemon = _daemon(tmp_path)

    daemon.close()
    daemon.close()

    assert Path(daemon._state_path).exists()
