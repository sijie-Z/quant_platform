"""Verify the frozen forward artifacts against their registered hashes.

The forward paper window is a pre-registered experiment: the running code,
the inception targets and the PIT ST list must not change between the freeze
and the 3/6-month evaluation. This read-only guard:

- recomputes ``sha256[:16]`` for every frozen artifact;
- pins each ledger's row count + prefix hash so past rows cannot be edited
  after the fact (a full rebuild that appends is fine; rewriting is not);
- pins every forward target the first time it is seen, so each month-end
  target is locked at creation and cannot be silently regenerated later;
- exits non-zero on any drift so a scheduled task can alarm instead of
  silently tracking a different strategy.

The registered hashes mirror ``docs/research/FREEZE_MANIFEST_2026-10-25.md``
(§6 v3 plus the §7 v3.1 superseded entries). When a subsequent section
supersedes a hash, update the table below in the same change.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs" / "research" / "paper_track"
FREEZE_DATE = "2026-10-23"

FROZEN = {
    "tools/daily_signal.py": "ee4773925b1a7d5f",
    "tools/paper_track.py": "ef4d8323d38375e6",
    "tools/forward_update.py": "1d97834af3e14448",
    "backtest/engine.py": "d937c3403d83c2fd",
    "backtest/cost_model.py": "18a71fb6cec9e2be",
    "portfolio/optimizers.py": "1e703bd95fd68a57",
    "portfolio/covariance.py": "20a51b273adc9550",
    "portfolio/constraints.py": "7e1c65c80599921b",
    "factors/processing.py": "467221452b51130b",
    "factors/evaluation.py": "0a8c0dfb1d08878e",
    "backtest/metrics.py": "49399a9141d95171",
    "utils/exchange_calendar.py": "00260b954308468a",
    "tools/fetch_valuation.py": "a286f5449fba5bc7",
    ".cache/baostock/industry_map.parquet": "5da906a2e4bca0c0",
    "docs/research/paper_track/forward_targets/turnover/target_2026-10-23.json":
        "bdfb3e7c814aa3b7",
    "docs/research/paper_track/forward_targets/ep_turnover/target_2026-10-23.json":
        "b4d49582459af14c",
    "docs/research/paper_track/forward_targets/st_codes_2026-10-23.txt":
        "b42b7ab7a67a28f6",
}

TARGET_DIRS = [
    BASE / "forward_targets" / "turnover",
    BASE / "forward_targets" / "ep_turnover",
]

LEDGERS = [
    BASE / "forward_2026" / "nav.csv",
    BASE / "forward_2026_turnover" / "nav.csv",
]

STATE = ROOT / ".scratch" / "orchestrate" / "freeze_guard_state.json"


def digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix != ".parquet":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:16]


def prefix_digest(path: Path, lines: int) -> tuple[str, int]:
    content = path.read_text(encoding="utf-8").splitlines(keepends=True)
    head = content[:lines]
    normalized = "".join(head).replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest(), len(content)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accept", action="store_true",
        help="after human review of a reported drift, rebaseline the state",
    )
    args = parser.parse_args()
    problems: list[str] = []

    for relative, expected in FROZEN.items():
        path = ROOT / relative
        if not path.is_file():
            problems.append(f"missing frozen artifact: {relative}")
            continue
        actual = digest(path)
        if actual != expected:
            problems.append(
                f"hash drift: {relative} expected {expected}, got {actual}"
            )

    state = {}
    if STATE.is_file():
        state = json.loads(STATE.read_text(encoding="utf-8"))
    pinned_targets = {
        relative: expected for relative, expected in FROZEN.items()
        if relative.startswith("docs/research/paper_track/forward_targets/")
    }
    pinned_targets.update(state.get("targets") or {})
    new_targets = {}
    for directory in TARGET_DIRS:
        for path in sorted(directory.glob("target_*.json")):
            relative = path.relative_to(ROOT).as_posix()
            actual = digest(path)
            as_of = path.stem[len("target_"):]
            pinned = pinned_targets.get(relative)
            if pinned is not None:
                if pinned != actual:
                    problems.append(
                        f"target rewritten after first sight: {relative}"
                    )
                new_targets[relative] = pinned
            elif as_of <= FREEZE_DATE:
                problems.append(f"unregistered historical target: {relative}")
            else:
                new_targets[relative] = actual

    new_state = {"targets": new_targets}
    for ledger in LEDGERS:
        key = ledger.relative_to(ROOT).as_posix()
        if not ledger.is_file():
            problems.append(f"missing ledger: {key}")
            continue
        previous = state.get(key) or {}
        lines = int(previous.get("lines", 0))
        if lines:
            head_hash, _ = prefix_digest(ledger, lines)
            if head_hash != previous.get("head_hash"):
                problems.append(
                    f"ledger history rewritten: {key} first {lines} lines changed"
                )
        head_hash, total = prefix_digest(ledger, 10**9)
        new_state[key] = {"lines": total, "head_hash": head_hash}

    if problems:
        for problem in problems:
            print(f"DRIFT {problem}")
        if not args.accept:
            print("state not updated: drift alarms persist until a person resolves it")
            return 1
        print("drift accepted after review: rebaselining state")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    print(f"OK {len(FROZEN)} frozen artifacts, "
          f"{len(new_targets)} targets pinned, {len(LEDGERS)} ledgers append-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
