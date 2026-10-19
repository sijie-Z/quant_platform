# Dead-code and contract status (2026-10)

Issue #77 (and #70) found seven packages with zero runtime importers.
Deleting them is a product decision, not a bug fix, so this file records the
decision for each — "is this wired?" should be answerable without re-running
an AST scan.

| Package | Runtime importers | Decision |
|---|---|---|
| `framework/contracts/` | 0 (self + tests) | **PROPOSAL** — six Protocols, zero conforming production implementations. |
| `services/` | 0 (self only) | **PROPOSAL** — standalone microservice skeletons; `app.py`/`main.py` never start them. |
| `kernel/` | 0 (self only) | **PROPOSAL** — `kernel/kernel.py` is unwired; `daemon/runner.py` uses `LiveEngine` directly. |
| `shared/` | 0 | **COMPATIBILITY SHIM** — re-exports `quant_core.*`. |
| `regime_router/` | 0 (validation scripts only) | **RESEARCH ONLY**. |
| `compliance/` | tests only | **TESTS ONLY** — wire into the API or delete deliberately. |
| `research/` | tests only | **TESTS ONLY**. |

Each package's `__init__.py` now starts with a machine-greppable `STATUS:`
line, enforced by `tests/test_architecture/test_dead_code_status.py`.

Deletion is deliberately deferred: these modules have tests and are cited by
historical documents, so removing them should be its own change with an
explicit **wire / delete** decision. Do not treat any of them as production
functionality in the meantime.
