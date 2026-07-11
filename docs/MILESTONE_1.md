# Milestone 1 — Research OS MVP

**Tag**: `v0.1-research-os`  
**Date**: 2026-07-11

## What was verified

Three Honest Research Runs, each following the same Trust → Factor → IC/ICIR → Registry → Report path:

| Factor | Data | IC | ICIR | IC>0% | PIT | Adjust |
|--------|------|-----|------|-------|-----|--------|
| volatility_60d | OHLCV (tx) | 0.0349 | 0.1144 | 53% | false | hfq |
| momentum_12m | OHLCV (tx) | 0.0114 | 0.0515 | 54% | false | hfq |
| roe | Financial (indicator) | 0.0017 | 0.0116 | 50% | false | none |

## What was demonstrated

- [x] Registry can honestly record data source, PIT, adjust, bias_warning, DSR state
- [x] Path reuse: second and third runs required zero framework changes — only content-layer scripts
- [x] Report auto-generation: WARNINGs produced from machine-readable trust fields, not hand-written
- [x] Cross-run comparison: one SQL query answers "which factor has the highest ICIR?" purely from Registry
- [x] Zero existing code modified (only new files in `framework/`, `lab/`, `run_slice.py`)

## Delivered artifacts

| Artifact | Path |
|----------|------|
| Registry | `data/trading.db` → `research_runs` (3 records) |
| Single-run reports | `data/reports/run_*.md` (3 files) |
| Cross-run comparison | `data/reports/factor_comparison.md` |
| Factor diagnostic | `data/reports/diagnostic_volatility_60d.md` |
| Governance | `CONSTITUTION.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `SUCCESS_METRICS.md`, `docs/ADR/0001..0004` |

## What "honest" means here

All three runs have `pit=false`, all three carry `survivorship_bias_possible`. DSR is marked `insufficient_trials`. BH-FDR is n/a for single factors. The system does not pretend otherwise — every report auto-emits the corresponding WARNING.

## Next phase

Milestone 2 — Knowledge Layer: factor diagnostics, multi-factor validation, walk-forward OOS. The Registry moves from recording runs to answering research questions.
