# Forward paper track 2026 (R31)

**Status: waiting for the execution session.** The target book below was
generated from data through **2026-10-23**; the ledger cannot start until the
first cached session *after* that date exists. The expected execution day is
**2026-10-26** (Monday) at the close.

**paper tracking only; style-tilt demo, not validated alpha**.

## Target

- JSON: `docs/research/target_portfolio_2026-10-23.json`
- CSV: `docs/research/target_portfolio_2026-10-23.csv`
- `as_of` / `signal_date` / `data_end`: 2026-10-23
- Holdings: 32; gross weight 99.9999%;
  max single weight 5.0000% (5% cap);
  max sector weight 13.4924% (30% cap);
  first build (no `--prev-weights`).
- Estimated one-sided turnover of the first build:
  50.00% (against a zero book).
- Effective valuation age on 2026-10-23 (Baidu points are sparse):
  median 7 days,
  max 21 days.
  The 2026-10-24 source point is excluded by construction (strict PIT).

Top 10 weights:

| code | weight |
|---|---:|
| 002508 | 5.0000% |
| 600436 | 5.0000% |
| 002650 | 5.0000% |
| 000883 | 5.0000% |
| 002168 | 5.0000% |
| 002262 | 5.0000% |
| 600872 | 5.0000% |
| 000543 | 5.0000% |
| 600817 | 5.0000% |
| 000755 | 5.0000% |

## How to start the ledger once data updates

1. Extend the Sina daily cache (`.scratch/wide_sina_2026/daily_wide.parquet`)
   through at least the execution day and refresh the Baidu valuation cache.
2. Rebuild the merged panel (causal ffill, no bfill):

   ```
   python .scratch/orchestrate/r31_wide_value.py --merge
   ```

3. Sanity-check the frozen target: re-run `daily_signal` with
   `--as-of 2026-10-23` against the refreshed cache and confirm the weights
   are unchanged. Post-as_of rows must not move the book (R31's PIT test uses
   exactly this construction). Weights are rounded to 1e-10; compare with a
   tolerance of 1e-8, or set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` for bit-level equality (R31
   pins these threads for every research run). Do **not** overwrite this file
   with a later-`as_of` book; write a new dated target instead (paper_track
   refuses to mix targets in one ledger).
4. Run the forward ledger:

   ```
   python tools/paper_track.py \
     --target docs/research/target_portfolio_2026-10-23.json \
     --cache .scratch/orchestrate/merged_wide.parquet \
     --out-dir docs/research/paper_track/forward_2026 \
     --capital 2000000 --cost-mode impact
   ```

The tool writes `nav.csv` and `summary.json` into this directory. It is
idempotent: re-running after more sessions appends only the new rows. A target
whose `as_of` equals the cache's last date is reported as *pending* and is
executed on the next data refresh.

## Known limitations (disclosed, not fixed here)

- The wide cache `close` is Sina **hfq** (back-adjusted). `paper_track.py`
  estimates CNY daily volume as `volume x close`, which overstates the true
  traded value on this cache and therefore **understates** the impact cost
  (R30 measured a median 2.78x ADV overstatement). NAV returns themselves are
  built from `pct_chg` and are unaffected. Use `--cost-mode fixed` for a
  conservative sensitivity check if needed.
- Survivorship: the 300-name Sina universe contains only currently listed
  names (delisted names were excluded before sampling). Forward paper tracking
  is unaffected by historical survivorship, but the research evidence is an
  upper bound.
- Static industry map (latest classification, no `effective_date`), no ST
  filter, no board-lot rounding (weight-level book).
- No broker is called, no order is placed, no money moves.
