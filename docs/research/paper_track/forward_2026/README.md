# Forward paper track 2026

**Status: waiting for the first real execution session.** The active targets
were generated from data through **2026-10-23** and execute at the close of
**2026-10-26**, the first trading session after their `as_of`. This directory
must not be seeded from the superseded single-target files.

**Paper tracking only; style-tilt demo, not validated alpha. No broker is
called and no money moves.**

## Active versioned targets

| Strategy | Targets directory | Ledger directory |
|---|---|---|
| EP + turnover | `docs/research/paper_track/forward_targets/ep_turnover/` | `docs/research/paper_track/forward_2026/` |
| Turnover only | `docs/research/paper_track/forward_targets/turnover/` | `docs/research/paper_track/forward_2026_turnover/` |

The active `target_2026-10-23.json` files are the complete-ST-snapshot version:
13 names present in the 300-stock universe with Baostock `isST=1` on
2026-10-23 were excluded before construction. Earlier v1/v2 files are kept
only under each target directory's `superseded/` subdirectory; do not move
them back into the active glob. Exact hashes and the correction history are in
`docs/research/FREEZE_MANIFEST_2026-10-25.md`.

## Daily update (after the exchange close)

```bash
python tools/forward_update.py
```

This refreshes the Sina cache, rebuilds `.scratch/orchestrate/merged_wide.parquet`,
creates a new month-end target when due, and rebuilds both rolling ledgers.
There is no `--track` argument. Month-end target generation requires a complete
current ST snapshot covering the universe; snapshot, target-generation, or
ledger failures produce a nonzero exit. A first rebalance with zero executed
buys is rejected rather than published as an all-cash inception ledger.

The desktop Scheduled task is `quant-forward-paper-daily` (weekdays 16:20).
Use **Run now** once in the Scheduled panel to pre-approve its tools.

## Manual ledger rebuild

```bash
python tools/paper_track.py --targets-dir docs/research/paper_track/forward_targets/turnover --cache .scratch/orchestrate/merged_wide.parquet --out-dir docs/research/paper_track/forward_2026_turnover --capital 2000000 --cost-mode impact --raw-price-mode auto
```

For the EP+turnover ledger:

```bash
python tools/paper_track.py --targets-dir docs/research/paper_track/forward_targets/ep_turnover --cache .scratch/orchestrate/merged_wide.parquet --out-dir docs/research/paper_track/forward_2026 --capital 2000000 --cost-mode impact --raw-price-mode auto
```

## Known limitations

- The 300-name universe contains currently listed names only; historical
  research retains survivorship bias. Forward tracking itself starts now.
- Valuation history is still the Baidu panel with causal forward fill; industry
  mapping is static. These are research limitations, not reasons to rewrite a
  frozen target after tracking begins.
- Execution is a next-session-close model at weight level, without board-lot
  rounding or intraday queue simulation. Costs use the platform impact model;
  `--raw-price-mode auto` selects `amount/volume` for the Sina HFQ panel.
- Corrections require new versioned targets/ledgers. Do not overwrite an
  earlier target in place or rewrite inception history.
