# results/ — historical artifacts (void)

**Everything in this directory was produced before the 2026-09 honesty
cleanup. Treat every performance number here as void.**

The cleanup fixed defects that made prior backtests meaningless:

- the alpha signal was empty (`ma_convergence` returned a `Series` and turned
  the whole cross-section into NaN, so every backtest fell back to
  equal-weight buy-and-hold);
- `embedded_alpha` in the nested config was never read;
- execution timing, cost model, walk-forward OOS window, ST/PIT filters and
  several accounting paths were wrong.

`backtest_report.html` is the concrete example: it reports **Sharpe 1.45 with
`Rebalances 0`**, which is not a description of a rebalanced strategy. A
visible VOID banner has been added to its body.

Do not delete these files — they are the record of what changed. If you need
current numbers, regenerate them from the fixed code:

```bash
python main.py run --force
```

See `docs/HANDOVER.md` §2 and Issues #54, #60, #72.
