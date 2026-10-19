# First real research run — 80-day reversal

> **Run date**: 2026-10-19
> **Registry run id**: `run_1792393532_37320d14c0bff82c`
> **Status**: pre-registered before the results were computed; OOS = 2023-01 onward.

## 1. Pre-registered definition

| Item | Definition |
|---|---|
| Data | Baostock daily close, back-adjusted (`adjustflag="1"`, hfq) |
| Universe | HS300 constituents queried at every month-end, 2020-01..2025-12 |
| PIT | Yes — each month uses the constituent list effective on that date |
| Factor | `reversal_80d = -(close[t] / close[t-80 trading days] - 1)` |
| IC target | forward 80-trading-day return |
| Backtest | monthly, long top 20%, equal weight, 15 bps per unit of traded value (charged on 2 x one-sided turnover) |
| OOS | 2023-01-01 onward |

The 80-day horizon was fixed from `docs/research/MARKET_STRUCTURE_DISCOVERY_V3_REPORT.md`
(IC −0.081, t −4.58). No other horizon or parameter was tried in this run.

## 2. Rank IC

| Sample | Months | Mean IC | ICIR | t-stat | IC > 0 |
|---|---:|---:|---:|---:|---:|
| Full 2020-2025 | 68 | -0.0136 | -0.066 | -0.54 | 38.2% |
| In-sample 2020-2022 | 36 | -0.0023 | -0.012 | -0.07 | 41.7% |
| Out-of-sample 2023-2025 | 32 | -0.0263 | -0.120 | -0.68 | 34.4% |

The t-stat treats monthly ICs as independent. Forward 80-day returns overlap
across adjacent months, so the effective sample is smaller and the true
t-stat is lower than shown; the OOS column is the one that matters.

## 3. Cost-aware monthly backtest

| Portfolio | Ann. return | Ann. vol | Sharpe | Max DD | Hit rate |
|---|---:|---:|---:|---:|---:|
| Reversal top quintile, gross | 3.11% | 25.42% | 0.12 | -36.79% | 42.3% |
| Reversal top quintile, net | 1.60% | 25.43% | 0.06 | -38.15% | 42.3% |
| HS300 equal-weight benchmark | 5.31% | 18.79% | 0.28 | -29.41% | 52.1% |

Average one-sided monthly turnover: 41.1%.

## 4. What this says about the pre-registered hypothesis

Our factor is the **negative** of the past 80-day return, so a *positive* IC
means reversal works. The observed IC is -0.0136
(t=-0.54) full-sample and -0.0263
(t=-0.68) out-of-sample. The sign is **opposite** to the V3
report's raw past-return IC (−0.081): on this PIT HS300 sample the raw
past-return IC is +0.0136, i.e. mild momentum, and it is not
statistically distinguishable from zero.

**Conclusion: the 80-day reversal result does not reproduce in this
specification.** The pre-registered strategy underperforms its equal-weight
benchmark after costs. That is the honest answer this run was designed to
produce — not a positive alpha claim.

## 5. Limitations (read before citing)

- **ST stocks are not filtered point-in-time.** Baostock does not expose a
  historical ST flag; names that are ST today are not removed from history.
- Price limits and trading halts are not modelled; a real book cannot trade
  every name at the month-end close.
- Membership is HS300 PIT, so the study says nothing about stocks that were
  never index members.
- The factor and the OOS split were fixed before this run; no parameter
  search was performed, but the horizon itself was chosen using earlier
  research on the same market (a mild selection risk, disclosed).
- Costs use a flat 15 bps per unit of traded value (charged on two-way
  turnover); stamp tax, minimum commission and market impact are not itemised.

## Reproduction

```bash
python -m quant_platform.lab.runs.first_real_research_80d_reversal
```
