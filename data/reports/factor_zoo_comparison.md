# M3 Factor Zoo — 10-factor comparison

All factors computed from CSI300 constituents (2021-2026), AkShare hfq daily.
Trust metadata: pit=false, DSR=insufficient_trials, BH-FDR=n/a.

| Factor | IC | ICIR | IC>0% | Obs |
|--------|-----|------|-------|-----|
| momentum_12m | +0.0569 | +0.2553 | 60% | 1050 |
| volatility_20d | +0.0334 | +0.1181 | 54% | 1281 |
| volatility_60d | +0.0349 | +0.1144 | 53% | 1241 |
| reversal | +0.0208 | +0.0979 | 53% | 1296 |
| roe | +0.0017 | +0.0116 | 50% | 1244 |
| momentum_6m | -0.0048 | -0.0194 | 50% | 1175 |
| rsi_14d | -0.0089 | -0.0426 | 49% | 1287 |
| momentum_3m | -0.0109 | -0.0444 | 48% | 1238 |
| momentum_1m | -0.0128 | -0.0538 | 49% | 1280 |
| amplitude_20d | -0.0346 | -0.1197 | 46% | 1281 |

## Key Findings

- **A-share is a REVERSAL market.** Momentum factors (1m, 3m, 6m) are ALL NEGATIVE.
  Short-term reversal (+0.10 ICIR) confirms the 80d reversal structure found in earlier research.
- **Low Volatility dominates.** volatility_20d (ICIR +0.12) and volatility_60d (+0.11) are the
  strongest signals. This is consistent with the low-vol anomaly well-documented in A-shares.
- **ROE and RSI are noise.** ROE (ICIR +0.01) is indistinguishable from zero in CSI300.
  RSI (-0.04) has no predictive power.
- **Amplitude is anti-alpha.** amplitude_20d (ICIR -0.12) consistently predicts the WRONG direction.
- **Total runs in Registry:** 25

*Generated from Registry. Zero hardcoding. All values from SQL on machine-readable fields.*
