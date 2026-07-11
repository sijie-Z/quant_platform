# Factor Diagnostic Report — volatility_60d (Low Vol)

Generated: 2026-07-11T17:24:41 | Diagnostic type: M2 Factor Diagnostic
IC observations: 1241
Overall IC mean: 0.0349 | ICIR: 0.1144
Registry run_id: diag_vol60d_1783761881

## Per-Year IC
| Year | N    | IC Mean | IC Std |
|------|------|---------|--------|
| 2021 | 183 | +0.0057 | 0.2334 |
| 2022 | 242 | +0.0456 | 0.2708 |
| 2023 | 242 | +0.0533 | 0.2681 |
| 2024 | 242 | +0.0583 | 0.3598 |
| 2025 | 243 | +0.0113 | 0.3522 |
| 2026 | 89 | +0.0165 | 0.3139 |
## Rolling 60-Day IC
| Min | Max | Latest |
|-----|-----|--------|
| -0.0588 | 0.1445 | 0.0184 |

## Monthly Stability
- Positive months: 60%
- Month count: 62
- Monthly IC mean: 0.0318
- Monthly IC std: 0.0744

## Data Quality
- Average asset coverage: 91.5%
- Min coverage date: 0.0%
- Days with >=80% coverage: 1242 / 1302 (95%)
- Fetch duration: 1s

## Trust Metadata
- PIT: False (current constituent snapshot)
- Bias Warning: survivorship_bias_possible
- Adjust: hfq (back-adjusted, ipo_date_per_stock)
- Provider: akshare (TX source)
- DSR: insufficient_trials
- BH-FDR: n/a

## Key Findings
- **Best years**: 2023-2024 (IC ~0.05-0.06), consistent signal above noise
- **Signal degraded** in 2025 (IC ~0.0113) — possible alpha decay or market regime shift
- **Positive months**: 60% — better than random but not robust
- **Coverage**: 91.5% assets, 95% days above 80% — adequate free-source quality
- **Sharpe proxy** negative despite positive IC → long-short construction needs refinement
- **IC volatility**: intra-month noise (0.305) high relative to signal (0.0349)

_Findings from M2 diagnostic pipeline. All values from raw computation on real A-share CSI300 data (2021-2026)._
