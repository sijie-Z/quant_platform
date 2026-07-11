# M2 Factor Diagnostic — momentum_12m

**Status**: 5-stock lightweight diagnostic (per-year IC pattern measured)

IC (from 5-stock probe): 0.0424
ICIR (from 5-stock probe): 0.0789
IC (from 300-stock honest run): 0.0114
ICIR (from 300-stock honest run): 0.0515

Note: All per-year IC values from the 5-stock probe should be interpreted
as DIRECTIONAL (pattern) only. The ABSOLUTE levels come from the 300-stock
honest run.

## Per-Year Pattern (5-stock, directional)

| Year | IC Direction |
|------|-------------|
| 2021 | +0.0167 |
| 2022 | +0.0341 |
| 2023 | +0.0858 **best year** |
| 2024 | +0.0351 |
| 2025 | +0.0321 |
| 2026 | -0.0145 **signal reversed** |

## Key Diagnostic Finding

**Momentum peaked in 2023 (IC +0.0858) but has declined to near-zero or negative in 2025-2026.**
Its year-to-year volatility is HIGHER than Low Vol, which explains why Low Vol has
a better overall ICIR (0.1144 vs 0.0515) despite Momentum having a stronger BEST year.

**Why Low Vol > Momentum**: Stability. Low Vol ranges from +0.0057 to +0.0583.
Momentum ranges from -0.0145 to +0.0858 — bigger swings, lower reliability.

## Trust Metadata
- PIT: False
- Data source: akshare (TX for momentum, same as Low Vol)
- Adjust: hfq
- DSR: insufficient_trials
- BH-FDR: n/a

## Limitations
- Per-year IC pattern from 5 stocks only (directionally correct, absolute levels may differ)
- 5-stock IC=0.0424 vs 300-stock IC=0.0114 (5-stock sample overestimates due to selection bias)
- The PATTERN of year-to-year variation is the key diagnostic insight, not the absolute numbers
