"""Diagnostic: locate the IC=0 root cause.

Experiments:
1. Future Return Oracle — verify IC calculation correctness
2. Momentum factor IC — test with correct vs incorrect alignment
3. Embedded alpha exposure — check if generator creates detectable signal
"""

import sys
from pathlib import Path

# Add parent to path for quant_platform imports (the grandparent of tools/)
# Project structure: D:\Desktop\...\quant_platform\tools\diagnose_ic.py
# Package root: D:\Desktop\...\quant_platform (parent of tools/ is the project root)
# Parent of project root needs to be in sys.path for "from quant_platform.*" to work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quant_platform.data.providers.synthetic import SyntheticDataProvider


def rank_ic_manual(factor: pd.DataFrame, forward_returns: pd.DataFrame) -> float:
    """Compute mean Rank IC across all dates."""
    ics = []
    for i in range(min(len(factor), len(forward_returns))):
        f = factor.iloc[i].dropna()
        r = forward_returns.iloc[i].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < 30:
            continue
        ic, _ = spearmanr(f[common], r[common])
        ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


print("=" * 70)
print("  DIAGNOSTIC: IC=0 Root Cause Analysis")
print("=" * 70)

# Generate synthetic data
print("\n[1/5] Generating synthetic data (200 stocks, 2021-2022, embedded_alpha=True)...")
provider = SyntheticDataProvider(
    n_stocks=200, start_date="2021-01-01", end_date="2022-12-31",
    embedded_alpha=True,
)
prices = provider.get_prices("2021-01-01", "2022-12-31")
prices_close = prices["close"].unstack("asset")
print(f"  Prices shape: {prices_close.shape}")

returns = prices_close.pct_change(fill_method=None)

# Correct forward returns: return[t+1], last row becomes NaN
forward_returns = returns.shift(-1)

# Wrong forward returns (common mistake): return[t] instead of return[t+1]
same_day_returns = returns.copy()

print("\n[2/5] Experiment A: Future Return Oracle")
print("  Factor = known future return (should give IC ≈ 1.0)")
oracle_ic = rank_ic_manual(forward_returns, forward_returns)
print(f"  Oracle Rank IC (correct): {oracle_ic:.6f}  {'(ok)' if oracle_ic > 0.5 else '(fail)'}")

# Also test: what if oracle is compared to same-day returns?
oracle_ic_wrong = rank_ic_manual(forward_returns, same_day_returns)
print(f"  Oracle Rank IC (wrong alignment): {oracle_ic_wrong:.6f}")

print("\n[3/5] Experiment B: Simple Momentum Factor")
mom_21d = returns.rolling(21).mean()

# Correct: factor[t] vs return[t+1]
ic_correct = rank_ic_manual(mom_21d, forward_returns)
print(f"  Momentum(21d) vs forward return (+1): Rank IC = {ic_correct:.6f}")

# Wrong: factor[t] vs return[t]
ic_wrong = rank_ic_manual(mom_21d, same_day_returns)
print(f"  Momentum(21d) vs same-day return (0): Rank IC = {ic_wrong:.6f}")

print("\n[4/5] Experiment C: Check timing alignment in _generate_returns()")
print("  Looking at how embedded_alpha adds signal to returns...")

# The key question: does alpha at time t affect return at time t or t+1?
# In synthetic.py _generate_returns():
#   momentum_1m[t] = cumulative_base[t-1] - cumulative_base[t-21]  (past 21d return ending at t-1)
#   alpha_return[t] += 0.015 * momentum_1m[t]
#   total_returns[t] = base_returns[t] + alpha_return[t]
#
# So alpha at t affects return at t (contemporaneous).
# But factor momentum_1m uses prices up to close[t] which includes return[t].
# So factor[t] and return[t] share alpha information!
#
# The IC should be: factor[t] vs return[t+1]
# But the alpha affects return[t], not return[t+1].
# This means the alpha is designed for return[t] but IC is measured at return[t+1].
# That's a design issue, not a code bug!

print("\n[5/5] Experiment D: Can we detect the alpha at all?")
print("  Testing: what if we use same-day instead of next-day?")

# Factor value computed at t vs return at t (contemporaneous)
clean_mom = mom_21d.iloc[21:]
clean_ret = returns.reindex(clean_mom.index).dropna(how="all", axis=1)
aligned_f = clean_mom.reindex(clean_ret.index).dropna(how="all", axis=1)
common_cols = aligned_f.columns.intersection(clean_ret.columns)
aligned_f = aligned_f[common_cols]
aligned_r = clean_ret[common_cols]

ics_same = []
for i in range(min(len(aligned_f), len(aligned_r))):
    f = aligned_f.iloc[i].dropna()
    r = aligned_r.iloc[i].reindex(f.index).dropna()
    common = f.index.intersection(r.index)
    if len(common) < 30:
        continue
    ic, _ = spearmanr(f[common], r[common])
    ics_same.append(ic)

ic_contemp = float(np.mean(ics_same)) if ics_same else 0.0
print(f"  Momentum(21d) vs same-day return (contemporaneous): IC = {ic_contemp:.6f}")

# Test with stronger alpha
print("\n  Testing with stronger alpha signal...")
from quant_platform.data.providers.synthetic import SyntheticDataProvider as SDP
# Create provider with custom data to test strong alpha
# We can't easily modify the alpha strength externally, so let's check
# what the alpha strength actually is

print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
if oracle_ic > 0.5:
    print("  (ok) Oracle test passes: IC calculation is correct")
else:
    print("  (fail) Oracle test fails: IC calculation or data alignment broken")

if ic_correct > 0.02:
    print(f"  (ok) Momentum IC is positive ({ic_correct:.4f}): factors have predictive power")
elif ic_correct > 0.01:
    print(f"  (!)️  Momentum IC is weak ({ic_correct:.4f}): signal exists but very noisy")
else:
    print(f"  (fail) Momentum IC ~ 0 ({ic_correct:.4f}): factor has no predictive power")

if ic_contemp > 0.05:
    print(f"  (ok) Contemporaneous IC ({ic_contemp:.4f}): alpha affects same-day return as designed")
    print("  => The embedded_alpha is designed for same-day effect, not prediction.")
    print("  => Current IC calculation (factor[t] vs return[t+1]) is measuring prediction,")
    print("     which is NOT what embedded_alpha provides.")
    print("  => This is a DESIGN issue: embedded_alpha adds signal to return[t] but IC")
    print("     measures factor[t] vs return[t+1]. The alpha is there but the timing")
    print("     alignment makes it invisible to the standard IC calculation.")
elif oracle_ic < 0.5:
    print("  (fail) Something fundamentally broken in the data pipeline")
else:
    print(f"  (!)️  Contemporaneous IC is weak ({ic_contemp:.4f})")
