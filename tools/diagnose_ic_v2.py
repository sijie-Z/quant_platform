"""Diagnostic v2: pin down why pipeline outputs IC=0 but manual test finds IC>0.

Tests:
1. Replicate the EXACT pipeline flow step by step
2. Compare IC before and after EACH processing step
3. Identify the exact step where IC disappears
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.synthetic import SyntheticDataProvider
from quant_platform.factors.evaluation import rank_ic, ic_summary
from quant_platform.factors.processing import process_factor
from quant_platform.factors.technical import Momentum1M
from quant_platform.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

print("=" * 70)
print("  DIAGNOSTIC V2: Exact pipeline replication")
print("=" * 70)

# Use the EXACT same config as the pipeline
config_universe = {
    "n_stocks": 500,
    "exclude_st": True,
    "exclude_suspended": True,
}
config_data = {
    "provider": "synthetic",
    "start_date": "2021-01-01",
    "end_date": "2025-12-31",
    "synthetic": {"embedded_alpha": True},
}

print(f"\n[Step 0] Config: {config_data['start_date']} to {config_data['end_date']}, "
      f"{config_universe['n_stocks']} stocks, embedded_alpha=True")

# Step 1: Load data through the exact pipeline
print(f"\n[Step 1] Loading data through DataPipeline...")
provider = SyntheticDataProvider(
    n_stocks=config_universe["n_stocks"],
    start_date=config_data["start_date"],
    end_date=config_data["end_date"],
    embedded_alpha=config_data["synthetic"]["embedded_alpha"],
)
pipeline = DataPipeline(
    provider=provider,
    start_date=config_data["start_date"],
    end_date=config_data["end_date"],
    exclude_st=config_universe["exclude_st"],
    exclude_suspended=config_universe["exclude_suspended"],
)
pipeline.run()

prices = pipeline.get_close()
returns = pipeline.returns  # THIS is what the pipeline uses
print(f"  prices: {prices.shape}, returns: {returns.shape}")

# Step 2: The pipeline computes forward returns itself? No — it uses pipeline.returns
# Let's compute raw forward returns too for comparison
calc_ret = prices.pct_change(fill_method=None)

# Step 3: Compute momentum_1m the exact way the pipeline does
print(f"\n[Step 2] Computing momentum_1m factor...")
mom = Momentum1M()
raw_factor = mom.compute(prices)
print(f"  raw factor: {raw_factor.shape}")

# Step 4: IC BEFORE any processing (using pipeline returns)
print(f"\n[Step 3] IC BEFORE processing (using pipeline.returns):")
common = raw_factor.index.intersection(returns.index)
ic_before = rank_ic(raw_factor.loc[common], returns.loc[common])
s_before = ic_summary(ic_before)
print(f"  momentum_1m IC={s_before['mean_ic']:.6f} ICIR={s_before['icir']:.4f}")

# Step 5: IC AFTER processing
print(f"\n[Step 4] IC AFTER processing (winsorize+standardize+neutralize):")
sector_map = provider.get_metadata()["sector"]
fin = provider.get_financials(config_data["start_date"], config_data["end_date"])
fin_unstacked = fin.unstack("asset") if fin is not None else None
mcap = fin_unstacked.get("market_cap") if fin_unstacked is not None else None

proc_factor = process_factor(raw_factor, sector_map=sector_map, market_cap=mcap)
common_p = proc_factor.index.intersection(returns.index)
proc = rank_ic(proc_factor.loc[common_p], returns.loc[common_p])
s_proc = ic_summary(proc)
print(f"  momentum_1m IC={s_proc['mean_ic']:.6f} ICIR={s_proc['icir']:.4f}")

# Step 6: Check if data is aligned correctly
print(f"\n[Step 5] Data alignment check:")
print(f"  raw_factor index[0]: {raw_factor.index[0]}, returns index[0]: {returns.index[0]}")
print(f"  raw_factor index[-1]: {raw_factor.index[-1]}, returns index[-1]: {returns.index[-1]}")

# Check that returns[t] corresponds to price return t->t+1
print(f"\n[Step 6] What does pipeline.returns[date] actually represent?")
t0 = returns.index[0]
t1 = returns.index[1]
# returns[t0] should be return from t0 to t1
p0 = prices.loc[t0]
p1 = prices.loc[t1]
actual_ret = p1 / p0 - 1
print(f"  returns[{t0.date()}] for stock 000001.SH: {returns.loc[t0, '000001.SH']:.6f}")
print(f"  actual (p[{t1.date()}]/p[{t0.date()}] - 1): {actual_ret['000001.SH']:.6f}")
print(f"  pct_change[{t0.date()}] for 000001.SH: {calc_ret.loc[t0, '000001.SH']:.6f}")

# CONCLUSION
print()
print("=" * 70)
print("  CONCLUSION")
print("=" * 70)
if abs(s_before['mean_ic']) > 0.01:
    print(f"  (ok) Factor IC IS positive ({s_before['mean_ic']:.4f}) using pipeline.returns")
    print("  => The factor engine and IC calculation are WORKING")
    print("  => The original main.py output showing IC=0 needs investigation")
    print("     but the underlying computation is correct")
else:
    print(f"  (fail) Factor IC is still ~0 ({s_before['mean_ic']:.4f})")
    print("  => Something is being lost between my manual test and the pipeline flow")
