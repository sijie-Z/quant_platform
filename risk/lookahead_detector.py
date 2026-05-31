"""Lookahead Bias Detector for Factor Pipeline.

Inspired by freqtrade's lookahead-analysis tool, this module detects
future data leakage (lookahead bias) in the factor computation and
alpha generation pipeline.

How it works:
1. For each factor, compute values using the FULL dataset.
2. For each rebalance date T, recompute the factor using only data up to T.
3. Compare: if the factor value at T differs between the two runs, there may
   be lookahead bias in that factor.

The factor computation itself (rolling windows) is inherently free of
lookahead bias. The risk is in:
  - Factor processing (winsorize/standardize/neutralize) using cross-sectional
    statistics that might be influenced by future outliers
  - Alpha combination using IC/ICIR weights that might use future IC data
    (our code uses point-in-time IC, but this detector verifies it)
  - Any global computation that aggregates across all dates

This detector is a VERIFICATION tool, not a fix. It proves that the
pipeline is free of lookahead bias, or flags where bias exists.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class LookaheadDetector:
    """Detect lookahead bias in factor computation and signal generation.

    For each date in the signal history, re-computes the signal using
    only data available up to that date and compares with the full-data signal.

    Attributes:
        signal_diff: DataFrame showing signal difference at each date.
        factor_diffs: Dict mapping factor name → DataFrame of factor differences.
        biased_dates: List of dates where signal bias was detected.
    """

    def __init__(
        self,
        threshold: float = 1e-4,
        sample_fraction: float = 1.0,
        max_check_dates: int = 50,
    ):
        self.threshold = threshold
        self.sample_fraction = sample_fraction
        self.max_check_dates = max_check_dates
        self.signal_diff: pd.DataFrame | None = None
        self.factor_diffs: dict[str, pd.DataFrame] = {}
        self.biased_dates: list[pd.Timestamp] = []

    def detect(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        financials: pd.DataFrame | None,
        metadata: pd.DataFrame | None,
        config: Any,
    ) -> dict[str, Any]:
        """Run lookahead bias detection on the full pipeline.

        Args:
            prices: (date × asset) price DataFrame.
            returns: (date × asset) forward returns.
            financials: Optional (date × asset) financial data.
            metadata: Optional metadata.
            config: Loaded config object.

        Returns:
            Dict with:
                has_bias: bool
                biased_dates: list of date strings
                max_signal_diff: float
                signal_diff_summary: DataFrame
                factor_bias_report: dict
        """
        from quant_platform.main import _compute_factors, _generate_signal

        logger.info("=" * 60)
        logger.info(" Lookahead Bias Detection")
        logger.info("=" * 60)

        # ── Step 1: Compute full-data factors and signal ──
        logger.info("[1/4] Computing full-data factors...")
        full_factors, full_ic, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata,
            turnover=None, config=config,
        )
        logger.info(
            "  Computed %d factors, %d dates, %d assets",
            len(full_factors), len(prices), len(prices.columns),
        )

        logger.info("[2/4] Generating full-data signal...")
        full_signal = _generate_signal(config, full_factors, returns,
                                       prices=prices)
        logger.info("  Signal shape: %s", full_signal.shape)

        # ── Step 2: Determine dates to check ──
        check_dates = self._select_dates(full_signal)
        logger.info(
            "[3/4] Checking %d dates for lookahead bias...",
            len(check_dates),
        )

        # ── Step 3: For each date, recompute with truncated data ──
        signal_diffs = []
        factor_diffs: dict[str, list] = {
            name: [] for name in full_factors
        }
        biased_dates = []

        for i, date in enumerate(check_dates):
            # Truncate data
            trunc_idx = prices.index <= date
            trunc_prices = prices.loc[trunc_idx]
            trunc_returns = returns.loc[trunc_idx]

            if len(trunc_prices) < 30:  # Minimum data requirement
                logger.debug("  Skipping %s: only %d days of data", date, len(trunc_prices))
                continue

            # Recompute factors with truncated data
            trunc_factors, _, _, _ = _compute_factors(
                trunc_prices, trunc_returns, financials, metadata,
                turnover=None, config=config,
            )

            # Recompute signal with truncated data
            trunc_signal = _generate_signal(
                config, trunc_factors, trunc_returns, prices=trunc_prices,
            )

            # Compare factors at this date
            date_factor_diff = {}
            max_factor_bias = 0.0
            for name in full_factors:
                if name not in trunc_factors:
                    continue
                if date not in trunc_factors[name].index:
                    continue

                full_vals = full_factors[name].loc[date]
                trunc_vals = trunc_factors[name].loc[date]

                # Align assets
                common = full_vals.index.intersection(trunc_vals.index)
                diff = (full_vals[common] - trunc_vals[common]).abs().max()
                date_factor_diff[name] = float(diff) if not pd.isna(diff) else 0.0
                factor_diffs[name].append(float(diff) if not pd.isna(diff) else 0.0)
                max_factor_bias = max(max_factor_bias, date_factor_diff[name])

            # Compare signal at this date
            if date in trunc_signal.index:
                full_sig = full_signal.loc[date]
                trunc_sig = trunc_signal.loc[date]
                common = full_sig.index.intersection(trunc_sig.index)
                diff = (full_sig[common] - trunc_sig[common]).abs().max()
                signal_diff = float(diff) if not pd.isna(diff) else 0.0
            else:
                signal_diff = float("nan")

            signal_diffs.append({
                "date": date,
                "signal_diff": signal_diff,
                "max_factor_diff": max_factor_bias,
                "biased_factors": [
                    n for n, d in date_factor_diff.items() if d > self.threshold
                ],
            })

            if signal_diff > self.threshold:
                biased_dates.append(date)
                logger.info(
                    "  ⚠  %s — signal diff=%.6f (factors: %s)",
                    date.date(), signal_diff,
                    ", ".join(signal_diffs[-1]["biased_factors"][:5]),
                )
            else:
                logger.debug("  ✓ %s — signal diff=%.4e", date.date(), signal_diff)

            if (i + 1) % 5 == 0:
                logger.info("  Progress: %d/%d dates checked", i + 1, len(check_dates))

        # ── Step 4: Compile report ──
        logger.info("[4/4] Compiling results...")

        diff_df = pd.DataFrame(signal_diffs).set_index("date") if signal_diffs else pd.DataFrame()

        # Factor bias summary
        factor_bias_report = {}
        for name, diffs in factor_diffs.items():
            diffs_arr = np.array(diffs, dtype=float)
            factor_bias_report[name] = {
                "max_diff": float(np.max(diffs_arr)) if len(diffs_arr) > 0 else 0.0,
                "mean_diff": float(np.mean(diffs_arr)) if len(diffs_arr) > 0 else 0.0,
                "pct_biased": float(
                    np.mean(diffs_arr > self.threshold)
                ) if len(diffs_arr) > 0 else 0.0,
            }

        has_bias = len(biased_dates) > 0
        max_signal_diff = float(diff_df["signal_diff"].max()) if not diff_df.empty else 0.0

        self.signal_diff = diff_df
        self.factor_diffs = {
            name: pd.Series(diffs, name=name)
            for name, diffs in factor_diffs.items()
        }
        self.biased_dates = biased_dates

        if has_bias:
            logger.warning(
                "⚠  Lookahead bias detected on %d/%d dates "
                "(threshold=%.0e, max_signal_diff=%.6f)",
                len(biased_dates), len(check_dates),
                self.threshold, max_signal_diff,
            )
        else:
            logger.info(
                "✓  No lookahead bias detected (threshold=%.0e, max_signal_diff=%.6f)",
                self.threshold, max_signal_diff,
            )

        return {
            "has_bias": has_bias,
            "dates_checked": len(check_dates),
            "biased_dates": [str(d) for d in biased_dates],
            "max_signal_diff": max_signal_diff,
            "threshold": self.threshold,
            "signal_diff_summary": diff_df,
            "factor_bias_report": factor_bias_report,
        }

    def _select_dates(self, signal: pd.DataFrame) -> list[pd.Timestamp]:
        """Select a subset of dates to check for bias detection."""
        dates = signal.index.tolist()

        # Apply max_check_dates limit by uniform sampling
        if len(dates) > self.max_check_dates:
            step = max(1, len(dates) // self.max_check_dates)
            dates = dates[::step]

        # Apply sample fraction
        if self.sample_fraction < 1.0:
            n_sample = max(1, int(len(dates) * self.sample_fraction))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(dates), size=n_sample, replace=False)
            dates = [dates[i] for i in sorted(idx)]

        return dates

    def print_report(self, result: dict[str, Any]) -> None:
        """Print a human-readable bias detection report."""
        has_bias = result["has_bias"]
        max_diff = result["max_signal_diff"]

        print()
        print("=" * 70)
        print("  LOOKAHEAD BIAS DETECTION REPORT")
        print("=" * 70)
        print(f"  Dates checked:  {result['dates_checked']}")
        print(f"  Threshold:      {result['threshold']:.0e}")
        print(f"  Max signal diff: {max_diff:.6f}")
        print(f"  Has bias:       {'⚠  YES' if has_bias else '✓  NO'}")

        if has_bias:
            print(f"\n  Biased dates ({len(result['biased_dates'])}):")
            for d in result['biased_dates'][:20]:
                print(f"    {d}")
            if len(result['biased_dates']) > 20:
                print(f"    ... and {len(result['biased_dates']) - 20} more")

        print(f"\n  {'─' * 70}")
        print(f"  Factor Bias Report:")
        print(f"  {'Factor':<25} {'Max Diff':>10} {'Mean Diff':>10} {'% Biased':>10}")
        print(f"  {'─' * 55}")
        for name, stats in sorted(
            result['factor_bias_report'].items(),
            key=lambda x: x[1]['max_diff'],
            reverse=True,
        ):
            if stats['max_diff'] > 0:
                print(
                    f"  {name:<25} {stats['max_diff']:>10.6f} "
                    f"{stats['mean_diff']:>10.6f} {stats['pct_biased']:>9.1%}"
                )
        print("=" * 70)
        print()

    def suggest_fixes(self, result: dict[str, Any]) -> list[str]:
        """Suggest fixes for detected biases."""
        suggestions = []
        if not result["has_bias"]:
            return ["No bias detected — no fixes needed."]

        for name, stats in result["factor_bias_report"].items():
            if stats["max_diff"] > self.threshold:
                suggestions.append(
                    f"Factor '{name}': max diff={stats['max_diff']:.6f}. "
                    "Check if this factor uses global statistics or future data."
                )

        suggestions.append(
            "If ICIR-weighted alpha is used, verify that point-in-time "
            "IC computation is correctly implemented (our code ensures this)."
        )
        return suggestions
