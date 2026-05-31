"""Multi-source validated data provider with confidence scoring.

Inspired by Senior Analyst's multi-source cross-validation pattern.
Instead of trusting a single data source, queries 2+ providers and
compares results. Returns fused data with confidence scores.

Confidence scoring:
  ≥0.90  — Multiple sources agree (deviation < threshold)
  0.70-0.89 — Single source only, or small deviations
  0.50-0.69 — Significant deviations, needs human review
  <0.50  — Conflicting data, not usable
"""

from __future__ import annotations

import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.data.providers.base import DataProvider

logger = logging.getLogger(__name__)

# Confidence thresholds
HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.75
LOW_CONFIDENCE = 0.55
UNUSABLE = 0.0

# Max relative deviation before flagging (%)
PRICE_DEVIATION_THRESHOLD = 2.0   # 2% for prices
FINANCIAL_DEVIATION_THRESHOLD = 10.0  # 10% for financial metrics


class DataDiscrepancy(Exception):
    """Raised when providers return conflicting data."""


@dataclass
class ValidatedResult:
    """Result from cross-validation."""
    values: pd.DataFrame | pd.Series
    confidence: float
    n_sources: int
    discrepancies: list[str] | None = None


class ValidatedProvider(DataProvider):
    """Data provider that cross-validates across multiple sources.

    Wraps multiple DataProvider implementations, queries them in parallel,
    and fuses results with confidence scoring.

    Args:
        providers: Dict of {name: DataProvider} to validate across.
        primary: Name of the primary provider (for fallback).
        price_deviation: Max price deviation % before flagging.
        fin_deviation: Max financial deviation % before flagging.
    """

    def __init__(
        self,
        providers: dict[str, DataProvider],
        primary: str | None = None,
        price_deviation: float = PRICE_DEVIATION_THRESHOLD,
        fin_deviation: float = FINANCIAL_DEVIATION_THRESHOLD,
    ):
        self.providers = providers
        self.primary = primary or next(iter(providers.keys()))
        self.price_deviation = price_deviation
        self.fin_deviation = fin_deviation

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------

    def get_prices(
        self,
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        return self._validate(
            "get_prices",
            {"start_date": start_date, "end_date": end_date, "fields": fields},
            self.price_deviation,
        ).values

    def get_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return self._validate(
            "get_financials",
            {"start_date": start_date, "end_date": end_date},
            self.fin_deviation,
        ).values

    def get_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        return self._validate(
            "get_benchmark",
            {"start_date": start_date, "end_date": end_date},
            self.price_deviation,
        ).values

    def get_metadata(self) -> pd.DataFrame:
        return self.providers[self.primary].get_metadata()

    # ------------------------------------------------------------------
    # Cross-validation logic
    # ------------------------------------------------------------------

    def _validate(
        self,
        method: str,
        kwargs: dict[str, Any],
        deviation_threshold: float,
    ) -> ValidatedResult:
        """Query multiple providers and cross-validate results."""
        results: dict[str, pd.DataFrame | pd.Series] = {}
        errors: dict[str, str] = {}

        for name, provider in self.providers.items():
            try:
                fn = getattr(provider, method)
                result = fn(**kwargs)
                if result is not None and not (
                    isinstance(result, pd.DataFrame) and result.empty
                ):
                    results[name] = result
                else:
                    errors[name] = "Empty result"
            except Exception as e:
                errors[name] = str(e)
                logger.debug("Provider %s failed for %s: %s", name, method, e)

        if not results:
            logger.error(
                "All providers failed for %s: %s", method, errors,
            )
            return ValidatedResult(
                values=pd.DataFrame(),
                confidence=UNUSABLE,
                n_sources=0,
                discrepancies=[f"All sources failed: {errors}"],
            )

        if len(results) == 1:
            name = next(iter(results.keys()))
            logger.info(
                "Single source %s for %s — confidence %.2f",
                name, method, MEDIUM_CONFIDENCE,
            )
            return ValidatedResult(
                values=results[name],
                confidence=MEDIUM_CONFIDENCE,
                n_sources=1,
            )

        # Multiple sources — cross-validate
        return self._compare(results, deviation_threshold, method)

    def _compare(
        self,
        results: dict[str, pd.DataFrame | pd.Series],
        threshold: float,
        method: str,
    ) -> ValidatedResult:
        """Compare results from multiple providers.

        Calculates pair-wise deviations. If all within threshold,
        fuses them (mean). Otherwise returns primary with warning.
        """
        items = list(results.items())
        discrepancies = []
        max_deviation = 0.0

        # Compare each pair
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                name1, df1 = items[i]
                name2, df2 = items[j]
                dev = self._compute_deviation(df1, df2, name1, name2)
                max_deviation = max(max_deviation, dev)
                if dev > threshold:
                    discrepancies.append(
                        f"{name1} vs {name2}: {dev:.1f}% deviation "
                        f"(threshold: {threshold:.0f}%)"
                    )
                    logger.warning(
                        "Data discrepancy: %s vs %s = %.1f%% for %s",
                        name1, name2, dev, method,
                    )

        if discrepancies:
            logger.warning(
                "%d discrepancies found in %s", len(discrepancies), method,
            )
            # Return primary source with low confidence, plus discrepancies
            primary_result = results[self.primary]
            return ValidatedResult(
                values=primary_result,
                confidence=max(MEDIUM_CONFIDENCE - 0.25 * len(discrepancies), LOW_CONFIDENCE),
                n_sources=len(results),
                discrepancies=discrepancies,
            )

        # All sources agree — fuse by averaging
        fused = self._fuse(results)
        confidence = HIGH_CONFIDENCE - 0.05 * (len(results) - 2)
        confidence = min(confidence, 0.99)
        logger.info(
            "Multi-source validated %s: %d sources, conf=%.2f, max_dev=%.1f%%",
            method, len(results), confidence, max_deviation,
        )
        return ValidatedResult(
            values=fused,
            confidence=confidence,
            n_sources=len(results),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_deviation(
        self,
        df1: pd.DataFrame | pd.Series,
        df2: pd.DataFrame | pd.Series,
        name1: str,
        name2: str,
    ) -> float:
        """Compute max relative deviation between two DataFrames."""
        try:
            # Align on common index/columns
            if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
                common_idx = df1.index.intersection(df2.index)
                common_cols = df1.columns.intersection(df2.columns)
                if len(common_idx) == 0 or len(common_cols) == 0:
                    return 100.0
                a = df1.loc[common_idx, common_cols].values
                b = df2.loc[common_idx, common_cols].values
            elif isinstance(df1, pd.Series) and isinstance(df2, pd.Series):
                common_idx = df1.index.intersection(df2.index)
                if len(common_idx) == 0:
                    return 100.0
                a = df1.loc[common_idx].values
                b = df2.loc[common_idx].values
            else:
                return 100.0

            # Ignore NaN
            mask = np.isfinite(a) & np.isfinite(b) & (a != 0)
            if not mask.any():
                return 0.0
            deviations = np.abs((a[mask] - b[mask]) / a[mask]) * 100
            return float(np.max(deviations))
        except Exception:
            return 100.0

    def _fuse(
        self,
        results: dict[str, pd.DataFrame | pd.Series],
    ) -> pd.DataFrame | pd.Series:
        """Fuse multiple results by mean."""
        # Simple average for now
        first = list(results.values())[0]
        if isinstance(first, pd.DataFrame):
            dfs = [df for df in results.values() if isinstance(df, pd.DataFrame)]
            return sum(dfs) / len(dfs) if len(dfs) > 0 else first
        else:
            series_list = [s for s in results.values() if isinstance(s, pd.Series)]
            return sum(series_list) / len(series_list) if len(series_list) > 0 else first
