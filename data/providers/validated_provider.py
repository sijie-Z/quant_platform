"""Multi-source validated data provider with confidence scoring.

Inspired by Senior Analyst's multi-source cross-validation pattern.
Instead of trusting a single data source, queries 2+ providers and
compares results. Returns fused data with confidence scores.

Price basis contract
--------------------
Price panels are only fused when every contributing source declares the
same close basis.  Mixing a raw (CNY) panel with an adjusted one would
average incomparable levels, so :class:`DataDiscrepancyError` is raised
instead of silently coercing either side.  The fused frame keeps the
reference source's ``adj_factor`` column (a divisor is only meaningful
against its own provider's reference date) and the reference source's
basis ``attrs`` (``price_basis`` / ``adjusted_price_basis`` /
``adjustment_factor_available``), so ``DataPipeline.get_raw_close()``
continues to work on validated runs.  The resolved declaration is also
forwarded onto the wrapper itself (``price_adjustment`` etc.).

Confidence scoring:
  ≥0.90  — Multiple sources agree (deviation < threshold)
  0.70-0.89 — Single source only, or small deviations
  0.50-0.69 — Significant deviations, needs human review
  <0.50  — Conflicting data, not usable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.data.providers.base import DataProvider
from quant_platform.data.schema import PRICE_COLUMNS, DataQualityError

logger = logging.getLogger(__name__)

# Confidence thresholds
HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.75
LOW_CONFIDENCE = 0.55
UNUSABLE = 0.0

# Max relative deviation before flagging (%)
PRICE_DEVIATION_THRESHOLD = 2.0   # 2% for prices
FINANCIAL_DEVIATION_THRESHOLD = 10.0  # 10% for financial metrics

# Provider declarations that mean "raw close" in the shared divisor contract.
_RAW_DECLARATIONS = frozenset({"raw", "synthetic_adj_factor", "none"})
_PRICE_BASES = frozenset({"raw", "qfq", "hfq"})


class DataDiscrepancyError(DataQualityError):
    """Raised when providers return conflicting data (e.g. price bases)."""


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

        # Forward the primary source's price-basis declaration.  The primary
        # defines the panel's semantics; DataPipeline reads these attributes
        # through the wrapper when a frame carries no attrs of its own.
        primary_provider = self.providers[self.primary]
        self.price_adjustment = getattr(
            primary_provider, "price_adjustment", "provider-defined",
        )
        self.adjusted_price_basis = getattr(
            primary_provider, "adjusted_price_basis", None,
        )
        self.adjustment_factor_available = getattr(
            primary_provider, "adjustment_factor_available", None,
        )

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------

    def get_prices(
        self,
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Validated price panel with the reference source's basis contract.

        The returned frame carries ``adj_factor`` (when the reference source
        supplies it) and the ``price_basis`` / ``adjusted_price_basis`` /
        ``adjustment_factor_available`` attrs, so it can be handed to
        :class:`~quant_platform.data.pipeline.DataPipeline` unchanged.
        """
        return self.get_validated_prices(start_date, end_date, fields).values

    def get_validated_prices(
        self,
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
    ) -> ValidatedResult:
        """Get prices with full validation result (incl. confidence score)."""
        return self._validate(
            "get_prices",
            {"start_date": start_date, "end_date": end_date, "fields": fields},
            self.price_deviation,
        )

    def get_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return self.get_validated_financials(start_date, end_date).values

    def get_validated_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> ValidatedResult:
        """Get financials with full validation result."""
        return self._validate(
            "get_financials",
            {"start_date": start_date, "end_date": end_date},
            self.fin_deviation,
        )

    def get_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        return self.get_validated_benchmark(start_date, end_date).values

    def get_validated_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> ValidatedResult:
        """Get benchmark with full validation result."""
        return self._validate(
            "get_benchmark",
            {"start_date": start_date, "end_date": end_date},
            self.price_deviation,
        )

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
            except DataQualityError:
                # A source that violates the data contract (wrong index,
                # missing columns, double-adjusted prices) returns corrupt
                # data, not merely unavailable data. Masking it as a failed
                # source would silently serve a different panel under the
                # validated label.
                raise
            except Exception as e:
                errors[name] = str(e)
                logger.debug("Provider %s failed for %s: %s", name, method, e)

        if not results:
            logger.error(
                "All providers failed for %s: %s", method, errors,
            )
            if method == "get_prices":
                values: pd.DataFrame | pd.Series = self._empty_price_panel()
            elif method == "get_benchmark":
                values = pd.Series(dtype=float, name="benchmark")
            else:
                values = pd.DataFrame()
            return ValidatedResult(
                values=values,
                confidence=UNUSABLE,
                n_sources=0,
                discrepancies=[f"All sources failed: {errors}"],
            )

        if len(results) == 1:
            name = next(iter(results.keys()))
            value = results[name]
            if method == "get_prices" and isinstance(value, pd.DataFrame):
                # No fusion happened; still stamp the source's explicit basis
                # so a validated run resolves the same contract as a direct
                # run of that provider.
                value = self._tag_price_frame(value, name)
            logger.info(
                "Single source %s for %s — confidence %.2f",
                name, method, MEDIUM_CONFIDENCE,
            )
            return ValidatedResult(
                values=value,
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
        Price panels are only fused when the sources declare the same
        close basis; a mismatch raises before any averaging.
        """
        if method == "get_prices":
            self._check_price_basis(results)

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
            # Return primary source with low confidence, plus discrepancies.
            # The primary may itself have failed (only secondaries agreed to
            # disagree); fall back to the first successful source.
            if self.primary in results:
                primary_result = results[self.primary]
            else:
                primary_result = next(iter(results.values()))
            if method == "get_prices" and isinstance(primary_result, pd.DataFrame):
                primary_result = self._tag_price_frame(
                    primary_result,
                    self.primary if self.primary in results
                    else next(iter(results)),
                )
            return ValidatedResult(
                values=primary_result,
                confidence=max(MEDIUM_CONFIDENCE - 0.25 * len(discrepancies), LOW_CONFIDENCE),
                n_sources=len(results),
                discrepancies=discrepancies,
            )

        # All sources agree — fuse by averaging
        fused = self._fuse(results, method)
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

    # ------------------------------------------------------------------
    # Price-basis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_price_basis(
        provider: DataProvider,
        frame: pd.DataFrame,
    ) -> tuple[str, str, bool]:
        """Resolve ``(price_basis, adjusted_price_basis, factor_available)``.

        Frame ``attrs`` are authoritative (same precedence as
        ``DataPipeline._resolve_price_basis``), then the provider's
        ``price_adjustment`` declaration.  Without either, a non-constant
        ``adj_factor`` column means the shared raw-close + divisor contract;
        everything else stays ``unknown`` rather than being guessed.
        """
        attrs = getattr(frame, "attrs", {}) or {}
        declared = attrs.get("price_basis")
        if declared is None:
            declared = getattr(provider, "price_adjustment", None)
        text = str(declared or "").strip().lower()
        if text in _RAW_DECLARATIONS:
            basis = "raw"
        elif text in {"qfq", "hfq"}:
            basis = text
        else:
            basis = "unknown"

        factor_varies = False
        if "adj_factor" in getattr(frame, "columns", ()) and len(frame):
            factor = pd.to_numeric(frame["adj_factor"], errors="coerce")
            values = factor.to_numpy(dtype=float)
            finite = np.isfinite(values)
            factor_varies = bool(
                finite.any()
                and (np.abs(values[finite] - 1.0) > 1e-12).any()
            )
        if basis == "unknown" and factor_varies:
            basis = "raw"

        available = attrs.get("adjustment_factor_available")
        if available is None:
            available = getattr(
                provider, "adjustment_factor_available", None,
            )
        if available is None:
            available = bool(factor_varies)

        adjusted = attrs.get("adjusted_price_basis")
        if adjusted is None:
            adjusted = getattr(provider, "adjusted_price_basis", None)
        adjusted = str(adjusted or "").strip().lower()
        if adjusted not in _PRICE_BASES:
            if basis in {"qfq", "hfq"}:
                adjusted = basis
            elif basis == "raw":
                # The shared divisor contract produces a back-adjusted (hfq)
                # series; a provider with no corporate-action factor keeps
                # the raw series as its research path.
                adjusted = "hfq" if available else "raw"
            else:
                adjusted = "unknown"

        return basis, adjusted, bool(available)

    def _check_price_basis(
        self, results: dict[str, pd.DataFrame | pd.Series],
    ) -> None:
        """Refuse to fuse price panels whose close bases disagree.

        Averaging a raw level with an adjusted one silently corrupts both
        the panel and any downstream return series, so a mismatch (or a
        known basis mixed with an undeclared source) is a hard error rather
        than an implicit normalization.  Fully undeclared legacy frames are
        left to the existing deviation checks (no basis evidence exists to
        compare).
        """
        bases = {
            name: self._frame_price_basis(self.providers[name], frame)[0]
            for name, frame in results.items()
            if isinstance(frame, pd.DataFrame)
        }
        known = {name: basis for name, basis in bases.items() if basis != "unknown"}
        unknown = [name for name, basis in bases.items() if basis == "unknown"]
        if len(set(known.values())) > 1 or (known and unknown):
            detail = ", ".join(f"{name}={basis}" for name, basis in bases.items())
            raise DataDiscrepancyError(
                "Refusing to fuse price panels with incompatible close bases "
                f"({detail}). Raw and adjusted levels (and undeclared "
                "sources) cannot be averaged. Configure sources with the "
                "same adjustment basis, or use a single primary source for "
                "prices."
            )

    def _empty_price_panel(self) -> pd.DataFrame:
        """Contract-shaped empty price panel (MultiIndex + declared basis).

        Used when every source failed: callers still get a frame the
        pipeline can validate (and a confidence of ``UNUSABLE`` with the
        per-source errors), instead of an opaque ``DataContractError`` about
        a RangeIndex that hides the real cause.
        """
        index = pd.MultiIndex.from_arrays([[], []], names=["date", "asset"])
        empty = pd.DataFrame(index=index, columns=list(PRICE_COLUMNS), dtype=float)
        empty.attrs.update({
            "price_basis": self.price_adjustment,
            "adjusted_price_basis": self.adjusted_price_basis,
            "adjustment_factor_available": bool(
                self.adjustment_factor_available
            ),
        })
        return empty

    def _tag_price_frame(
        self, frame: pd.DataFrame, name: str,
    ) -> pd.DataFrame:
        """Stamp the resolved basis contract onto a price frame."""
        basis, adjusted, available = self._frame_price_basis(
            self.providers[name], frame,
        )
        frame.attrs.update({
            "price_basis": basis,
            "adjusted_price_basis": adjusted,
            "adjustment_factor_available": bool(available),
        })
        return frame

    def _fuse_price_frames(
        self,
        results: dict[str, pd.DataFrame | pd.Series],
        dfs: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Fuse raw price panels without losing the adjustment contract.

        Values (except ``adj_factor``) are averaged across the sources that
        provide each column.  The divisor is *not* averaged: its scale
        depends on the provider's reference date, so the reference source's
        factor is carried over unchanged.  The reference (primary when
        available) also defines the column set and the basis attrs.
        """
        reference_name = (
            self.primary if self.primary in results else next(iter(results))
        )
        reference = results[reference_name]
        if not isinstance(reference, pd.DataFrame):
            reference_name = next(
                name for name, value in results.items()
                if isinstance(value, pd.DataFrame)
            )
            reference = results[reference_name]
        basis, adjusted, available = self._frame_price_basis(
            self.providers[reference_name], reference,
        )
        for name, frame in results.items():
            if name == reference_name or not isinstance(frame, pd.DataFrame):
                continue
            _, other_adjusted, _ = self._frame_price_basis(
                self.providers[name], frame,
            )
            if other_adjusted != adjusted:
                # Both closes are raw, but their adjustment paths differ
                # (e.g. qfq vs hfq divisors).  The fused panel keeps the
                # reference's factor/basis; say so instead of coercing
                # silently.
                logger.warning(
                    "Fusing raw sources with different adjusted bases "
                    "(%s=%s, reference %s=%s); keeping the reference "
                    "divisor and adjusted basis",
                    name, other_adjusted, reference_name, adjusted,
                )

        if list(reference.index.names) == ["date", "asset"]:
            fused = pd.concat(dfs).groupby(
                level=["date", "asset"],
            ).mean()
            # The reference source defines the panel schema; columns only a
            # validator carries are not part of the fused panel.
            fused = fused[
                [c for c in reference.columns if c in fused.columns]
            ]
        else:
            # Legacy wide frame (no (date, asset) MultiIndex contract yet):
            # keep the historical arithmetic-mean behavior.
            fused = sum(dfs) / len(dfs)

        # Restrict to the reference's row set.  Averaging the union would
        # emit rows only a validator covers, whose `adj_factor` cannot be
        # taken from the reference (NaN -> the pipeline's repair fails
        # closed) and whose close levels would carry the reference's basis
        # attrs under mixed provenance.
        fused = fused.loc[fused.index.isin(reference.index)]

        if (
            "adj_factor" in reference.columns
            and "adj_factor" in fused.columns
        ):
            fused["adj_factor"] = reference["adj_factor"].reindex(fused.index)

        fused.attrs.update({
            "price_basis": basis,
            "adjusted_price_basis": adjusted,
            "adjustment_factor_available": bool(available),
        })
        return fused

    def _fuse(
        self,
        results: dict[str, pd.DataFrame | pd.Series],
        method: str = "",
    ) -> pd.DataFrame | pd.Series:
        """Fuse multiple results by mean (price panels keep their basis)."""
        first = list(results.values())[0]
        if isinstance(first, pd.DataFrame):
            dfs = [df for df in results.values() if isinstance(df, pd.DataFrame)]
            if not dfs:
                return first
            if method == "get_prices":
                return self._fuse_price_frames(results, dfs)
            return sum(dfs) / len(dfs)
        else:
            series_list = [s for s in results.values() if isinstance(s, pd.Series)]
            return sum(series_list) / len(series_list) if len(series_list) > 0 else first
