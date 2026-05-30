"""Factor Screener: boolean condition screening on factor values.

Inspired by BlackOil-OmniAlpha's Strategy + AnalysisEngine pattern, this
module adds a second operating mode to the platform:

  Ranking mode (original):  Factors → Alpha → Optimizer → Portfolio weights
  Screener mode (new):      Factors → Bool rules → Equal-weight qualifiers

The screener lets users define rules like "pe_ratio < 30 AND roe > 15" and
quickly find which stocks pass all conditions. It shares the same factor
engine as the ranking pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

OPERATORS = {"gt", "gte", "lt", "lte", "eq", "ne", "between"}
LOGIC_MODES = {"and", "or"}


@dataclass
class ScreenRule:
    """A single screening condition.

    Args:
        factor: Factor name, must match a key in the processed_factors dict.
        operator: Comparison operator.
            gt / gte / lt / lte / eq / ne / between
        value: Threshold value. For 'between', this is [low, high].
        tolerance: Floating-point tolerance for comparisons (default 1e-8).
    """

    factor: str
    operator: str
    value: float | list[float]
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.operator not in OPERATORS:
            raise ValueError(
                f"Unknown operator '{self.operator}'. "
                f"Must be one of {sorted(OPERATORS)}"
            )
        if self.operator == "between":
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                raise ValueError(
                    "'between' operator requires value=[low, high]"
                )

    def apply(self, series: pd.Series) -> pd.Series:
        """Apply this rule to a single cross-section of factor values.

        Args:
            series: (asset) factor values for one date.

        Returns:
            Boolean Series: True where the condition holds.
        """
        tol = self.tolerance

        if self.operator == "gt":
            return series > self.value  # type: ignore[operator]
        elif self.operator == "gte":
            return series >= self.value  # type: ignore[operator]
        elif self.operator == "lt":
            return series < self.value  # type: ignore[operator]
        elif self.operator == "lte":
            return series <= self.value  # type: ignore[operator]
        elif self.operator == "eq":
            return (series - self.value).abs() <= tol  # type: ignore[operator]
        elif self.operator == "ne":
            return (series - self.value).abs() > tol  # type: ignore[operator]
        elif self.operator == "between":
            lo, hi = self.value  # type: ignore[misc]
            return (series >= lo) & (series <= hi)
        return pd.Series(False, index=series.index)


@dataclass
class ScreenConfig:
    """Configuration for the Factor Screener.

    Args:
        enabled: Whether to enable screener mode (vs ranking).
        rules: List of screening conditions.
        logic: How to combine rules — 'and' (all must pass) or 'or' (any).
        min_stocks: Minimum number of stocks to return. If screening yields
            fewer, rules are progressively relaxed (removes the most
            restrictive rule) until min_stocks is met.
        max_stocks: Maximum number of stocks to return. If screening yields
            more, ties broken by multi-factor score (equal-weighted zscore).
    """

    enabled: bool = False
    rules: list[ScreenRule] = field(default_factory=list)
    logic: str = "and"
    min_stocks: int = 5
    max_stocks: int = 200

    def __post_init__(self) -> None:
        if self.logic not in LOGIC_MODES:
            raise ValueError(
                f"Unknown logic '{self.logic}'. Must be one of {sorted(LOGIC_MODES)}"
            )


# ---------------------------------------------------------------------------
# FactorScreener
# ---------------------------------------------------------------------------


class FactorScreener:
    """Screen stocks using boolean conditions on factor values.

    Usage:
        screener = FactorScreener(config)
        qualifiers = screener.screen(processed_factors)
        # qualifiers is a list of asset codes

    The screener works on the LATEST cross-section of processed factor
    values. For backtesting, it can be run period-by-period.
    """

    def __init__(self, config: ScreenConfig | dict[str, Any] | None = None):
        if config is None:
            self.config = ScreenConfig()
        elif isinstance(config, dict):
            self.config = self._config_from_dict(config)
        else:
            self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def screen(
        self,
        processed_factors: dict[str, pd.DataFrame],
        rules: list[ScreenRule] | None = None,
        logic: str | None = None,
        date: str | None = None,
    ) -> list[str]:
        """Screen the latest cross-section of factor values.

        Args:
            processed_factors: {factor_name: DataFrame(date × asset)}.
            rules: Override rules. If None, uses self.config.rules.
            logic: Override logic. If None, uses self.config.logic.
            date: Specific date to screen. If None, uses the latest date
                common across all factors.

        Returns:
            Sorted list of qualifying asset codes.
        """
        rules = rules if rules is not None else self.config.rules
        logic = logic if logic is not None else self.config.logic

        if not rules:
            logger.warning("No rules defined for screener — returning empty")
            return []

        # Validate factor names exist
        for rule in rules:
            if rule.factor not in processed_factors:
                logger.warning(
                    "Factor '%s' not found in processed_factors. "
                    "Available: %s",
                    rule.factor,
                    sorted(processed_factors.keys()),
                )
                return []

        # Determine target date
        target_date = self._resolve_date(processed_factors, rules, date)
        if target_date is None:
            logger.warning("No data available for screening")
            return []

        # Extract the cross-section: asset → factor values
        cross = self._build_cross_section(processed_factors, rules, target_date)

        if cross.empty:
            logger.warning("Cross-section is empty — no stocks to screen")
            return []

        # Apply rules
        mask = self._apply_rules(cross, rules, logic)

        qualifying = cross.index[mask].tolist()
        logger.info(
            "Screener [%s %s] on %s: %d / %d stocks passed",
            logic,
            ", ".join(r.factor for r in rules),
            target_date,
            len(qualifying),
            len(cross),
        )

        # Enforce min / max bounds
        qualifying = self._enforce_bounds(
            qualifying, cross, rules, self.config.min_stocks,
            self.config.max_stocks,
        )

        return sorted(qualifying)

    def screen_historical(
        self,
        processed_factors: dict[str, pd.DataFrame],
        rules: list[ScreenRule] | None = None,
        logic: str | None = None,
    ) -> pd.Series:
        """Screen every date in the factor history.

        Returns a Series with date index, each value being the sorted list
        of qualifying assets on that date.
        """
        rules = rules if rules is not None else self.config.rules
        logic = logic if logic is not None else self.config.logic

        if not rules:
            return pd.Series(dtype=object)

        # Find common date index across all needed factors
        dates = self._common_dates(processed_factors, rules)
        if dates.empty:
            return pd.Series(dtype=object)

        results: dict[str, list[str]] = {}
        for d in dates:
            cross = self._build_cross_section(processed_factors, rules, d)
            if cross.empty:
                continue
            mask = self._apply_rules(cross, rules, logic)
            results[str(d)] = cross.index[mask].tolist()

        return pd.Series(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_date(
        self,
        processed_factors: dict[str, pd.DataFrame],
        rules: list[ScreenRule],
        date: str | None,
    ) -> str | None:
        """Find the target date for screening."""
        if date is not None:
            return date
        # Latest common date
        dates = self._common_dates(processed_factors, rules)
        return str(dates[-1]) if not dates.empty else None

    def _common_dates(
        self,
        processed_factors: dict[str, pd.DataFrame],
        rules: list[ScreenRule],
    ) -> pd.DatetimeIndex:
        """Find the intersection of all date indices for relevant factors."""
        indices = []
        for rule in rules:
            df = processed_factors.get(rule.factor)
            if df is not None and not df.empty:
                indices.append(df.index)
        if not indices:
            return pd.DatetimeIndex([])
        common = indices[0]
        for idx in indices[1:]:
            common = common.intersection(idx)
        return common.sort_values()  # type: ignore[return-value]

    def _build_cross_section(
        self,
        processed_factors: dict[str, pd.DataFrame],
        rules: list[ScreenRule],
        date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Build a (asset × factor) DataFrame for one date.

        Only includes columns for factors referenced by rules.
        Only includes assets that have non-NaN values for ALL rule factors.
        """
        date_key = pd.Timestamp(date)
        series_list = []

        for rule in rules:
            df = processed_factors.get(rule.factor)
            if df is None or date_key not in df.index:
                logger.debug("Date %s not found in factor %s", date, rule.factor)
                return pd.DataFrame()
            ser = df.loc[date_key].copy()
            ser.name = rule.factor
            series_list.append(ser)

        cross = pd.concat(series_list, axis=1)

        # Drop rows where ANY rule factor is NaN
        before = len(cross)
        cross = cross.dropna()
        if len(cross) < before:
            logger.debug(
                "Dropped %d assets with NaN factor values", before - len(cross)
            )

        return cross

    def _apply_rules(
        self,
        cross: pd.DataFrame,
        rules: list[ScreenRule],
        logic: str,
    ) -> pd.Series:
        """Apply all rules and combine with AND/OR logic."""
        if not rules:
            return pd.Series(False, index=cross.index)

        masks = []
        for rule in rules:
            if rule.factor not in cross.columns:
                masks.append(pd.Series(False, index=cross.index))
            else:
                masks.append(rule.apply(cross[rule.factor]))

        if logic == "and":
            combined = pd.concat(masks, axis=1).all(axis=1)
        else:
            combined = pd.concat(masks, axis=1).any(axis=1)

        return combined

    def _enforce_bounds(
        self,
        qualifying: list[str],
        cross: pd.DataFrame,
        rules: list[ScreenRule],
        min_stocks: int,
        max_stocks: int,
    ) -> list[str]:
        """Ensure the result meets min/max stock bounds.

        If fewer than min_stocks qualify, relax rules progressively
        (remove the most restrictive rule) until we hit the minimum.

        If more than max_stocks qualify, rank by equal-weighted zscore
        of all screened factors and take the top max_stocks.
        """
        if len(qualifying) >= min_stocks and len(qualifying) <= max_stocks:
            return qualifying

        # Relax: if too few, drop the rule with the fewest passes
        if len(qualifying) < min_stocks and len(rules) > 1:
            logger.info(
                "Only %d stocks passed — relaxing rules to meet min=%d",
                len(qualifying),
                min_stocks,
            )
            # Find the most restrictive rule (fewest passes)
            rule_pass_counts = []
            for rule in rules:
                if rule.factor in cross.columns:
                    mask = rule.apply(cross[rule.factor])
                    rule_pass_counts.append((rule, mask.sum()))
                else:
                    rule_pass_counts.append((rule, 0))

            # Drop the strictest rule and retry
            strictest = min(rule_pass_counts, key=lambda x: x[1])
            remaining_rules = [r for r in rules if r != strictest[0]]
            if remaining_rules:
                new_mask = self._apply_rules(cross, remaining_rules, "and")
                relaxed = cross.index[new_mask].tolist()
                return self._enforce_bounds(
                    relaxed, cross, remaining_rules,
                    min_stocks, max_stocks,
                )

        # Cap: if too many, score and take top
        if len(qualifying) > max_stocks:
            logger.info(
                "%d stocks passed — capping to top %d by multi-factor score",
                len(qualifying),
                max_stocks,
            )
            # Equal-weighted zscore of all rule factors
            factor_cols = [r.factor for r in rules if r.factor in cross.columns]
            if factor_cols:
                scores = cross[factor_cols].sub(cross[factor_cols].mean())
                scores = scores.div(scores.std().replace(0, 1))
                cross["_score"] = scores.mean(axis=1)
                top = cross.nlargest(max_stocks, "_score")
                return top.index.tolist()

        return qualifying

    @staticmethod
    def _config_from_dict(d: dict[str, Any]) -> ScreenConfig:
        """Build ScreenConfig from a parsed YAML dict."""
        rules_raw = d.get("rules", [])
        rules = []
        for r in rules_raw:
            rules.append(ScreenRule(
                factor=r.get("factor", ""),
                operator=r.get("operator", "gt"),
                value=r.get("value", 0),
            ))
        return ScreenConfig(
            enabled=d.get("enabled", False),
            rules=rules,
            logic=d.get("logic", "and"),
            min_stocks=d.get("min_stocks", 5),
            max_stocks=d.get("max_stocks", 200),
        )
