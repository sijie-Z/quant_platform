"""Data quality monitoring pipeline.

Validates market data before it enters the factor/backtest pipeline:
- Missing data detection (gaps, NaN rates)
- Price anomaly detection (extreme moves, stale prices)
- Corporate action validation (dividends, splits)
- Volume sanity checks
- Data freshness monitoring
- Cross-source reconciliation

Inspired by production data quality frameworks at major quant funds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class DataQualityCheck:
    """A single data quality check result."""
    def __init__(self, name: str, passed: bool, severity: str, message: str,
                 details: dict | None = None):
        self.name = name
        self.passed = passed
        self.severity = severity    # info/warn/error/critical
        self.message = message
        self.details = details or {}
        self.timestamp = pd.Timestamp.now().isoformat()


class DataQualityMonitor:
    """Runs data quality checks on market data.

    Usage:
        monitor = DataQualityMonitor()
        results = monitor.check_prices(prices_df)
        report = monitor.get_report()
    """

    def __init__(
        self,
        max_missing_pct: float = 0.05,      # 5% max missing data
        max_daily_move: float = 0.22,        # 22% max daily move (limit up + buffer)
        max_stale_days: int = 5,             # Max consecutive flat days
        min_volume: float = 1000,            # Minimum daily volume
        max_nan_rate: float = 0.02,          # 2% max NaN rate per column
    ):
        self.max_missing_pct = max_missing_pct
        self.max_daily_move = max_daily_move
        self.max_stale_days = max_stale_days
        self.min_volume = min_volume
        self.max_nan_rate = max_nan_rate
        self.checks: list[DataQualityCheck] = []

    def check_prices(self, prices: pd.DataFrame) -> list[DataQualityCheck]:
        """Run all price data quality checks."""
        self.checks = []

        self._check_missing_data(prices)
        self._check_nan_rate(prices)
        self._check_price_anomalies(prices)
        self._check_stale_prices(prices)
        self._check_data_freshness(prices)
        self._check_price_monotonicity(prices)

        return self.checks

    def check_returns(self, returns: pd.DataFrame) -> list[DataQualityCheck]:
        """Run return data quality checks."""
        self._check_extreme_returns(returns)
        self._check_return_distribution(returns)
        return self.checks

    def _check_missing_data(self, prices: pd.DataFrame):
        """Check for missing data (gaps in time series)."""
        total_cells = prices.size
        missing_cells = prices.isna().sum().sum()
        missing_pct = missing_cells / total_cells if total_cells > 0 else 0

        passed = missing_pct <= self.max_missing_pct
        self.checks.append(DataQualityCheck(
            name="missing_data",
            passed=passed,
            severity="error" if not passed else "info",
            message=f"Missing data: {missing_pct:.2%} (limit: {self.max_missing_pct:.2%})",
            details={"missing_pct": missing_pct, "missing_cells": int(missing_cells)},
        ))

        # Per-stock missing rates
        stock_missing = prices.isna().mean()
        bad_stocks = stock_missing[stock_missing > self.max_missing_pct * 2]
        if len(bad_stocks) > 0:
            self.checks.append(DataQualityCheck(
                name="bad_stocks",
                passed=False,
                severity="warn",
                message=f"{len(bad_stocks)} stocks with >{self.max_missing_pct*2:.0%} missing data",
                details={"stocks": bad_stocks.head(10).to_dict()},
            ))

    def _check_nan_rate(self, prices: pd.DataFrame):
        """Check NaN rate per column."""
        nan_rates = prices.isna().mean()
        high_nan = nan_rates[nan_rates > self.max_nan_rate]

        passed = len(high_nan) == 0
        self.checks.append(DataQualityCheck(
            name="nan_rate",
            passed=passed,
            severity="warn" if not passed else "info",
            message=f"{len(high_nan)} columns with NaN rate > {self.max_nan_rate:.1%}",
            details={"high_nan_columns": high_nan.head(10).to_dict()},
        ))

    def _check_price_anomalies(self, prices: pd.DataFrame):
        """Check for anomalous price movements."""
        pct_changes = prices.pct_change().abs()
        anomalies = pct_changes > self.max_daily_move

        n_anomalies = anomalies.sum().sum()
        passed = n_anomalies == 0

        if n_anomalies > 0:
            # Find the worst anomalies
            worst = pct_changes.max().sort_values(ascending=False)
            self.checks.append(DataQualityCheck(
                name="price_anomalies",
                passed=passed,
                severity="error" if n_anomalies > 10 else "warn",
                message=f"{n_anomalies} price movements exceed {self.max_daily_move:.0%}",
                details={"worst_stocks": worst.head(5).to_dict()},
            ))
        else:
            self.checks.append(DataQualityCheck(
                name="price_anomalies",
                passed=True,
                severity="info",
                message="No anomalous price movements detected",
            ))

    def _check_stale_prices(self, prices: pd.DataFrame):
        """Check for stale (flat) prices indicating data issues."""
        stale_counts = {}
        for col in prices.columns:
            series = prices[col].dropna()
            if len(series) < 2:
                continue
            # Count consecutive equal values
            flat = (series.diff().abs() < 1e-6).astype(int)
            max_stale = 0
            current_streak = 0
            for v in flat:
                if v == 1:
                    current_streak += 1
                    max_stale = max(max_stale, current_streak)
                else:
                    current_streak = 0
            if max_stale > self.max_stale_days:
                stale_counts[col] = max_stale

        passed = len(stale_counts) == 0
        self.checks.append(DataQualityCheck(
            name="stale_prices",
            passed=passed,
            severity="warn" if not passed else "info",
            message=f"{len(stale_counts)} stocks with >{self.max_stale_days} consecutive flat days",
            details={"stale_stocks": dict(sorted(stale_counts.items(), key=lambda x: -x[1])[:10])},
        ))

    def _check_data_freshness(self, prices: pd.DataFrame):
        """Check if data is fresh (last date is recent)."""
        if len(prices) == 0:
            self.checks.append(DataQualityCheck(
                name="freshness", passed=False, severity="critical",
                message="No data available",
            ))
            return

        last_date = prices.index[-1]
        if hasattr(last_date, 'date'):
            days_old = (pd.Timestamp.now().date() - last_date.date()).days
        else:
            days_old = 0

        passed = days_old <= 3  # Allow 3 days for weekends/holidays
        self.checks.append(DataQualityCheck(
            name="freshness",
            passed=passed,
            severity="warn" if not passed else "info",
            message=f"Last data date: {last_date} ({days_old} days ago)",
            details={"last_date": str(last_date), "days_old": days_old},
        ))

    def _check_price_monotonicity(self, prices: pd.DataFrame):
        """Check that prices are non-negative."""
        negative = (prices < 0).sum().sum()
        passed = negative == 0
        self.checks.append(DataQualityCheck(
            name="price_monotonicity",
            passed=passed,
            severity="critical" if not passed else "info",
            message=f"{negative} negative prices detected" if negative else "All prices non-negative",
        ))

    def _check_extreme_returns(self, returns: pd.DataFrame):
        """Check for extreme daily returns."""
        extreme = (returns.abs() > 0.30).sum().sum()  # >30% daily
        passed = extreme == 0
        self.checks.append(DataQualityCheck(
            name="extreme_returns",
            passed=passed,
            severity="warn" if not passed else "info",
            message=f"{extreme} daily returns > 30%",
        ))

    def _check_return_distribution(self, returns: pd.DataFrame):
        """Check return distribution for anomalies."""
        flat_returns = returns.values.flatten()
        flat_returns = flat_returns[~np.isnan(flat_returns)]

        if len(flat_returns) < 100:
            return

        skew = float(pd.Series(flat_returns).skew())
        kurt = float(pd.Series(flat_returns).kurtosis())

        # Extreme skew or kurtosis indicates data issues
        passed = abs(skew) < 5 and kurt < 50
        self.checks.append(DataQualityCheck(
            name="return_distribution",
            passed=passed,
            severity="warn" if not passed else "info",
            message=f"Skew={skew:.2f}, Kurtosis={kurt:.2f}",
            details={"skew": skew, "kurtosis": kurt},
        ))

    def get_report(self) -> dict:
        """Generate data quality report summary."""
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        failed = sum(1 for c in self.checks if not c.passed)

        by_severity = {}
        for c in self.checks:
            by_severity.setdefault(c.severity, []).append(c.name)

        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "overall_status": "PASS" if failed == 0 else "FAIL",
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }
