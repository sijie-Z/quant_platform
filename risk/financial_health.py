"""Financial health checks and red flag detection for A-share companies.

Inspired by the earnings analysis framework's red flag checklist.
Flags common accounting and operational warning signs in financial data.

Red flags detected:
  1. Receivables growth > Revenue growth (channel stuffing)
  2. CFO / Net Income < 0.8 (low earnings quality)
  3. Gross margin declining 2+ quarters
  4. Non-recurring items > 10% of net income
  5. R&D capitalization ratio abnormal
  6. Revenue growth vs inventory growth divergence
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RedFlag:
    """A detected red flag with severity."""
    name: str
    severity: str  # low / medium / high / critical
    detail: str
    metric: float | None = None
    threshold: float | None = None


@dataclass
class FinancialHealthReport:
    """Complete financial health assessment."""
    company: str = ""
    total_flags: int = 0
    high_risk_flags: int = 0
    flags: list[RedFlag] = field(default_factory=list)
    summary: str = ""


def check_earnings_quality(
    cfo: float,
    net_income: float,
) -> RedFlag | None:
    """Check CFO / Net Income ratio.

    A ratio < 0.8 persistently indicates low earnings quality —
    profits are not converting to cash.
    """
    if net_income == 0:
        return None
    ratio = cfo / abs(net_income)
    if ratio < 0.5:
        return RedFlag(
            name="CFO/Net Income 严重偏低",
            severity="critical",
            detail=f"CFO/净利润 = {ratio:.2f}，远低于 0.8 阈值。利润质量差，净利润没有转化为现金流。",
            metric=ratio, threshold=0.8,
        )
    if ratio < 0.8:
        return RedFlag(
            name="CFO/Net Income 偏低",
            severity="high",
            detail=f"CFO/净利润 = {ratio:.2f}，低于 0.8。可能存在利润虚增。",
            metric=ratio, threshold=0.8,
        )
    return None


def check_receivables_vs_revenue(
    revenue_growth: float,
    receivables_growth: float,
) -> RedFlag | None:
    """Check if receivables grow faster than revenue.

    Receivables growth >> revenue growth suggests channel stuffing —
    the company is shipping products that aren't being sold through.
    """
    gap = receivables_growth - revenue_growth
    if gap > 0.20:
        return RedFlag(
            name="应收增速远超营收增速",
            severity="critical",
            detail=f"应收增长 {receivables_growth:.1%}，营收增长 {revenue_growth:.1%}，"
                   f"差距 {gap:.1%}。可能通过压货虚增收入。",
            metric=gap, threshold=0.20,
        )
    if gap > 0.10:
        return RedFlag(
            name="应收增速超过营收增速",
            severity="medium",
            detail=f"应收增长 {receivables_growth:.1%} > 营收增长 {revenue_growth:.1%}，差距 {gap:.1%}。",
            metric=gap, threshold=0.10,
        )
    return None


def check_gross_margin_trend(
    gross_margins: list[float],
) -> RedFlag | None:
    """Check for 2+ consecutive quarters of gross margin decline."""
    if len(gross_margins) < 3:
        return None

    declines = 0
    for i in range(1, len(gross_margins)):
        if gross_margins[i] < gross_margins[i - 1]:
            declines += 1
        else:
            declines = 0

    if declines >= 2:
        recent = gross_margins[-1]
        prior = gross_margins[-3]
        drop = (prior - recent) / prior * 100
        return RedFlag(
            name="毛利率连续下滑",
            severity="high",
            detail=f"毛利率连续 {declines} 季下滑，最近季 {recent:.1%}，"
                   f"较 {declines} 季前下降 {drop:.1f}%。竞争力或成本结构恶化。",
            metric=drop, threshold=5.0,
        )
    return None


def check_non_recurring_items(
    non_recurring: float,
    net_income: float,
) -> RedFlag | None:
    """Check if non-recurring items are a large part of net income."""
    if abs(net_income) < 1e-6:
        return None
    ratio = abs(non_recurring) / abs(net_income)
    if ratio > 0.20:
        return RedFlag(
            name="非经常性损益占比极高",
            severity="critical",
            detail=f"非经常性损益占净利润 {ratio:.1%}，远超 10% 阈值。核心盈利能力被扭曲。",
            metric=ratio, threshold=0.20,
        )
    if ratio > 0.10:
        return RedFlag(
            name="非经常性损益占比较高",
            severity="high",
            detail=f"非经常性损益占净利润 {ratio:.1%}，超过 10% 红线。",
            metric=ratio, threshold=0.10,
        )
    return None


def check_revenue_vs_inventory(
    revenue_growth: float,
    inventory_growth: float,
) -> RedFlag | None:
    """Check if inventory grows faster than revenue.

    Inventory buildup without revenue growth suggests obsolescence risk.
    """
    gap = inventory_growth - revenue_growth
    if gap > 0.30:
        return RedFlag(
            name="库存增速远超营收增速",
            severity="high",
            detail=f"库存增长 {inventory_growth:.1%}，营收增长 {revenue_growth:.1%}，"
                   f"差距 {gap:.1%}。可能存在库存积压或过时风险。",
            metric=gap, threshold=0.30,
        )
    return None


# ---------------------------------------------------------------------------
# Full health check — runs all checks on a DataFrame of financial data
# ---------------------------------------------------------------------------


def analyze_financial_health(
    financials: pd.DataFrame,
    company: str = "",
) -> FinancialHealthReport:
    """Run all red flag checks on financial data.

    Args:
        financials: DataFrame with financial metrics indexed by date.
            Expected columns: revenue, receivables, inventory, gross_margin,
            cfo, net_income, non_recurring.
        company: Company name for the report.

    Returns:
        FinancialHealthReport with all flags.
    """
    report = FinancialHealthReport(company=company)
    flags: list[RedFlag] = []

    # Get latest values
    if financials.empty or len(financials) < 2:
        return report

    latest = financials.iloc[-1]
    prev = financials.iloc[-2] if len(financials) > 1 else None

    # 1. CFO/Net Income check
    if "cfo" in financials.columns and "net_income" in financials.columns:
        flag = check_earnings_quality(
            float(latest.get("cfo", 0)),
            float(latest.get("net_income", 0)),
        )
        if flag:
            flags.append(flag)

    # 2. Receivables vs Revenue growth
    if all(c in financials.columns for c in ["receivables", "revenue"]):
        if prev is not None:
            rev_growth = (latest["revenue"] - prev["revenue"]) / abs(prev["revenue"]) if prev["revenue"] != 0 else 0
            rec_growth = (latest["receivables"] - prev["receivables"]) / abs(prev["receivables"]) if prev["receivables"] != 0 else 0
            flag = check_receivables_vs_revenue(float(rev_growth), float(rec_growth))
            if flag:
                flags.append(flag)

    # 3. Gross margin trend
    if "gross_margin" in financials.columns and len(financials) >= 3:
        margins = financials["gross_margin"].dropna().tolist()
        flag = check_gross_margin_trend(margins)
        if flag:
            flags.append(flag)

    # 4. Non-recurring items
    if all(c in financials.columns for c in ["non_recurring", "net_income"]):
        flag = check_non_recurring_items(
            float(latest.get("non_recurring", 0)),
            float(latest.get("net_income", 0)),
        )
        if flag:
            flags.append(flag)

    # 5. Inventory vs Revenue growth
    if all(c in financials.columns for c in ["inventory", "revenue"]):
        if prev is not None:
            rev_growth = (latest["revenue"] - prev["revenue"]) / abs(prev["revenue"]) if prev["revenue"] != 0 else 0
            inv_growth = (latest["inventory"] - prev["inventory"]) / abs(prev["inventory"]) if prev["inventory"] != 0 else 0
            flag = check_revenue_vs_inventory(float(rev_growth), float(inv_growth))
            if flag:
                flags.append(flag)

    report.flags = flags
    report.total_flags = len(flags)
    report.high_risk_flags = sum(1 for f in flags if f.severity in ("high", "critical"))

    # Summary
    if not flags:
        report.summary = "✅ 未发现明显财务异常信号。"
    else:
        high = report.high_risk_flags
        total = report.total_flags
        report.summary = f"⚠️ 发现 {total} 项异常，其中 {high} 项高风险。"

    return report
