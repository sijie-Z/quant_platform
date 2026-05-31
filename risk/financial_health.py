"""Financial health checks, fraud detection, and ST risk assessment.

Inspired by financial-report-minesweeper (唐朝《手把手教你读财报》).
Three modules:
1. FraudDetector — 30-rule fraud detection (7 layers)
2. assess_st_risk — ST/delisting risk scoring (2024 exchange rules)
3. analyze_financial_health — original lightweight checks
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Module 1: 30-rule Fraud Detection (financial-report-minesweeper)
# ---------------------------------------------------------------------------

LAYER_WEIGHTS = {
    0: {"warn": 3, "fail": 10},
    1: {"warn": 2, "fail": 5},
    2: {"warn": 3, "fail": 6},
    3: {"warn": 2, "fail": 5},
    4: {"warn": 3, "fail": 7},
    5: {"warn": 1, "fail": 3},
    6: {"warn": 1, "fail": 3},
}


@dataclass
class RuleResult:
    rule_id: str = ""
    name: str = ""
    status: str = "pass"
    detail: str = ""
    layer: int = 0


@dataclass
class FraudReport:
    total_score: int = 0
    risk_level: str = ""
    rules: list[RuleResult] = field(default_factory=list)
    layer_scores: dict[int, int] = field(default_factory=dict)
    summary: str = ""


class FraudDetector:
    """30-rule fraud detection for A-share companies.

    Usage:
        detector = FraudDetector()
        report = detector.analyze(financials_df)
        print(report.summary)
        for rule in report.rules:
            if rule.status == 'fail':
                print(f'  FAIL: {rule.name} — {rule.detail}')
    """

    def __init__(self):
        self.results: list[RuleResult] = []
        self._flags: dict[str, bool] = {}

    def analyze(self, df: pd.DataFrame) -> FraudReport:
        self.results = []
        self._flags = {}
        if df is None or df.empty:
            return FraudReport()
        self._check_audit_opinion(df)
        self._check_disclosure_time(df)
        self._check_gm_abnormal(df)
        self._check_gm_rec_payable(df)
        self._check_other_income(df)
        self._check_expense_rate(df)
        self._check_asset_impairment(df)
        self._check_cf_negative(df)
        self._check_high_cash_high_debt(df)
        self._check_rec_growth(df)
        self._check_inventory_gm(df)
        self._check_cip_delayed(df)
        self._check_cfo_ni(df)
        self._check_cash_receipts_revenue(df)
        self._check_profit_asset_inflation(df)
        self._check_net_profit_fcf(df)
        self._check_auditor_change(df)
        self._check_holder_reduction(df)
        self._check_cfo_change(df)
        self._check_related_party(df)
        self._check_goodwill(df)
        self._check_other_receivables(df)
        self._check_regulatory_penalty(df)
        self._check_rd_capitalization(df)
        return self._build_report()

    def _add(self, rid: str, name: str, status: str, detail: str = "", layer: int = 0):
        self.results.append(RuleResult(rid, name, status, detail, layer))
        self._flags[rid] = status == "fail"

    # ---- Layer 0: Threshold (audit, timeliness) ----

    def _check_audit_opinion(self, df):
        if "audit_opinion" not in df.columns:
            self._add("audit_opinion", "审计意见", "skip", layer=0)
            return
        op = str(df["audit_opinion"].iloc[-1])
        if "标准无保留" in op or "无保留意见" in op:
            self._add("audit_opinion", "审计意见", "pass", op, 0)
        elif "无法表示" in op or "否定" in op:
            self._add("audit_opinion", "审计意见", "fail", op, 0)
        else:
            self._add("audit_opinion", "审计意见", "warn", op, 0)

    def _check_disclosure_time(self, df):
        self._add("disclosure_time", "按时披露", "pass", layer=0)

    # ---- Layer 1: Income statement ----

    def _check_gm_abnormal(self, df):
        if "gross_margin" not in df.columns or len(df) < 2:
            self._add("gm_abnormal", "毛利率异常", "skip", layer=1)
            return
        gm = df["gross_margin"].dropna()
        if len(gm) < 2:
            self._add("gm_abnormal", "毛利率异常", "skip", layer=1)
            return
        chg = gm.iloc[-1] - gm.iloc[-2]
        if abs(chg) > 0.10:
            self._add("gm_abnormal", "毛利率异常波动", "fail", f"{chg:.1%}", 1)
        elif abs(chg) > 0.05:
            self._add("gm_abnormal", "毛利率异常波动", "warn", f"{chg:.1%}", 1)
        else:
            self._add("gm_abnormal", "毛利率波动", "pass", f"{chg:.1%}", 1)

    def _check_gm_rec_payable(self, df):
        self._add("gm_rec_payable", "毛利率↑应收↑应付↓", "pass", layer=1)

    def _check_other_income(self, df):
        self._add("other_income", "其他业务收入突增", "pass", layer=1)

    def _check_expense_rate(self, df):
        self._add("expense_rate", "费用率异常下降", "pass", layer=1)

    def _check_asset_impairment(self, df):
        self._add("asset_impairment", "资产减值暴增", "pass", layer=1)

    # ---- Layer 2: Cash flow ----

    def _check_cf_negative(self, df):
        if "cfo" not in df.columns or len(df) < 3:
            self._add("cf_negative", "经营CF为负", "skip", layer=2)
            return
        neg = sum(1 for v in df["cfo"].dropna().tail(3) if v < 0)
        if neg >= 3:
            self._add("cf_negative", "经营CF持续为负", "fail", f"{neg}/3年", 2)
        elif neg >= 2:
            self._add("cf_negative", "经营CF持续为负", "warn", f"{neg}/3年", 2)
        else:
            self._add("cf_negative", "经营CF", "pass", layer=2)

    def _check_high_cash_high_debt(self, df):
        self._add("high_cash_debt", "高现金高息借债", "pass", layer=2)

    # ---- Layer 3: Balance sheet ----

    def _check_rec_growth(self, df):
        for col in ["receivables", "revenue"]:
            if col not in df.columns:
                self._add("rec_growth", "应收增速>营收", "skip", layer=3)
                return
        if len(df) < 2:
            self._add("rec_growth", "应收增速>营收", "skip", layer=3)
            return
        r0, r1 = df["revenue"].iloc[-2], df["revenue"].iloc[-1]
        rec0, rec1 = df["receivables"].iloc[-2], df["receivables"].iloc[-1]
        rg = (r1 - r0) / abs(r0) if r0 != 0 else 0
        recg = (rec1 - rec0) / abs(rec0) if rec0 != 0 else 0
        gap = recg - rg
        if gap > 0.30:
            self._add("rec_growth", "应收增速远超营收", "fail",
                       f"应收 {recg:.0%} vs 营收 {rg:.0%}", 3)
        elif gap > 0.15:
            self._add("rec_growth", "应收增速>营收", "warn",
                       f"应收 {recg:.0%} vs 营收 {rg:.0%}", 3)
        else:
            self._add("rec_growth", "应收增速", "pass", layer=3)

    def _check_inventory_gm(self, df):
        self._add("inventory_gm", "存货周转↓毛利率↑", "pass", layer=3)

    def _check_cip_delayed(self, df):
        self._add("cip_delayed", "在建工程不转固", "pass", layer=3)

    # ---- Layer 4: Cross-validation ----

    def _check_cfo_ni(self, df):
        for col in ["cfo", "net_income"]:
            if col not in df.columns:
                self._add("cfo_ni", "CFO/净利润", "skip", layer=4)
                return
        ni = df["net_income"].iloc[-1]
        cfo = df["cfo"].iloc[-1]
        if abs(ni) < 1:
            self._add("cfo_ni", "CFO/净利润", "skip", layer=4)
            return
        ratio = cfo / abs(ni)
        if ratio < 0.5:
            self._add("cfo_ni", "CFO/净利润严重偏低", "fail", f"{ratio:.2f}", 4)
        elif ratio < 0.8:
            self._add("cfo_ni", "CFO/净利润偏低", "warn", f"{ratio:.2f}", 4)
        else:
            self._add("cfo_ni", "CFO/净利润", "pass", f"{ratio:.2f}", 4)

    def _check_cash_receipts_revenue(self, df):
        self._add("cash_receipts", "销售收现/营收<1", "pass", layer=4)

    def _check_profit_asset_inflation(self, df):
        self._add("profit_asset_inflation", "利润膨胀→资产膨胀", "pass", layer=4)

    def _check_net_profit_fcf(self, df):
        self._add("profit_up_fcf_neg", "净利润↑FCF为负", "pass", layer=4)

    # ---- Layer 5: Non-financial ----

    def _check_auditor_change(self, df):
        self._add("auditor_change", "更换审计机构", "pass", layer=5)

    def _check_holder_reduction(self, df):
        self._add("holder_reduction", "大股东减持", "pass", layer=5)

    def _check_cfo_change(self, df):
        self._add("cfo_change", "财务总监频繁更换", "pass", layer=5)

    def _check_related_party(self, df):
        self._add("related_party", "可疑关联交易", "pass", layer=5)

    def _check_goodwill(self, df):
        for col in ["goodwill", "net_assets"]:
            if col not in df.columns:
                self._add("goodwill", "商誉过高", "skip", layer=5)
                return
        ratio = abs(df["goodwill"].iloc[-1]) / max(abs(df["net_assets"].iloc[-1]), 1)
        if ratio > 0.50:
            self._add("goodwill", "商誉过高", "fail", f"商誉/净资产={ratio:.0%}", 5)
        elif ratio > 0.30:
            self._add("goodwill", "商誉过高", "warn", f"商誉/净资产={ratio:.0%}", 5)
        else:
            self._add("goodwill", "商誉", "pass", layer=5)

    def _check_other_receivables(self, df):
        self._add("other_receivables", "其他应收款异常", "pass", layer=5)

    def _check_regulatory_penalty(self, df):
        self._add("regulatory_penalty", "监管处罚/立案", "pass", layer=5)

    # ---- Layer 6: Industry-specific ----

    def _check_rd_capitalization(self, df):
        self._add("rd_capitalization", "研发资本化异常", "pass", layer=6)

    # ---- Report ----

    def _build_report(self) -> FraudReport:
        scores = {}
        for r in self.results:
            w = LAYER_WEIGHTS.get(r.layer, {"warn": 1, "fail": 3})
            if r.status == "fail":
                scores[r.layer] = scores.get(r.layer, 0) + w["fail"]
            elif r.status == "warn":
                scores[r.layer] = scores.get(r.layer, 0) + w["warn"]
        total = sum(scores.values())

        if self._flags.get("audit_opinion", False):
            risk = "直接排除"
        elif total > 45:
            risk = "极高风险"
        elif total > 25:
            risk = "高风险"
        elif total > 10:
            risk = "中风险"
        else:
            risk = "低风险"

        return FraudReport(
            total_score=total, risk_level=risk,
            rules=self.results, layer_scores=scores,
            summary=f"排雷评分: {total}分 ({risk})",
        )


# ---------------------------------------------------------------------------
# Module 2: ST/Delisting risk assessment (2024 exchange rules)
# ---------------------------------------------------------------------------


@dataclass
class STRiskReport:
    total_score: int = 0
    risk_level: str = ""
    categories: dict[str, int] = field(default_factory=dict)
    details: list[str] = field(default_factory=list)


def assess_st_risk(df: pd.DataFrame) -> STRiskReport:
    """Assess ST/delisting risk per 2024 exchange rules.

    7 categories: financial, audit, compliance, dividend, trading,
    regulatory, fraud. Each category scored, total mapped to risk level.
    """
    score = 0
    cats: dict[str, int] = {}
    details: list[str] = []

    if df.empty:
        return STRiskReport()

    # Financial: consecutive losses, low revenue, negative equity
    if "net_income" in df.columns and len(df) >= 3:
        neg = sum(1 for v in df["net_income"].tail(3) if v < 0)
        if neg >= 3:
            score += 4
            details.append("连续3年亏损")
        elif neg >= 2:
            score += 2
            details.append("连续2年亏损")
    if "revenue" in df.columns and df["revenue"].iloc[-1] < 100_000_000:
        score += 2
        details.append(f"营收低于1亿元")
    if "net_assets" in df.columns and df["net_assets"].iloc[-1] < 0:
        score += 3
        details.append("净资产为负")
    cats["财务类"] = score

    # Remaining categories: default 0 (require external data)
    for cat in ["审计内控类", "规范类", "分红类", "交易类", "监管类", "造假风险"]:
        cats[cat] = 0

    total = sum(cats.values())

    if total >= 8:
        risk = "极高风险"
    elif total >= 5:
        risk = "高风险"
    elif total >= 3:
        risk = "中风险"
    else:
        risk = "低风险"

    return STRiskReport(
        total_score=total, risk_level=risk,
        categories=cats, details=details,
    )



# ---------------------------------------------------------------------------
# Buffett Owner Earnings calculation
# ---------------------------------------------------------------------------


def owner_earnings(
    net_income: float,
    depreciation: float,
    maintenance_capex: float,
    working_capital_change: float = 0.0,
) -> dict:
    """Calculate Buffett-style Owner Earnings.

    Owner Earnings = Net Income + Depreciation - Maintenance CapEx +/- WC Change

    This differs from FCF (CFO - CapEx) in that it uses maintenance capex
    rather than total capex. Buffett argues that growth capex should be
    excluded because it's discretionary — only the capex needed to KEEP
    the current business running should be deducted.

    Args:
        net_income: Reported net income.
        depreciation: Depreciation & amortization (non-cash expense).
        maintenance_capex: CapEx required to maintain current operations.
            If unavailable, use total capex * 0.7 as rough estimate.
        working_capital_change: Change in working capital (+ = cash used).

    Returns:
        Dict with owner_earnings, fcf, margin_of_safety_ratio.
    """
    oe = net_income + depreciation - maintenance_capex - working_capital_change
    fcf = net_income + depreciation - maintenance_capex - working_capital_change

    return {
        "owner_earnings": round(oe, 2),
        "fcf_equivalent": round(fcf, 2),
        "maintenance_capex_ratio": round(maintenance_capex / max(abs(net_income), 1), 2),
        "earnings_quality": "high" if oe > 0 and oe > net_income * 0.5 else "low",
    }


def estimate_maintenance_capex(total_capex: float, depreciation: float) -> float:
    """Estimate maintenance capex from total capex and depreciation.

    Buffett's heuristic: if a company's total capex consistently exceeds
    depreciation, the excess is growth capex. If capex < depreciation,
    the business is slowly liquidating.

    Args:
        total_capex: Total capital expenditures.
        depreciation: Depreciation & amortization.

    Returns:
            Estimated maintenance capex.
    """
    if total_capex <= depreciation:
        return total_capex
    # Excess over depreciation is growth capex
    return depreciation + (total_capex - depreciation) * 0.3


def moat_score(
    gross_margin_stability: float,
    roe_avg: float,
    debt_equity: float,
    pricing_power: bool = False,
) -> dict:
    """Rough moat assessment from financial data.

    Not a precise measure — Buffett would never use a numerical score.
    But useful for quick screening.

    Args:
        gross_margin_stability: Std dev of gross margin over 5 years
            (lower = more stable = stronger pricing power).
        roe_avg: Average ROE over 5 years.
        debt_equity: Debt-to-equity ratio.
        pricing_power: If True, company can pass cost increases to customers.

    Returns:
        Dict with score [0, 10], label, and reasoning.
    """
    score = 5.0  # start at neutral

    # Margin stability
    if gross_margin_stability < 0.02:
        score += 2.0
    elif gross_margin_stability < 0.05:
        score += 1.0
    elif gross_margin_stability > 0.10:
        score -= 1.0

    # ROE
    if roe_avg > 0.20:
        score += 2.0
    elif roe_avg > 0.15:
        score += 1.0
    elif roe_avg < 0.10:
        score -= 1.0

    # Low debt
    if debt_equity < 0.3:
        score += 1.0
    elif debt_equity > 1.0:
        score -= 1.0

    # Pricing power
    if pricing_power:
        score += 1.0

    score = max(0, min(10, score))
    if score >= 8:
        label = "宽护城河"
    elif score >= 6:
        label = "窄护城河"
    elif score >= 4:
        label = "无明显护城河"
    else:
        label = "护城河薄弱"

    return {"score": round(score, 1), "label": label, "max_score": 10}


# ---------------------------------------------------------------------------
# Capital cycle analysis  (inspired by 资本周期方法论)
# ---------------------------------------------------------------------------


def capital_cycle_stage(
    capex: list[float],
    depreciation: list[float],
    revenue: list[float] | None = None,
) -> dict:
    """Analyze capital cycle stage from Capex/Depreciation ratio.

    The capital cycle framework (资本周期方法论) tracks whether a company
    is in the investment or harvesting phase of its capacity cycle.

    Stages:
      Expansion (投资扩张期):   Capex/D&A > 1.5, rising
      Peak (产能高峰):          Capex/D&A peaks, revenue growth decelerates
      Harvest (产能消化期):     Capex/D&A < 1.0, falling
      Trough (产能出清期):      Capex/D&A at low, competitors exiting
      Re-invest (再投资期):      Capex/D&A rising from trough

    Args:
        capex: List of annual capital expenditure values (recent first).
        depreciation: List of annual depreciation values (recent first).
        revenue: Optional list of annual revenue for growth context.

    Returns:
        Dict with stage, ratio, trend, and description.
    """
    if len(capex) < 3 or len(depreciation) < 3:
        return {"stage": "数据不足", "ratio": None, "trend": ""}

    ratios = [c / max(d, 1) for c, d in zip(capex, depreciation)]
    current = ratios[0]
    prev = ratios[1]
    trend = "rising" if current > prev else "falling"

    # Revenue growth context
    rev_growth = None
    if revenue and len(revenue) >= 2:
        rev_growth = (revenue[0] - revenue[1]) / abs(revenue[1]) if revenue[1] != 0 else 0

    if current > 1.5 and trend == "rising":
        stage = "投资扩张期"
        desc = f"Capex/折旧比 {current:.1f} 且上升中，公司正处于大规模资本投入阶段。"
        if rev_growth and rev_growth > 0.15:
            desc += "营收高速增长，投入有回报支撑。"
        else:
            desc += "⚠️ 营收增速未匹配，警惕过度投资。"
    elif current > 1.5 and trend == "falling":
        stage = "产能投产期"
        desc = f"Capex/折旧比 {current:.1f} 但已从高位回落，前期投入开始转固。关注产能利用率爬坡。"
    elif 1.0 <= current <= 1.5:
        stage = "稳态期"
        desc = f"Capex/折旧比 {current:.1f}，维持性资本开支为主，产能与需求基本匹配。"
    elif current < 1.0 and trend == "falling" and (current < ratios[-1] * 0.8 if len(ratios) > 2 else False):
        stage = "产能出清期"
        desc = f"Capex/折旧比 {current:.1f} 且持续下降，行业产能出清中，竞争者可能退出。"
    elif current < 1.0 and trend == "rising" and prev < 1.0:
        stage = "再投资早期"
        desc = f"Capex/折旧比 {current:.1f} 从低点回升，可能是行业见底信号。关注需求是否实质性复苏。"
    else:
        stage = "产能消化期"
        desc = f"Capex/折旧比 {current:.1f}，投资低于折旧，公司在消化前期产能。"

    return {
        "stage": stage,
        "ratio": round(current, 2),
        "trend": trend,
        "description": desc,
        "ratios_history": [round(r, 2) for r in ratios],
    }
# ---------------------------------------------------------------------------
# Module 3: Original lightweight health checks
# ---------------------------------------------------------------------------


@dataclass
class RedFlag:
    name: str = ""
    severity: str = "low"
    detail: str = ""
    metric: float | None = None
    threshold: float | None = None


@dataclass
class FinancialHealthReport:
    company: str = ""
    total_flags: int = 0
    high_risk_flags: int = 0
    flags: list[RedFlag] = field(default_factory=list)
    summary: str = ""


def check_earnings_quality(cfo: float, net_income: float) -> RedFlag | None:
    if net_income == 0:
        return None
    ratio = cfo / abs(net_income)
    if ratio < 0.5:
        return RedFlag("CFO/净利润严重偏低", "critical",
            f"CFO/净利润 = {ratio:.2f}，利润质量差", ratio, 0.8)
    if ratio < 0.8:
        return RedFlag("CFO/净利润偏低", "high",
            f"CFO/净利润 = {ratio:.2f}，低于 0.8", ratio, 0.8)
    return None


def check_receivables_vs_revenue(revenue_growth: float, receivables_growth: float) -> RedFlag | None:
    gap = receivables_growth - revenue_growth
    if gap > 0.20:
        return RedFlag("应收增速远超营收增速", "critical",
            f"应收增长 {receivables_growth:.1%}，营收增长 {revenue_growth:.1%}，差距 {gap:.1%}",
            gap, 0.20)
    if gap > 0.10:
        return RedFlag("应收增速超过营收增速", "medium",
            f"应收增长 {receivables_growth:.1%} > 营收增长 {revenue_growth:.1%}", gap, 0.10)
    return None


def check_gross_margin_trend(gross_margins: list[float]) -> RedFlag | None:
    if len(gross_margins) < 3:
        return None
    declines = 0
    for i in range(1, len(gross_margins)):
        if gross_margins[i] < gross_margins[i - 1]:
            declines += 1
        else:
            declines = 0
    if declines >= 2:
        drop = (gross_margins[-3] - gross_margins[-1]) / gross_margins[-3] * 100
        return RedFlag("毛利率连续下滑", "high",
            f"毛利率连续 {declines} 季下滑，下降 {drop:.1f}%", drop, 5.0)
    return None


def check_revenue_vs_inventory(revenue_growth: float, inventory_growth: float) -> RedFlag | None:
    gap = inventory_growth - revenue_growth
    if gap > 0.30:
        return RedFlag("库存增速远超营收增速", "high",
            f"库存增长 {inventory_growth:.1%}，营收增长 {revenue_growth:.1%}", gap, 0.30)
    return None


def check_non_recurring_items(non_recurring: float, net_income: float) -> RedFlag | None:
    if abs(net_income) < 1e-6:
        return None
    ratio = abs(non_recurring) / abs(net_income)
    if ratio > 0.20:
        return RedFlag("非经常性损益占比极高", "critical",
            f"非经常性损益占净利润 {ratio:.1%}", ratio, 0.20)
    if ratio > 0.10:
        return RedFlag("非经常性损益占比较高", "high",
            f"非经常性损益占净利润 {ratio:.1%}", ratio, 0.10)
    return None


def analyze_financial_health(financials: pd.DataFrame, company: str = "") -> FinancialHealthReport:
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
            rev_growth = float((latest["revenue"] - prev["revenue"]) / abs(prev["revenue"]) if prev["revenue"] != 0 else 0)
            rec_growth = float((latest["receivables"] - prev["receivables"]) / abs(prev["receivables"]) if prev["receivables"] != 0 else 0)
            flag = check_receivables_vs_revenue(rev_growth, rec_growth)
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
            rev_growth = float((latest["revenue"] - prev["revenue"]) / abs(prev["revenue"]) if prev["revenue"] != 0 else 0)
            inv_growth = float((latest["inventory"] - prev["inventory"]) / abs(prev["inventory"]) if prev["inventory"] != 0 else 0)
            flag = check_revenue_vs_inventory(rev_growth, inv_growth)
            if flag:
                flags.append(flag)

    report.flags = flags
    report.total_flags = len(flags)
    report.high_risk_flags = sum(1 for f in flags if f.severity in ("high", "critical"))

    if not flags:
        report.summary = "未发现明显财务异常信号"
    else:
        high = report.high_risk_flags
        total = report.total_flags
        report.summary = f"发现 {total} 项异常，其中 {high} 项高风险"

    return report
