"""Shareholder structure analysis for A-share companies.

Analyzes top-10 shareholder data to extract:
1. Shareholder type: state-owned, foreign institutional, domestic institutional
2. Ownership concentration: top-1 ratio, top-10 ratio, Z-index, checks-and-balances
3. Dynamic changes: new entries, increases, decreases (future)

Data source: akshare (stock_gdfx_top_10_em) — free, no API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# Keyword lists for shareholder classification

STATE_OWN_KEYWORDS = [
    "国有资产监督委员会",
    "中央汇金资产管理有限责任公司",
    "中央汇金投资有限责任公司",
    "全国社保基金",
    "中国证券金融股份有限公司",
    "财政局", "财政部", "人民政府",
    "国有资本经营管理有限公司",
    "市投资", "省投资",
    "城投控股",
    "国资委",
]

FOREIGN_CAPITAL_KEYWORDS = [
    "香港中央結算有限公司",
    "香港中央结算有限公司",
    "瑞士联合银行集团", "瑞士嘉盛银行",
    "汇丰", "花旗", "渣打", "法国巴黎银行",
    "高盛", "Goldman Sachs",
    "贝莱德", "Blackrock", "BLACKROCK",
    "摩根大通", "摩根士丹利",
    "美林", "Merrill Lynch",
    "巴克莱", "Barclays",
    "阿布达比投资局",
    "科威特政府投资局",
    "淡马锡",
    "Limited", "LIMITED",
]

DOMESTIC_INSTITUTION_KEYWORDS = [
    "保险产品", "保险责任有限公司",
    "股票型证券投资基金",
    "混合型证券投资基金",
    "混合型发起式证券投资基金",
    "股票型发起式证券投资基金",
    "证券股份有限公司", "证券有限公司",
    "资产管理计划",
]

ETF_FILTER_KEYWORDS = [
    "交易性开放式指数证券投资基金",
    "沪深300", "中证500", "中证1000",
    "ETF",
]


@dataclass
class ShareholderInfo:
    """Information about a single top-10 shareholder."""
    name: str = ""
    hold_pct: float = 0.0
    is_state_owned: bool = False
    is_foreign: bool = False
    is_domestic_inst: bool = False
    is_etf: bool = False


@dataclass
class ShareholderStructure:
    """Shareholder structure analysis for one stock."""
    code: str = ""
    n_shareholders: int = 0
    top1_pct: float = 0.0
    top10_pct: float = 0.0
    has_state_owned: bool = False
    has_foreign: bool = False
    has_domestic_inst: bool = False
    z_index: float = 0.0        # 第一大 / 第二大
    checks_balance: float = 0.0  # 第二至第十 / 第一大
    concentration_label: str = "未知"
    shareholders: list[ShareholderInfo] = field(default_factory=list)
    raw_data: pd.DataFrame = field(default_factory=pd.DataFrame)


def _keyword_match(name: str, keywords: list[str]) -> bool:
    """Check if shareholder name matches any keyword."""
    for kw in keywords:
        if kw in name:
            return True
    return False


def _is_etf(name: str) -> bool:
    """Check if shareholder is an ETF fund."""
    return _keyword_match(name, ETF_FILTER_KEYWORDS)


def _classify_shareholder(name: str) -> dict:
    """Classify a shareholder by type.

    Returns:
        Dict with is_state_owned, is_foreign, is_domestic_inst.
    """
    result = {
        "is_state_owned": False,
        "is_foreign": False,
        "is_domestic_inst": False,
    }
    if _keyword_match(name, STATE_OWN_KEYWORDS):
        result["is_state_owned"] = True
    elif _keyword_match(name, FOREIGN_CAPITAL_KEYWORDS):
        result["is_foreign"] = True
    elif _keyword_match(name, DOMESTIC_INSTITUTION_KEYWORDS) and not _is_etf(name):
        result["is_domestic_inst"] = True
    return result


def classify_shareholders(
    df: pd.DataFrame,
    code: str = "",
) -> ShareholderStructure:
    """Classify top-10 shareholders and compute ownership structure.

    Args:
        df: DataFrame with columns: holder_name, holder_pct (and optionally
            hold_num, change, change_ratio). Can be fetched via akshare's
            stock_gdfx_top_10_em().
        code: Stock code.

    Returns:
        ShareholderStructure with classification and concentration metrics.
    """
    structure = ShareholderStructure(code=code)
    if df.empty:
        return structure

    # Normalize column names
    rename_map = {
        "股东名称": "holder_name",
        "持股数": "hold_num",
        "占总股本持股比例": "holder_pct",
        "增减": "change",
        "变动比率": "change_ratio",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.reset_index(drop=True)

    structure.raw_data = df
    structure.n_shareholders = len(df)

    # Classify each shareholder
    shareholders = []
    for _, row in df.iterrows():
        name = str(row.get("holder_name", ""))
        pct = float(row.get("holder_pct", 0))
        cls = _classify_shareholder(name)
        info = ShareholderInfo(
            name=name,
            hold_pct=pct / 100 if pct > 1 else pct,  # Normalize percentage
            is_state_owned=cls["is_state_owned"],
            is_foreign=cls["is_foreign"],
            is_domestic_inst=cls["is_domestic_inst"],
            is_etf=_is_etf(name),
        )
        shareholders.append(info)
    structure.shareholders = shareholders

    # Ownership concentration
    if shareholders:
        pcts = [s.hold_pct for s in shareholders]
        structure.top1_pct = pcts[0] if len(pcts) >= 1 else 0.0
        structure.top10_pct = sum(pcts)

        # Presence flags
        structure.has_state_owned = any(s.is_state_owned for s in shareholders)
        structure.has_foreign = any(s.is_foreign for s in shareholders)
        structure.has_domestic_inst = any(s.is_domestic_inst for s in shareholders)

        # Z-index = top1 / top2 (measures dominance of largest shareholder)
        if len(pcts) >= 2 and pcts[1] > 0:
            structure.z_index = round(pcts[0] / pcts[1], 2)

        # Checks-and-balance = sum(top2..10) / top1
        if len(pcts) >= 2 and pcts[0] > 0:
            structure.checks_balance = round(sum(pcts[1:]) / pcts[0], 2)

        # Concentration label
        if structure.top1_pct >= 0.50:
            structure.concentration_label = "高度集中"
        elif structure.top1_pct >= 0.30:
            structure.concentration_label = "相对集中"
        elif structure.top10_pct >= 0.50:
            structure.concentration_label = "相对分散"
        else:
            structure.concentration_label = "高度分散"

    return structure


def concentration_score(structure: ShareholderStructure) -> float:
    """Convert shareholder structure to a factor score [0, 1].

    Higher score = more concentrated ownership.
    Useful as a factor input.
    """
    return min(structure.top1_pct * 1.5, 1.0)


def institutional_presence_score(structure: ShareholderStructure) -> float:
    """Score based on presence of institutional shareholders [0, 1].

    Higher score = more institutional interest.
    """
    score = 0.0
    if structure.has_state_owned:
        score += 0.3
    if structure.has_foreign:
        score += 0.5
    if structure.has_domestic_inst:
        score += 0.4
    # Cap
    return min(score + 0.1 if structure.top10_pct > 0.4 else score, 1.0)


def print_shareholder_summary(structure: ShareholderStructure) -> str:
    """Generate a human-readable summary of shareholder structure."""
    lines = [
        f"股东结构分析: {structure.code}",
        f"  股东数量: {structure.n_shareholders}",
        f"  第一大股东: {structure.top1_pct:.1%}",
        f"  前十大合计: {structure.top10_pct:.1%}",
        f"  集中度: {structure.concentration_label}",
        f"  Z指数: {structure.z_index}",
        f" 股权制衡度: {structure.checks_balance}",
        f"  国资持股: {'是' if structure.has_state_owned else '否'}",
        f"  外资持股: {'是' if structure.has_foreign else '否'}",
        f"  内资机构: {'是' if structure.has_domestic_inst else '否'}",
    ]
    return "\n".join(lines)
