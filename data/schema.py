"""Data schemas and validation utilities.

Defines expected column names, dtypes, and validation helpers for the
data pipeline to ensure data integrity at each stage.
"""

from __future__ import annotations

import pandas as pd

# Expected columns in price data
PRICE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "turnover", "adj_factor", "vwap",
]

# Expected columns in financial data
FINANCIAL_COLUMNS = [
    "market_cap", "total_assets", "net_assets",
    "revenue", "net_profit", "roe", "pb_ratio",
    "pe_ratio", "asset_growth",
]

# Expected columns in metadata
METADATA_COLUMNS = [
    "sector", "market_cap_group", "is_st",
    "listing_date", "delisting_date",
]

# A-share industry sectors (for synthetic data)
SECTORS = [
    "银行", "非银金融", "房地产", "建筑材料", "建筑装饰",
    "机械设备", "电力设备", "国防军工", "汽车", "家用电器",
    "食品饮料", "医药生物", "农林牧渔", "纺织服饰", "轻工制造",
    "电子", "计算机", "通信", "传媒",
    "基础化工", "石油石化", "钢铁", "有色金属", "煤炭",
    "公用事业", "交通运输", "环保",
    "商贸零售", "社会服务", "综合",
]

# Sectors with assigned probabilities for realistic distribution
SECTOR_WEIGHTS = {
    "医药生物": 0.08, "电子": 0.08, "计算机": 0.07,
    "机械设备": 0.06, "基础化工": 0.06, "电力设备": 0.06,
    "食品饮料": 0.05, "汽车": 0.05, "房地产": 0.04,
    "非银金融": 0.04, "银行": 0.03, "有色金属": 0.04,
    "传媒": 0.04, "通信": 0.03, "国防军工": 0.03,
    "交通运输": 0.03, "公用事业": 0.03, "建筑装饰": 0.03,
    "家用电器": 0.03, "农林牧渔": 0.02, "商贸零售": 0.02,
    "建筑材料": 0.02, "轻工制造": 0.02, "纺织服饰": 0.02,
    "钢铁": 0.02, "煤炭": 0.01, "石油石化": 0.01,
    "环保": 0.02, "社会服务": 0.02, "综合": 0.01,
}


def validate_prices(df: pd.DataFrame) -> None:
    """Validate price DataFrame has required columns and index."""
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Price data must have MultiIndex (date, asset)")
    missing = set(PRICE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing price columns: {missing}")


def validate_financials(df: pd.DataFrame) -> None:
    """Validate financial DataFrame has required columns and index."""
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Financial data must have MultiIndex (date, asset)")
    missing = set(FINANCIAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing financial columns: {missing}")
