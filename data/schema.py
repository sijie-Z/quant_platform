"""Data schemas and validation utilities.

Defines expected column names, dtypes, and validation helpers for the
data pipeline to ensure data integrity at each stage.

Contract (enforced at the provider boundary by :meth:`DataPipeline.run`):

* Prices -- MultiIndex ``(date, asset)``, levels named ``date``/``asset``,
  sorted, with at least :data:`REQUIRED_PRICE_COLUMNS`.  ``vwap``/``amount``
  are optional because not every provider ships them.  ``volume`` is in
  shares and ``turnover`` is a fraction (see the provider wave's contract);
  the pipeline does not convert units.
* Financials -- MultiIndex ``(date, asset)`` keyed to the first trading day
  on/after ``publish_date``, with :data:`FINANCIAL_COLUMNS`.  An entirely
  empty frame is legal (a provider may have no fundamentals); the pipeline
  normalises it to the documented empty panel.  ``publish_date`` is expected
  from conforming providers but is not required here because the PIT
  accessor has a documented failure mode for providers that cannot supply it.
* Metadata -- indexed by ``asset`` with :data:`REQUIRED_METADATA_COLUMNS`.
  ``is_st`` may be absent or empty when the provider has no ST history; it
  is deliberately not required and the pipeline warns instead of assuming
  ``False``.

Violations raise :class:`DataContractError` naming the provider, the dataset
and the offending index level/column, so a provider bug fails at the
boundary with an actionable message instead of a downstream ``KeyError``.
"""

from __future__ import annotations

import pandas as pd

# Columns every price provider must supply.  `vwap` and `amount` are
# optional: Baostock/Postgres do not ship vwap, and neither synthetic nor
# several real feeds ship amount.  Consumers already treat them as optional
# (`_derive_suspensions` checks column presence; nothing indexes `amount`).
PRICE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "turnover", "adj_factor",
]
OPTIONAL_PRICE_COLUMNS = ["vwap", "amount", "pct_chg"]

# The column set consumed by the factor layer (`factors/fundamental.py`);
# a provider missing any of these silently kills a factor, so it is a hard
# contract violation.  Ratio names follow the repo convention
# (`pb_ratio`/`pe_ratio`) rather than the provider-native `pb`/`pe`.
FINANCIAL_COLUMNS = [
    "market_cap", "total_assets", "net_assets",
    "revenue", "net_profit", "roe", "pb_ratio",
    "pe_ratio", "asset_growth",
]

# Columns the pipeline itself reads.  `is_st` may be absent/None when the
# provider cannot supply ST history -- absence must not be read as "not ST".
METADATA_COLUMNS = [
    "sector", "market_cap_group", "is_st",
    "listing_date", "delisting_date",
]
REQUIRED_METADATA_COLUMNS = ["sector", "listing_date", "delisting_date"]

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


class DataQualityError(ValueError):
    """A provider output violates a documented data contract."""


class DataContractError(DataQualityError):
    """Structural contract violation (index levels, columns, dtypes)."""


def provider_label(provider) -> str:
    """Human-readable provider name for contract error messages."""
    if provider is None:
        return "provider"
    cls_name = type(provider).__name__
    name = getattr(provider, "provider_name", None) or getattr(provider, "name", None)
    if name and str(name) != cls_name:
        return f"{cls_name}({name})"
    return cls_name


def _require_multi_index(df: pd.DataFrame, kind: str, label: str) -> None:
    if not isinstance(df.index, pd.MultiIndex):
        raise DataContractError(
            f"{label}: {kind} data must have a MultiIndex (date, asset); "
            f"got a {type(df.index).__name__} index"
        )
    names = list(df.index.names)
    if names != ["date", "asset"]:
        raise DataContractError(
            f"{label}: {kind} index levels must be named ['date', 'asset']; "
            f"got {names!r}"
        )
    if len(df.index):
        date_level = df.index.get_level_values("date")
        if not pd.api.types.is_datetime64_any_dtype(date_level):
            raise DataContractError(
                f"{label}: {kind} 'date' index level must be datetime64; "
                f"got dtype {date_level.dtype}"
            )


def validate_prices(df: pd.DataFrame | None, provider=None) -> None:
    """Validate a provider price frame against the shared contract.

    Raises:
        DataContractError: naming the provider and the violated
            index/column, so a provider bug cannot surface later as an
            unrelated ``KeyError``/``ValueError`` inside the pipeline.
    """
    label = provider_label(provider)
    if df is None:
        raise DataContractError(
            f"{label}: get_prices() returned None; expected a DataFrame with "
            "MultiIndex (date, asset)"
        )
    if not isinstance(df, pd.DataFrame):
        raise DataContractError(
            f"{label}: get_prices() returned {type(df).__name__}; expected "
            "a pandas DataFrame"
        )

    _require_multi_index(df, "price", label)

    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(
            f"{label}: price data is missing required column(s) {missing}; "
            f"required {PRICE_COLUMNS}, optional {OPTIONAL_PRICE_COLUMNS}"
        )

    # Only the date level must be sorted: the pipeline unstacks by asset and
    # reindexes columns, so asset order within a date is irrelevant, but
    # `pct_change`/rolling consumers walk the date order.
    dates = df.index.get_level_values("date")
    if not dates.is_monotonic_increasing:
        raise DataContractError(
            f"{label}: price data 'date' index level is not sorted "
            "(non-decreasing); return data sorted by date (df.sort_index() "
            "sorts both levels and is always safe)"
        )


def validate_financials(df: pd.DataFrame | None, provider=None) -> None:
    """Validate a provider financial frame against the shared contract.

    An empty frame (including ``None``) is accepted: a provider may
    legitimately have no fundamentals, and ``DataPipeline._align`` gives the
    empty frame the documented ``(date, asset)`` MultiIndex.
    """
    label = provider_label(provider)
    if df is None:
        return
    if not isinstance(df, pd.DataFrame):
        raise DataContractError(
            f"{label}: get_financials() returned {type(df).__name__}; "
            "expected a pandas DataFrame"
        )
    if df.empty:
        return

    _require_multi_index(df, "financial", label)

    missing = [c for c in FINANCIAL_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(
            f"{label}: financial data is missing required column(s) {missing}; "
            f"required {FINANCIAL_COLUMNS}. Fundamental factors and "
            "market-cap neutralisation read these columns directly."
        )


def validate_metadata(df: pd.DataFrame | None, provider=None) -> None:
    """Validate a provider metadata frame against the shared contract.

    ``is_st`` is intentionally optional: providers that cannot supply ST
    history omit it (or leave it empty), and the pipeline must not read that
    absence as "not ST".
    """
    label = provider_label(provider)
    if df is None or not isinstance(df, pd.DataFrame):
        raise DataContractError(
            f"{label}: get_metadata() returned "
            f"{type(df).__name__}; expected a DataFrame indexed by asset"
        )
    if df.empty:
        raise DataContractError(
            f"{label}: metadata is empty; cannot build a point-in-time "
            "universe without listing/delisting information"
        )
    if df.index.name != "asset":
        raise DataContractError(
            f"{label}: metadata index must be named 'asset'; "
            f"got index name {df.index.name!r}"
        )

    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(
            f"{label}: metadata is missing required column(s) {missing}; "
            f"required {REQUIRED_METADATA_COLUMNS} "
            "(is_st is optional when the provider has no ST history)"
        )
