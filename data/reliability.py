"""Data Pipeline Reliability — 数据源可靠性层.

功能:
  - 多数据源自动切换 (主 → 备 → 回退)
  - 延迟监控 (数据是否停留在 N 天前)
  - 一致性校验 (日频 return vs price 是否匹配)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    name: str
    connect_fn: Callable
    priority: int = 0
    is_active: bool = True
    last_error: str = ""
    fail_count: int = 0
    max_fails: int = 3


class DataReliability:
    """数据源可靠性管理器."""

    def __init__(self):
        self.sources: list[DataSource] = []
        self.active_source: DataSource | None = None

    def add_source(self, name: str, connect_fn: Callable, priority: int = 0):
        self.sources.append(DataSource(name=name, connect_fn=connect_fn, priority=priority))
        self.sources.sort(key=lambda s: s.priority)

    def connect(self) -> Any | None:
        """从最高优先级开始依次尝试连接."""
        for src in self.sources:
            if src.fail_count >= src.max_fails:
                src.is_active = False
                logger.warning("Source %s disabled after %d failures", src.name, src.fail_count)
                continue
            try:
                result = src.connect_fn()
                src.is_active = True
                src.fail_count = 0
                self.active_source = src
                logger.info("Connected to %s", src.name)
                return result
            except Exception as e:
                src.fail_count += 1
                src.last_error = str(e)
                logger.warning("Failed to connect %s (fail %d/%d): %s",
                               src.name, src.fail_count, src.max_fails, e)
        logger.error("All data sources failed")
        return None

    def check_latency(self, data: pd.DataFrame, max_lag_days: int = 3) -> bool:
        """检查数据是否过时."""
        if data is None or len(data) == 0:
            return False
        last_date = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else None
        if last_date is None:
            return False
        lag = (datetime.now() - last_date).days
        if lag > max_lag_days:
            logger.warning("Data lag: %d days (limit %d)", lag, max_lag_days)
            return False
        return True

    @staticmethod
    def check_consistency(returns: pd.DataFrame, prices: pd.DataFrame, tol: float = 1e-4) -> bool:
        """检查 return 和 price 数据是否一致.

        return[t] ≈ price[t+1] / price[t] - 1
        """
        common_dates = sorted(set(returns.index) & set(prices.index))
        if len(common_dates) < 2:
            return False
        date = common_dates[-2]
        next_date = common_dates[-1]
        r = returns.loc[date].mean()
        p = prices.loc[next_date].mean() / prices.loc[date].mean() - 1
        consistent = abs(r - p) < tol
        if not consistent:
            logger.warning("Data inconsistency: return=%.6f price_change=%.6f", r, p)
        return consistent
