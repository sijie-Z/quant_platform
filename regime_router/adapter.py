"""ExecutionAdapter — 研究空间到执行空间的投影层.

核心思想: 不改 BacktestEngine.
将 80d 持有期策略投影到 20d 月频框架, 通过 position overlap 保持 alpha 结构.

架构:
  - Tranche: 单期持仓记录 (开仓日期, 平仓日期, 持仓权重)
  - ExposureScheduler: 管理 tranche 生命周期
  - ExecutionAdapter: 构建重叠持仓 → 直接生成收益序列

不修改 BacktestEngine. 不通过信号值编码 overlay (引擎无法区分).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.backtest.metrics import all_metrics
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tranche — 单期持仓记录
# ---------------------------------------------------------------------------

@dataclass
class Tranche:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    weights: pd.Series          # asset → weight

    def is_matured(self, as_of_date: pd.Timestamp) -> bool:
        return as_of_date >= self.exit_date


# ---------------------------------------------------------------------------
# ExecutionAdapter — 直接构建重叠持仓收益流
# ---------------------------------------------------------------------------

class ExecutionAdapter:
    """ExecutionAdapter: 通过 position overlap 保持 80d 持有期结构.

    工作方式:
    1. 每月用原始反转信号 (S=40) 开一个新 tranche
    2. 检查到期 tranche (80d), 到期关闭
    3. 聚合所有活跃 tranche → 当日持仓
    4. 计算每日持仓收益 → 收益序列
    5. 从收益序列计算 Sharpe/MDD

    不经过 BacktestEngine, 因为引擎无法理解 position overlap.
    """

    def __init__(
        self,
        signal_h: int = 40,
        hold_h: int = 80,
        select_pct: float = 0.20,
        rebalance_freq_days: int = 20,
    ):
        self.signal_h = signal_h
        self.hold_h = hold_h
        self.select_pct = select_pct
        self.freq_days = rebalance_freq_days
        self.n_tranches = max(1, hold_h // rebalance_freq_days)

        self.metrics: dict[str, Any] = {
            "n_tranches_opened": 0,
            "n_tranches_closed": 0,
            "n_rebalance_dates": 0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "ann_return": 0.0,
            "max_drawdown": 0.0,
        }

    # ── 信号生成 ──────────────────────────────────────────────

    def _compute_past_return(self, returns: pd.DataFrame) -> pd.DataFrame:
        """过去 signal_h 日累计收益."""
        return returns.rolling(self.signal_h, min_periods=self.signal_h).apply(
            lambda x: np.prod(1 + x) - 1 if len(x) == self.signal_h else np.nan,
            raw=True,
        )

    def _build_tranche_weights(self, past_ret_row: pd.Series) -> pd.Series:
        """从反转信号构建 tranche 权重: 选跌幅最大 select_pct 的股票, 等权."""
        valid = past_ret_row.dropna().sort_values()
        n_select = max(1, int(len(valid) * self.select_pct))
        selected = valid.head(n_select)  # 跌幅最大的
        weights = pd.Series(1.0 / n_select, index=selected.index)
        return weights

    # ── 核心: 构建收益流 ──────────────────────────────────────

    def run(self, returns: pd.DataFrame, prices: pd.DataFrame,
            rebalance_dates: list[pd.Timestamp] | None = None) -> pd.Series:
        """运行 adapter, 返回每日收益率序列.

        Args:
            returns: 日收益率 DataFrame (date × asset)
            prices: 日收盘价 DataFrame (date × asset)
            rebalance_dates: 调仓日期列表. None=每月末.

        Returns:
            pd.Series: 策略的日收益率 (index=dates)
        """
        if rebalance_dates is None:
            ed = returns.index.to_series()
            rebalance_dates = ed.groupby([ed.dt.year, ed.dt.month]).last().tolist()

        past_ret = self._compute_past_return(returns)
        scheduler = _ExposureScheduler(n_tranches=self.n_tranches)
        all_dates = returns.index

        # 预计算所有调仓日的权重
        daily_weights: dict[pd.Timestamp, pd.Series] = {}

        for rdate in rebalance_dates:
            if rdate not in past_ret.index:
                continue

            # 1. 平到期 tranche
            scheduler.close_matured(rdate)

            # 2. 开新 tranche
            pr_row = past_ret.loc[rdate]
            weights = self._build_tranche_weights(pr_row)
            exit_date = rdate + pd.Timedelta(days=self.hold_h)
            scheduler.open_tranche(Tranche(entry_date=rdate, exit_date=exit_date, weights=weights))

            # 3. 聚合权重 -> 记录到下一个调仓日
            next_rdate = None
            for dr in rebalance_dates:
                if dr > rdate:
                    next_rdate = dr
                    break

            agg = scheduler.aggregate_weights()
            if agg is not None and len(agg) > 0:
                # 从当天持有到下一次调仓
                end_date = next_rdate if next_rdate else all_dates[-1]
                current = rdate
                while current <= end_date and current <= all_dates[-1]:
                    if current in all_dates:
                        daily_weights[current] = agg
                    current += pd.Timedelta(days=1)

        self.metrics["n_tranches_opened"] = scheduler.n_opened
        self.metrics["n_tranches_closed"] = scheduler.n_closed
        self.metrics["n_rebalance_dates"] = len(rebalance_dates)

        # ── 计算每日收益 ──────────────────────────────────────
        port_returns = pd.Series(0.0, index=all_dates)
        # 找第一个有持仓的日期
        valid_dates = [d for d in all_dates if d in daily_weights and len(daily_weights[d]) > 0]
        if not valid_dates:
            logger.warning("Adapter: no valid positions!")
            return port_returns

        for i, date in enumerate(valid_dates):
            if i == 0:
                continue
            prev_date = valid_dates[i - 1]
            weights = daily_weights[prev_date]  # 使用前一天的持仓计算当日收益

            # 当日收益 = sum(weight * asset_return)
            day_ret = returns.loc[date]
            valid_assets = [a for a in weights.index if a in day_ret.index and pd.notna(day_ret.get(a))]
            if not valid_assets:
                continue

            # 归一化权重到有效资产
            w = weights[valid_assets].copy()
            w = w / w.sum()

            port_returns[date] = (w * day_ret[valid_assets]).sum()

        # ── 计算指标 ──────────────────────────────────────────
        non_zero = port_returns[port_returns != 0]
        if len(non_zero) > 10:
            metrics = all_metrics(port_returns, None)
            self.metrics["sharpe"] = metrics.get("sharpe_ratio", 0)
            self.metrics["ann_return"] = metrics.get("annual_return", 0)
            self.metrics["max_drawdown"] = metrics.get("max_drawdown", 0)
            self.metrics["total_return"] = float((1 + port_returns).prod() - 1)

        n_active = len(valid_dates)
        logger.info("Adapter: %d tranches, %d/%d trading days with positions, Sharpe=%.4f",
                     scheduler.n_opened, n_active, len(all_dates), self.metrics["sharpe"])

        return port_returns


# ---------------------------------------------------------------------------
# 内部: 持仓调度器
# ---------------------------------------------------------------------------

class _ExposureScheduler:
    """管理 tranche 生命周期."""

    def __init__(self, n_tranches: int = 4):
        self.n_tranches = n_tranches
        self._tranches: list[Tranche] = []
        self.n_opened = 0
        self.n_closed = 0

    @property
    def active(self) -> list[Tranche]:
        return self._tranches

    def open_tranche(self, tranche: Tranche):
        self._tranches.append(tranche)
        self.n_opened += 1

    def close_matured(self, as_of_date: pd.Timestamp):
        matured = [t for t in self._tranches if t.is_matured(as_of_date)]
        self._tranches = [t for t in self._tranches if not t.is_matured(as_of_date)]
        self.n_closed += len(matured)

    def aggregate_weights(self) -> pd.Series | None:
        if not self._tranches:
            return None
        w_per = 1.0 / self.n_tranches
        total = pd.Series(dtype=float)
        for t in self._tranches:
            total = total.add(t.weights * w_per, fill_value=0)
        return total / total.sum()
