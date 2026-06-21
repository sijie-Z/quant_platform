"""80d Reversal Paper Trader — 轻量级影子交易模块.

不是研究工具。是生产系统就绪验证。
跟踪 schedule, 执行模拟调仓, 记录 P&L, 输出日志。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── 策略参数 (已锁定) ──
SIGNAL_H = 40
HOLD_H = 80
SELECT_PCT = 0.20
VOL_PERCENTILE = 0.70
INITIAL_CAPITAL = 10_000_000


@dataclass
class TradeRecord:
    """一次调仓的完整记录."""
    rebalance_date: str
    action: str  # "enter" | "skip" | "exit"
    n_stocks: int = 0
    invested: float = 0.0
    cost: float = 0.0
    reason: str = ""
    exit_value: float = 0.0
    pnl: float = 0.0


@dataclass
class Position:
    """一个 active position (80d 持有期)."""
    entry_date: str
    exit_date: str
    assets: list[str]
    weights: np.ndarray
    capital: float
    cost: float


class ReversalPaperTrader:
    """80d Reversal Paper Trader.

    用法:
        trader = ReversalPaperTrader()
        trader.load_data(returns, prices)
        trader.run()  # 执行完整回放
        trader.report()  # 输出摘要
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: list[Position] = []
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[dict[str, Any]] = []
        self._returns: pd.DataFrame | None = None
        self._prices: pd.DataFrame | None = None
        self._vol_series: pd.Series | None = None
        self._vol_threshold: float | None = None

    def load_data(self, returns: pd.DataFrame, prices: pd.DataFrame):
        """加载历史数据."""
        self._returns = returns
        self._prices = prices

        # 预计算 vol filter
        market_ret = returns.mean(axis=1)
        self._vol_series = market_ret.rolling(20).std()
        self._vol_threshold = self._vol_series.quantile(VOL_PERCENTILE)

        logger.info("Data loaded: %d days, %d assets, vol threshold=%.6f",
                     len(returns), len(returns.columns), self._vol_threshold)

    def _compound(self, x):
        return np.prod(1 + x) - 1 if len(x) > 0 else 0

    def _is_high_vol(self, date: pd.Timestamp) -> bool:
        """检查 vol filter."""
        if self._vol_series is None or self._vol_threshold is None:
            return False
        return date in self._vol_series.index and self._vol_series[date] > self._vol_threshold

    def run(self):
        """执行完整回放. 按 80d schedule 模拟调仓."""
        if self._returns is None or self._prices is None:
            raise ValueError("No data loaded")

        returns = self._returns
        prices = self._prices
        dates = returns.index
        indices = list(range(HOLD_H, len(dates) - SIGNAL_H, HOLD_H))

        past_ret = returns.rolling(SIGNAL_H, min_periods=SIGNAL_H).apply(self._compound, raw=True)

        for i in indices:
            rdate = dates[i]
            date_str = str(rdate)[:10]

            # ── 检查 vol filter ──
            if self._is_high_vol(rdate):
                logger.info("[%s] HIGH VOL — skip", date_str)
                self.trades.append(TradeRecord(
                    rebalance_date=date_str, action="skip", reason="high_vol"
                ))
                # 平所有到期仓位 (但不新增)
                self._close_matured(rdate)
                self._log_equity(rdate)
                continue

            # ── 平到期仓位 ──
            self._close_matured(rdate)

            # ── 生成信号 ──
            pr = past_ret.iloc[i]
            valid = pr.dropna().sort_values()
            n_select = max(1, int(len(returns.columns) * SELECT_PCT))
            if len(valid) < n_select:
                self.trades.append(TradeRecord(
                    rebalance_date=date_str, action="skip", reason="insufficient_data"
                ))
                self._log_equity(rdate)
                continue

            selected = valid.head(n_select)
            assets = selected.index.tolist()
            n = len(assets)

            # ── 执行买入 ──
            cost_per_side = 0.0015  # 15bps total (commission + slippage)
            invest = self.cash / n if self.cash > 0 else 0
            total_cost = invest * cost_per_side * n

            if invest <= 0:
                self.trades.append(TradeRecord(
                    rebalance_date=date_str, action="skip", reason="no_cash"
                ))
                self._log_equity(rdate)
                continue

            pos = Position(
                entry_date=date_str,
                exit_date=str(dates[min(i + HOLD_H, len(dates) - 1)])[:10],
                assets=assets,
                weights=np.ones(n) / n,
                capital=invest * n,
                cost=total_cost,
            )
            self.positions.append(pos)

            self.cash -= (invest * n + total_cost)
            self.trades.append(TradeRecord(
                rebalance_date=date_str, action="enter",
                n_stocks=n, invested=invest * n, cost=total_cost,
            ))
            logger.info("[%s] ENTER: %d stocks, capital=%.0f, cost=%.0f",
                         date_str, n, invest * n, total_cost)

            self._log_equity(rdate)

        # 最后平所有剩余仓位
        self._close_all(dates[-1])
        logger.info("Run complete. Final capital: %.2f", self.cash)

    def _close_matured(self, current_date: pd.Timestamp):
        """平所有到期仓位."""
        still_active = []
        date_str = str(current_date)[:10]

        for pos in self.positions:
            if pos.exit_date <= date_str:
                # 卖出
                exit_value = pos.capital * (1 - 0.0015)  # 扣除卖出成本
                self.cash += exit_value
                self.trades.append(TradeRecord(
                    rebalance_date=date_str, action="exit",
                    n_stocks=len(pos.assets),
                    exit_value=exit_value,
                    pnl=exit_value - pos.capital - pos.cost,
                ))
            else:
                still_active.append(pos)

        self.positions = still_active

    def _close_all(self, final_date: pd.Timestamp):
        """平所有仓位."""
        for pos in self.positions:
            exit_value = pos.capital * (1 - 0.0015)
            self.cash += exit_value

        n = len(self.positions)
        self.positions = []
        logger.info("Closed all %d remaining positions", n)

    def _log_equity(self, date: pd.Timestamp):
        """记录净值."""
        pos_value = sum(p.capital for p in self.positions)
        self.equity_curve.append({
            "date": str(date)[:10],
            "cash": self.cash,
            "positions": pos_value,
            "total": self.cash + pos_value,
        })

    def report(self) -> str:
        """输出运行摘要."""
        if not self.equity_curve:
            return "No data"

        df = pd.DataFrame(self.equity_curve)
        total = df["total"].values
        rets = [total[t] / total[t - 1] - 1 for t in range(1, len(total))]

        if len(rets) < 3:
            return "Insufficient data"

        ps = pd.Series(rets)
        sharpe = ps.mean() / ps.std() * np.sqrt(252) if ps.std() > 1e-10 else 0
        ann = (1 + ps.mean()) ** 252 - 1
        cum = df["total"].values
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        mdd = dd.min()

        n_entries = sum(1 for t in self.trades if t.action == "enter")
        n_skips = sum(1 for t in self.trades if t.action == "skip")
        total_cost = sum(t.cost for t in self.trades if t.action == "enter")

        lines = [
            "=" * 60,
            "  80d Reversal Paper Trader — 运行报告",
            "=" * 60,
            f"  初始资金:       {self.initial_capital:>12,.0f}",
            f"  最终资金:       {self.cash:>12,.0f}",
            f"  总收益:         {(self.cash / self.initial_capital - 1) * 100:>10.2f}%",
            f"  年化收益:       {ann * 100:>10.2f}%",
            f"  Sharpe:         {sharpe:>12.4f}",
            f"  最大回撤:       {mdd * 100:>10.2f}%",
            f"  交易次数:       {n_entries:>12d}",
            f"  跳过次数:       {n_skips:>12d}",
            f"  总交易成本:     {total_cost:>12,.0f}",
            "-" * 60,
        ]
        return "\n".join(lines)
