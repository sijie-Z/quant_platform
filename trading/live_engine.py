"""Live Engine — alpha-v1.0 运行时系统.

把 batch backtest 改成 event-driven 持续运行程序.

架构:
  while market_is_open:
    data = get_data()
    state.update(data)
    signal = alpha(state)
    position = risk(signal)
    broker.send_orders(position)
    sleep(interval)

当前: paper mode (mock data + simulated execution)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── 策略参数 (alpha-v1.0, 已锁定) ──
SIGNAL_H = 40
HOLD_H = 80
SELECT_PCT = 0.20
VOL_PERCENTILE = 0.70
INITIAL_CAPITAL = 10_000_000
COST_BPS = 15
MAX_DRAWDOWN = 0.30


@dataclass
class Position:
    asset: str
    entry_date: str
    exit_date: str
    shares: int
    entry_price: float


@dataclass
class State:
    """系统运行时状态."""
    cash: float = INITIAL_CAPITAL
    positions: list[Position] = field(default_factory=list)
    equity_peak: float = INITIAL_CAPITAL
    current_equity: float = INITIAL_CAPITAL
    total_trades: int = 0
    last_rebalance: str = ""
    next_rebalance: str = ""
    vol_series: list[float] = field(default_factory=list)
    status: str = "init"


class LiveEngine:
    """alpha-v1.0 Live Runtime.

    Usage:
        engine = LiveEngine(data_provider, broker)
        engine.run()
    """

    def __init__(self, data_provider=None, broker=None):
        self.data_provider = data_provider
        self.broker = broker
        self.state = State()
        self._history: pd.DataFrame | None = None
        self._day_count = 0
        self._rebalance_dates: list[pd.Timestamp] = []

    # ── 初始化 ──

    def load_history(self, returns: pd.DataFrame, prices: pd.DataFrame):
        """加载历史数据用于状态初始化."""
        self._returns = returns
        self._prices = prices
        # 预计算调仓日
        dates = returns.index
        self._rebalance_dates = [dates[i] for i in range(HOLD_H, len(dates) - SIGNAL_H, HOLD_H)]
        self.state.next_rebalance = str(self._rebalance_dates[0])[:10] if self._rebalance_dates else ""
        logger.info("History loaded: %d days, %d rebalance dates", len(returns), len(self._rebalance_dates))

    # ── 核心循环 ──

    def run_once(self, date: pd.Timestamp) -> dict[str, Any]:
        """单日运行 (每次调用 = 一个交易日).

        这是 live engine 的核心——每次市场收盘后调用一次.
        """
        if self._returns is None or self._prices is None:
            return {"status": "no_data"}

        date_str = str(date)[:10]
        self._day_count += 1
        result: dict[str, Any] = {"date": date_str, "action": "none"}

        # 1. 更新权益
        self._update_equity(date)

        # 2. 检查是否需要调仓
        is_rebalance = date in self._rebalance_dates

        if is_rebalance:
            result["action"] = self._execute_rebalance(date)
            self.state.last_rebalance = date_str

            # 更新下次调仓日
            idx = self._rebalance_dates.index(date) if date in self._rebalance_dates else -1
            if idx >= 0 and idx + 1 < len(self._rebalance_dates):
                self.state.next_rebalance = str(self._rebalance_dates[idx + 1])[:10]

        # 3. 检查 kill switch
        drawdown = (self.state.equity_peak - self.state.current_equity) / self.state.equity_peak
        if drawdown > MAX_DRAWDOWN:
            self._emergency_flat(date)
            result["action"] = "kill_switch"
            result["reason"] = f"drawdown={drawdown:.2%}"

        result["equity"] = self.state.current_equity
        result["cash"] = self.state.cash
        result["drawdown"] = drawdown
        result["n_positions"] = len(self.state.positions)

        return result

    def run(self, start: str | None = None, end: str | None = None):
        """跑完整回放 (batch 模式, 模拟持续运行)."""
        if self._returns is None:
            raise ValueError("No data")

        dates = self._returns.index
        if start:
            dates = dates[dates >= start]
        if end:
            dates = dates[dates <= end]

        logger.info("LiveEngine starting: %s -> %s (%d days)",
                     str(dates[0])[:10], str(dates[-1])[:10], len(dates))

        results = []
        for date in dates:
            r = self.run_once(date)
            if r["action"] != "none":
                logger.info("[%s] %s%s", r["date"], r["action"],
                           f" ({r.get('reason','')})" if r.get("reason") else "")
            results.append(r)

        final = results[-1]
        logger.info("LiveEngine done. Equity: %.2f, Trades: %d, Status: %s",
                     final["equity"], self.state.total_trades, self.state.status)
        return results

    # ── 内部方法 ──

    def _update_equity(self, date: pd.Timestamp):
        """计算当日总权益 (逐日盯市)."""
        pos_value = 0.0
        if self._prices is not None and date in self._prices.index:
            px = self._prices.loc[date]
            for pos in self.state.positions:
                if pos.asset in px.index and not pd.isna(px[pos.asset]):
                    pos_value += pos.shares * px[pos.asset]

        self.state.current_equity = self.state.cash + pos_value
        if self.state.current_equity > self.state.equity_peak:
            self.state.equity_peak = self.state.current_equity

    def _execute_rebalance(self, date: pd.Timestamp) -> str:
        """执行调仓."""
        if self._returns is None or self._prices is None:
            return "error"

        day_idx = self._returns.index.get_loc(date)
        n_stocks = len(self._returns.columns)
        cost = COST_BPS / 10000

        # ── 平到期仓位 ──
        date_str = str(date)[:10]
        self.state.positions = [p for p in self.state.positions if p.exit_date > date_str]

        # ── Vol filter ──
        market_vol = self._returns.mean(axis=1).rolling(20).std()
        vol_thresh = market_vol.quantile(VOL_PERCENTILE)
        if date in market_vol.index and market_vol[date] > vol_thresh:
            return "skip_high_vol"

        # ── 信号 ──
        past_ret = self._returns.rolling(SIGNAL_H, min_periods=SIGNAL_H).apply(
            lambda x: np.prod(1 + x) - 1 if len(x) > 0 else 0, raw=True
        )
        pr = past_ret.iloc[day_idx]
        valid = pr.dropna().sort_values()
        n_select = max(1, int(n_stocks * SELECT_PCT))

        if len(valid) < n_select:
            return "skip_insufficient_data"

        selected = valid.head(n_select)
        n = len(selected)
        cap_per = self.state.cash / n

        # ── 买入 ──
        price_row = self._prices.loc[date] if date in self._prices.index else None
        if price_row is None:
            return "skip_no_price"

        total_spent = 0.0
        for asset in selected.index:
            ep = price_row[asset] if asset in price_row.index else 0
            if pd.isna(ep) or ep <= 0:
                continue
            shares = max(100, int(cap_per / ep / 100) * 100)
            spent = shares * ep
            if total_spent + spent > self.state.cash:
                break  # 现金不够

            exit_date = self._returns.index[min(day_idx + HOLD_H, len(self._returns.index) - 1)]
            self.state.positions.append(Position(
                asset=asset,
                entry_date=str(date)[:10],
                exit_date=str(exit_date)[:10],
                shares=shares,
                entry_price=ep,
            ))
            total_spent += spent

        self.state.cash -= total_spent  # 买入不扣成本, 卖出时扣
        if total_spent > 0:
            self.state.total_trades += 1

        self.state.total_trades += 1
        return f"enter_{n}"

    def _emergency_flat(self, date: pd.Timestamp):
        """紧急平仓 (kill switch)."""
        cost = COST_BPS / 10000
        price_row = self._prices.loc[date] if date in self._prices.index else None

        for pos in self.state.positions:
            if price_row is not None and pos.asset in price_row.index:
                px = price_row[pos.asset]
                if not pd.isna(px) and px > 0:
                    self.state.cash += pos.shares * px * (1 - cost)

        self.state.positions = []
        self.state.status = "killed"
        logger.warning("[%s] KILL SWITCH triggered — all positions closed", str(date)[:10])

    def status_report(self) -> str:
        """生成状态报告."""
        dd = (self.state.equity_peak - self.state.current_equity) / self.state.equity_peak if self.state.equity_peak > 0 else 0
        return (
            f"\n{'=' * 60}\n"
            f"  LiveEngine — alpha-v1.0\n"
            f"{'=' * 60}\n"
            f"  Status:      {self.state.status}\n"
            f"  Equity:      {self.state.current_equity:>12,.2f}\n"
            f"  Cash:        {self.state.cash:>12,.2f}\n"
            f"  Positions:   {len(self.state.positions):>6d}\n"
            f"  Drawdown:    {dd*100:>10.2f}%\n"
            f"  Trades:      {self.state.total_trades:>6d}\n"
            f"  Last rebal:  {self.state.last_rebalance}\n"
            f"  Next rebal:  {self.state.next_rebalance}\n"
            f"{'=' * 60}"
        )
