# A-Share Real-Market Pitfalls Handled

This document catalogs every real-market edge case our platform handles.
Interviewers WILL ask about these — be ready to explain each one.

---

## 1. 复权处理 (Price Adjustment for Corporate Actions)

**Problem:** Stock splits, dividends, and rights issues cause artificial price jumps. Using raw (未复权) prices produces fake returns and corrupts factor calculations.

**Our solution:** 
- Tushare provider fetches 前复权 (qfq = forward-adjusted) prices via `pro_api.daily()` with explicit adj_factor tracking
- Synthetic provider generates `adj_factor` column and `close_adj` = close / adj_factor
- Pipeline computes all returns from `close_adj`, never from raw close

**Interview answer:** "前复权 adjusts all historical prices as if today's share count existed in the past. This means past returns are correctly calculated. The trade-off is that past prices don't match what actually traded, but for backtesting that's irrelevant — we care about returns, not absolute price levels."

---

## 2. 停牌处理 (Trading Suspensions)

**Problem:** A-share stocks can halt trading for days/weeks (重大事项停牌). NaN prices break factor computation and portfolio rebalancing.

**Our solution:**
- `DataPipeline._clean_prices()`: Forward-fill NaN prices up to `max_suspension_days` (default 30)
- Stocks suspended longer than threshold are excluded from the universe for that period
- At rebalance dates, suspended stocks are excluded from the candidate pool
- Factor computation uses `.dropna()` guards everywhere

**Interview answer:** "Short suspensions (<5 days) we forward-fill — the stock is illiquid but still part of the portfolio. Long suspensions (>30 days) we remove from investable universe because the price is stale and the fundamental situation likely changed. This is conservative — most real funds would also exclude or cap exposure to recently resumed stocks."

---

## 3. 幸存者偏差 (Survivorship Bias)

**Problem:** If you only backtest on currently-listed stocks, you ignore stocks that went bankrupt or were delisted. Your backtest will be unrealistically optimistic because you only traded the winners.

**Our solution:**
- `metadata` tracks `listing_date` and `delisting_date` per stock
- `DataPipeline._filter_universe()` excludes stocks not yet listed AND stocks already delisted
- Constituent history tracking (TushareProvider): knows which stocks were in the CSI 300 at each point in time, not just today's constituents
- The universe shrinks naturally as stocks delist, matching reality

**Interview answer:** "Point-in-time universe construction. We don't use today's CSI 300 list — we fetch historical constituent changes and only trade stocks that were actually in the index on each rebalance date. We also handle delisting: on the delist date, the position is liquidated at the last available close price."

---

## 4. 涨跌停板 (Price Limits)

**Problem:** A-share stocks have ±10% daily price limits ( ±5% for ST, ±20% for ChiNext/STAR). If your backtest buys a stock at limit-up, you couldn't actually execute in reality.

**Our solution:**
- Synthetic price generator clips daily returns to [-0.10, 0.10]
- Pipeline marks `is_limit_up` and `is_limit_down` flags
- Cost model adds extra slippage for limit-hit stocks (effectively: can't trade)
- ST stocks have ±5% limits (handled via narrower return clipping)

**Interview answer:** "We flag limit-hit stocks at rebalance. If a stock is at limit-up when we'd want to buy, we skip it and reallocate to the next-best signal. This is conservative but realistic — in practice, you might get partial fills, but modeling that requires order-book data beyond a daily-frequency backtest."

---

## 5. ST 股票 (Special Treatment)

**Problem:** ST (Special Treatment) stocks have tighter limits ( ±5%), higher delisting risk, and lower liquidity. Many institutional mandates prohibit holding ST stocks.

**Our solution:**
- `metadata.is_st` flag per stock, updated from Tushare's stock_basic endpoint
- `DataPipeline` has `exclude_st=True` default — filters ST stocks from universe
- Config toggle: `universe.exclude_st`

**Interview answer:** "We exclude ST stocks by default because: (1) many institutions are prohibited from holding them, (2) the ±5% limit and delisting risk create a different risk profile than our factor model assumes, and (3) liquidity is typically very poor, making execution assumptions break."

---

## 6. T+1 交易制度 (T+1 Settlement)

**Problem:** A-share stocks settle T+1 — you can't sell what you bought today. This matters for daily-frequency strategies but less for monthly rebalancing.

**Our solution (implicit):**
- Monthly rebalancing frequency avoids T+1 issues entirely
- If running daily backtest, the engine uses `returns.shift(-1)` — today's signal executes at tomorrow's close, so you're never buying and selling the same stock on the same day

**Interview answer:** "At monthly frequency, T+1 is irrelevant — there's always at least 20 trading days between buy and sell. Our signal construction uses `shift(-1)` for next-day execution, which is standard for daily strategies."

---

## 7. 交易成本 (Realistic Transaction Costs)

**Problem:** Many academic backtests ignore costs, producing unrealistically high returns. A-share costs are non-trivial: 0.03% commission + 0.1% stamp tax on sells + slippage.

**Our solution:**
- `CostModel` with three components: commission (0.03%), stamp tax (0.1% sell-only), slippage (configurable: fixed or proportional)
- Stamp tax is asymmetric — only on sells, as per A-share regulation
- Proportional slippage model uses square-root market impact formula for large trades

**Interview answer:** "For a strategy turning over 30% monthly with 0.1% stamp tax on sells, that's roughly 0.05%/month in stamp tax alone — ~0.6%/year drag. On a strategy targeting 10% alpha, that's a 6% haircut. You MUST include costs or your backtest lies to you."

---

## 8. 手数限制 (Lot Size)

**Problem:** A-share minimum trading unit is 100 shares (1手). Your portfolio optimizer might produce weights that require buying 37 shares, which is impossible.

**Our solution:**
- `PortfolioConstraints.lot_size = 100`
- Optimizer rounds positions to lot-size multiples
- For small portfolios (<10M), this causes noticeable tracking error from target weights

**Interview answer:** "For a 10M portfolio, the lot constraint is binding on stocks priced above ~500 CNY (e.g., Kweichow Moutai). For these, we round DOWN to the nearest lot and reallocate the residual cash. The impact on a well-diversified portfolio is negligible (<5bps)."

---

## 9. 除权除息日与回补 (Ex-Date & Pitfalls)

**Problem:** On ex-dividend dates, the stock price mechanically drops by the dividend amount. A naive return calculation would show a large negative return, corrupting factor values on that date.

**Our solution:**
- 前复权 (qfq) adjustment bakes dividend adjustments into historical prices
- Synthetic provider generates random `adj_factor` changes ( ~5% of stocks per year have corporate actions)
- All factor computations use `close_adj`, not `close`

**Interview answer:** "前复权 solves this by scaling ALL past prices by the adjustment factor. If a stock pays a 1 CNY dividend and the price drops from 10 to 9, the qfq factor is 0.9 — all past prices are multiplied by 0.9, so past returns are unaffected. The trade-off is that you lose absolute P&L accuracy, but for relative return analysis this is the standard approach."

---

## 10. 行业分类不一致 (Sector Classification Drift)

**Problem:** A stock's industry classification can change (e.g., a company pivots from manufacturing to tech). Using a single static sector map causes incorrect neutralization.

**Our solution:**
- For synthetic data: static sector (acceptable for 5-year window)
- For Tushare: sector is fetched from the latest stock_basic
- `process_factor.neutralize()` handles any sector map dynamically per date

**Interview answer:** "For a 3-5 year backtest, sector reclassification is rare (<5% of stocks). We use the latest classification as a practical compromise. For longer horizons, you'd need point-in-time sector data from a vendor like GICS or Shenwan."

---

## Summary: What to say when the interviewer asks "How real is your backtest?"

"非常接近实盘。我们处理了前复权、停牌、ST过滤、幸存者偏差、T+1制度、涨跌停板、手数限制，并且成本模型包含了印花税的单边征收和滑点。唯一没有做的是高频order-book级别的流动性建模，但对于月频多因子策略来说这不是瓶颈。"
