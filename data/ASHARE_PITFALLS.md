# A-Share Real-Market Pitfalls Handled

This document catalogs every real-market edge case our platform handles.
Interviewers WILL ask about these — be ready to explain each one.

---

## 1. 复权处理 (Price Adjustment for Corporate Actions)

**Problem:** Stock splits, dividends, and rights issues cause artificial price jumps. Using raw (未复权) prices produces fake returns and corrupts factor calculations.

**Our solution:** 
- Tushare provider fetches 后复权 (hfq = back-adjusted) prices: raw prices are multiplied by the cumulative `adj_factor` exactly once; the pipeline divisor is set to 1.0
- Baostock uses `adjustflag="1"` (后复权) with the same "apply once" convention
- Synthetic provider generates `adj_factor` column and `close_adj` = close / adj_factor
- Pipeline computes all returns from `close_adj`, never from raw close

**Interview answer:** "后复权 scales historical prices forward by the cumulative adjustment factor, so past returns are correctly calculated and no dividend/split jump leaks into factors. The trade-off is that price levels don't match the exchange tape, but for backtesting we care about returns, not absolute price levels. All providers apply the adjustment exactly once, and the trust metadata records `adjust=hfq`."

---

## 2. 停牌处理 (Trading Suspensions)

**Problem:** A-share stocks can halt trading for days/weeks (重大事项停牌). NaN prices break factor computation and portfolio rebalancing.

**Our solution:**
- `DataPipeline._clean_prices()` derives suspension from the data itself — missing closes, zero volume, or a row whose whole OHLC(VWAP) equals the previous row (a provider forward-fill) — instead of trusting an optional `is_suspended` column. Tushare/Baostock emit no flag.
- Provider fills short gaps (synthetic ≤30 sessions via `max_suspension_days`, Tushare ≤5); long gaps stay NaN and are never bridged.
- Returns are NaN on the day before suspension through the last suspended day, so no fake zero return and no compounded "resume" move booked as a one-day return.
- Suspended asset-days are False in `tradable_mask`; factors, the alpha signal and optimizer candidate sets exclude them, so a suspended stock cannot receive a new target weight. Valuation keeps the stale mark, which equals the last clean adjusted close (the adjustment factor is frozen across the suspension).
- At rebalance dates, suspended stocks are excluded from the candidate pool; existing holdings are not topped up.

**Interview answer:** "Short suspensions we forward-fill — the stock is illiquid but still part of the portfolio. Across a suspension we never difference a forward-filled price: returns on both fill boundaries are NaN, and the name is excluded from every cross-section and new target weight while it cannot trade. Long suspensions (>30 sessions) keep their history but the gap stays missing; the name has no factor, signal or weight until it trades again."

---

## 3. 幸存者偏差 (Survivorship Bias)

**Problem:** If you only backtest on currently-listed stocks, you ignore stocks that went bankrupt or were delisted. Your backtest will be unrealistically optimistic because you only traded the winners.

**Our solution:**
- `metadata` tracks `listing_date` and `delisting_date` per stock
- `DataPipeline` builds a date-wise listing window: no factor, signal or weight exists before `listing_date`, and none exists from `delisting_date` onward. `_filter_universe()` still drops stocks that list after (or delist before) the sample.
- When a provider omits `delisting_date` but simply stops pricing, the first date of the terminal missing run (longer than `max_suspension_days`) is used as the delisting date.
- Frozen-holding behaviour: from the delisting date the name's factors and signal are NaN, so the optimizer cannot open or add to it. A position still marked between rebalances keeps its last value (its return is NaN, which the engine treats as 0 — no fabricated P&L); at the next rebalance the target weight is zero and the holding is liquidated at the last available close.
- Constituent history tracking (TushareProvider): knows which stocks were in the CSI 300 at each point in time (`index_weight`), not just today's constituents; fetches `list_status` D/P (delisted/suspended) names with `delist_date`
- BaostockProvider requests `status=0` (delisted) names and carries `outDate` as `delisting_date`, so a historical sample includes names that later delisted
- Remaining gap: the ST *history* is provider-dependent — Tushare/Baostock expose only the current name/list status, so neither supplies a true ST announcement history; the synthetic provider does. The system marks `st_available=False` rather than fabricating an all-False flag
- The universe shrinks naturally as stocks delist, matching reality

**Interview answer:** "Point-in-time universe construction. We don't use today's CSI 300 list — we fetch historical constituent changes and only trade stocks that were actually in the index on each rebalance date. Delisting is enforced date-wise: the stock leaves the factor cross-section, the alpha signal and the optimizer's candidate set on its delisting date. An existing holding is frozen at its last mark until the next rebalance, then liquidated at the last available close — no new weight, no fabricated return."

---

## 4. 涨跌停板 (Price Limits)

**Problem:** A-share stocks have ±10% daily price limits ( ±5% for ST, ±20% for ChiNext/STAR). If your backtest buys a stock at limit-up, you couldn't actually execute in reality.

**Our solution:**
- Synthetic price generator clips daily returns to [-0.10, 0.10]; ST names get ±5%
- Pipeline marks `is_limit_up` and `is_limit_down` flags
- The backtest engine consumes `tradable_mask` / `limit_up` / `limit_down` on the **execution date**: a buy blocked at limit-up (or on a suspended/ST/untradable day) is carried forward to the next tradable day, and no cost or turnover is booked for the blocked trade; sells are blocked at limit-down. This is the execution-day gate wired in Round 10 (was not present in the older vectorized engine)
- The vectorized engine still works at the weight level for lot sizes (see #8), so a blocked trade is retried, not partially filled
- ST stocks have ±5% limits (handled via narrower return clipping and the ST PIT mask)

**Interview answer:** "We flag limit-hit stocks and the backtest checks them on the execution date, not the signal date — if a stock closes limit-up when we'd want to buy, the trade is deferred to the next tradable day; sells blocked at limit-down likewise. This is conservative but realistic. Modeling partial fills would require order-book data beyond a daily-frequency backtest."

---

## 5. ST 股票 (Special Treatment)

**Problem:** ST (Special Treatment) stocks have tighter limits ( ±5%), higher delisting risk, and lower liquidity. Many institutional mandates prohibit holding ST stocks.

**Our solution:**
- PIT ST mask built from `announce_date` to `end_date` per episode; ST asset-days are False in `tradable_mask`, so they receive no factor value, signal or target weight
- `DataPipeline` has `exclude_st=True` default; the static `is_st` flag is only a fallback when a provider supplies no ST timeseries
- Providers without ST history (Tushare/Baostock) expose `st_available=False` instead of fabricating an all-False column; the synthetic provider supplies the announcement timeseries
- Config toggle: `universe.exclude_st`

**Interview answer:** "We exclude ST stocks using announcement dates, not a whole-sample flag — a stock becomes ineligible only from the date its ST announcement was public through the end of the episode. That matters because the naive version excludes stocks from day one based on an event years later, which is reverse selection and inflates backtests. With real providers that don't publish ST history, we say so explicitly rather than pretend the screen works."

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
- `PortfolioConstraints.lot_size = 100`, plus per-instrument `lot_size` from the cross-asset interface
- The execution layer (TWAP/VWAP/Iceberg algorithms, paper broker, OMS) rounds slice/order quantities down to whole lots
- The vectorized backtest engine is **weight-level** and explicitly documents that lot size is not enforced there; target weights can imply fractional lots
- For small portfolios (<10M), this causes noticeable tracking error from target weights

**Interview answer:** "Execution rounds down to the nearest 100-share lot and reallocates the residual cash, so live and paper orders are always valid. The daily vectorized backtest is weight-based — we document that it ignores lot granularity rather than claiming a constraint it doesn't enforce. For a 10M portfolio the lot constraint is binding on stocks priced above ~500 CNY (e.g., Kweichow Moutai); the impact on a diversified portfolio is small, but it's a real limitation of the weight-level engine."

---

## 9. 除权除息日与回补 (Ex-Date & Pitfalls)

**Problem:** On ex-dividend dates, the stock price mechanically drops by the dividend amount. A naive return calculation would show a large negative return, corrupting factor values on that date.

**Our solution:**
- 后复权 (hfq) adjustment bakes dividend/split adjustments into historical prices (applied exactly once by each provider)
- Synthetic provider generates random `adj_factor` changes ( ~5% of stocks per year have corporate actions)
- All factor computations use `close_adj`, not `close`

**Interview answer:** "后复权 solves this by scaling historical prices by the cumulative adjustment factor. If a stock pays a 1 CNY dividend and the price drops from 10 to 9, the factor changes so past returns are unaffected. The trade-off is that you lose absolute price-level accuracy, but for relative return analysis this is the standard approach."

---

## 10. 行业分类不一致 (Sector Classification Drift)

**Problem:** A stock's industry classification can change (e.g., a company pivots from manufacturing to tech). Using a single static sector map causes incorrect neutralization.

**Our solution:**
- `DataPipeline.get_sector_panel()` builds a date × asset sector panel from the provider's industry-change history (`effective_date`); `main.py` attaches it to metadata and `factors/processing.py` neutralizes date-by-date
- For synthetic data: industry changes are generated with `effective_date` (about 5% per half-year)
- For Tushare/Baostock: if no history is available, the current sector map is used as a documented fallback (stale, not look-ahead)
- `process_factor.neutralize()` accepts either the per-date panel or a static map

**Interview answer:** "We neutralize against the industry classification in effect on each date, not a single static map. Using the classification from before a change is stale but never look-ahead; the per-date panel removes that staleness for providers that publish industry history. When only the current classification is available, we document that limitation in the trust metadata."

---

## Summary: What to say when the interviewer asks "How real is your backtest?"

"非常接近实盘。我们处理了后复权、停牌、ST过滤、幸存者偏差、T+1制度、涨跌停板、手数限制，并且成本模型包含了印花税的单边征收和滑点。涨跌停/停牌按执行日阻断成交，未成交目标顺延到下一个可交易日；整手约束在执行层强制，向量化回测仍是权重层面（已明确标注）。唯一没有做的是高频order-book级别的流动性建模，但对于月频多因子策略来说这不是瓶颈。"
