# Milestone 1 — 自我复盘报告

**Tag**: `v0.1-research-os`
**Generated**: 2026-07-11
**Source**: Registry SQL + git log + disk artifacts — not hand-edited.

## 已验证成立的设计决策

1. **JSON trust metadata 在 Registry 中足够应对 3 次运行** — 每个跨 Factor 的问题（哪个 ICIR 最高？谁用 hfq？pit 状态？）都用单条 SQL 回答，无需打开任何 Run 脚本。
2. **报告生成器逐字读取 Registry 的 trust 字段** — WARNING 由 `pit=false` + `adjust=none` 自动生成，零硬编码。去重后重新生成报告能完全验证这一点。
3. **"单个 Run，不抽 Runner"的纪律成立** — 3 个 Factor = 3 个脚本，没有共享抽象。加第四个 Factor 只需新增 1 个文件，和第三个完全一样。
4. **framework/contracts 维持在 6 个 Protocol** — M1 执行路径不需要新增任何一个。Broker/LLM 仍然是 stub 且没有消费者，这反而印证了 ADR-0004 "Protocol before Plugin，不是 Protocol before Consumer"。
5. **零存量代码修改** — trust layer 是和现有系统共存建出来的，没有替换 4 万行旧回测。

## 被证伪的假设

1. **"AkShare 免费数据足够快来做 prototype"** → ❌ 伪。TX source `stock_zh_a_hist_tx` 对 300 × 5 年需要 15-25 分钟。数据拉取佔了切片墙上时钟的 90%。应把 `fetch_duration` 写入 Registry evaluation。
2. **"Momentum 在 CSI300 的近期 A 股有可辨识信号"** → ❌ 伪。IC=0.01，基本是噪声。平台正确地暴露了这一事实，但假设本身是错的。
3. **"ROE 从季度财务指标 API 能提供可用的横截面排序"** → ❌ 伪。IC=0.0017 统计上不可与零区分。季度 carry-forward ROE 在 CSI300 内没有区分度。
4. **"v0.1 需要 6 个 Protocol"** → ❌ 伪。M1 实际只用了 UniverseProvider + MarketDataProvider + Evaluator。FactorProtocol 没被用过（直接用了 FactorRegistry），Broker/LLM 都没有消费者。6 个里有 3 个未使用。

## 不该做的事

1. **设计 Factor Registry table** — 现有的 `lab/registry/__init__.py RunStore` 已经够用。M1 不需要单独的 factor_registry 表或 FactorProtocol。
2. **中途中途试图加 `has_report` / `fetch_duration` 列** — Edit 失败并暴露了一个事实：在执行过程中扩展 Registry schema 是过早的冲动。
3. **花多轮调试 TX/EM 接口切换** — 应该在写 300 股票切片之前先用一只股票测 `stock_zh_a_hist_tx`，而不是经过 4 次失败尝试才认识到。

## 对 M2 的建议

1. **Factor Diagnostics 优先于新增因子** — 对现有 3 个因子先做逐年 IC、滚动 IC、覆盖率分析，然后再加第 4 个。
2. **把 `fetch_duration` 写入 evaluation JSON** — 它是一个一级 trust metric（这个真理有多贵？）。
3. **加一个轻量级的 "runs" CLI 或单条命令来跑新因子** — 但要等到现有 3 个因子的诊断完成之后，而不能提前。
4. **如果 M2 中期仍没有消费者，考虑把未使用的 Protocol（Broker, LLM）缩减掉**。

---

*机器从 Registry + git log + 磁盘产物自动生成，非手工编辑。*
