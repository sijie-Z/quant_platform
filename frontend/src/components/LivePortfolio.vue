<template>
  <div class="live-portfolio">
    <!-- Header -->
    <div class="lp-header">
      <div class="lp-title">
        <span class="lp-dot alive"></span>
        {{ locale === 'zh-CN' ? '实时组合追踪' : 'LIVE PORTFOLIO TRACKER' }}
        <span v-if="sourceLabel" class="lp-chip">{{ sourceLabel }}</span>
        <span v-if="priceBasis" class="lp-chip lp-chip-basis" :title="basisTitle">{{ priceBasis }} CNY</span>
      </div>
      <div class="lp-actions">
        <span v-if="portfolio?.data_date" class="lp-price-date" :class="{ stale: portfolio.stale }">
          {{ locale === 'zh-CN' ? `行情日期 ${portfolio.data_date}` : `Prices ${portfolio.data_date}` }}
          <template v-if="portfolio.stale">{{ locale === 'zh-CN' ? '（非当日）' : ' (stale)' }}</template>
        </span>
        <button class="btn btn-sm btn-secondary" @click="$refs.fileInput.click()" :disabled="loading || loadingInitial || refreshing">
          {{ locale === 'zh-CN' ? '导入CSV' : 'Import CSV' }}
        </button>
        <button class="btn btn-sm btn-primary" @click="refreshPrices()" :disabled="refreshing || loadingInitial || loading">
          {{ refreshing ? (locale === 'zh-CN' ? '加载中...' : 'Loading...') : (locale === 'zh-CN' ? '刷新行情' : 'Refresh Prices') }}
        </button>
        <input ref="fileInput" type="file" accept=".csv" style="display:none" @change="onFileUpload" />
      </div>
    </div>

    <!-- Partial-book banner (server deadline hit) -->
    <div v-if="portfolio?.partial" class="lp-banner lp-banner-warn">
      {{ locale === 'zh-CN'
        ? `部分行情：服务端截止时间前仅取到 ${portfolio.n_with_price}/${portfolio.n_positions + (portfolio.n_no_price || 0)} 只持仓的价格`
        : `Partial prices: the server deadline hit before all quotes arrived (${portfolio.n_with_price}/${portfolio.n_positions + (portfolio.n_no_price || 0)} holdings priced)` }}
    </div>

    <!-- Summary Cards -->
    <div class="lp-summary" v-if="portfolio">
      <div class="lp-card">
        <div class="lp-card-label">{{ locale === 'zh-CN' ? '总价值' : 'Total Value' }}</div>
        <div class="lp-card-value lp-accent">{{ formatNumber(portfolio.total_value) }}</div>
      </div>
      <div class="lp-card">
        <div class="lp-card-label">{{ locale === 'zh-CN' ? '日盈亏' : 'Daily P&L' }}</div>
        <div :class="['lp-card-value', portfolio.total_pnl >= 0 ? 'lp-pos' : 'lp-neg']">
          {{ portfolio.total_pnl >= 0 ? '+' : '' }}{{ formatNumber(portfolio.total_pnl) }}
        </div>
      </div>
      <div class="lp-card">
        <div class="lp-card-label">{{ locale === 'zh-CN' ? '日收益率' : 'Daily Return' }}</div>
        <div :class="['lp-card-value', portfolio.daily_return_pct >= 0 ? 'lp-pos' : 'lp-neg']">
          {{ portfolio.daily_return_pct >= 0 ? '+' : '' }}{{ portfolio.daily_return_pct?.toFixed(2) }}%
        </div>
      </div>
      <div class="lp-card">
        <div class="lp-card-label">{{ locale === 'zh-CN' ? '持仓数' : 'Positions' }}</div>
        <div class="lp-card-value">{{ portfolio.n_positions }}</div>
      </div>
      <div class="lp-card">
        <div class="lp-card-label">{{ locale === 'zh-CN' ? '有行情' : 'With Price' }}</div>
        <div class="lp-card-value lp-green">{{ portfolio.n_with_price }}</div>
      </div>
      <div class="lp-card" v-if="portfolio.n_no_price > 0">
        <div class="lp-card-label">{{ locale === 'zh-CN' ? '无行情' : 'No Price' }}</div>
        <div class="lp-card-value lp-warn">{{ portfolio.n_no_price }}</div>
      </div>
    </div>

    <!-- Holdings without prices yet (server holdings exist, feed unavailable) -->
    <template v-if="!portfolio && hasHoldings">
      <div class="lp-banner lp-banner-info">
        <span>
          {{ locale === 'zh-CN'
            ? `已从${sourceLabel || '服务器'}加载 ${holdingsOnly.length} 条持仓，但还没有行情。点击“刷新行情”重试。`
            : `${holdingsOnly.length} holdings loaded from ${sourceLabel || 'the server'}, but no prices yet. Click "Refresh Prices" to retry.` }}
        </span>
        <span v-if="priceError" class="lp-banner-error">{{ priceError }}</span>
      </div>
      <div class="lp-table-wrap">
        <table class="lp-tbl">
          <thead>
            <tr>
              <th>#</th>
              <th>{{ locale === 'zh-CN' ? '代码' : 'Code' }}</th>
              <th>{{ locale === 'zh-CN' ? '持仓' : 'Shares' }}</th>
              <th>{{ locale === 'zh-CN' ? '价格' : 'Price' }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(h, i) in holdingsOnly" :key="h.code" class="lp-row">
              <td class="lp-rank">{{ i + 1 }}</td>
              <td class="lp-code">{{ h.code }}</td>
              <td class="lp-shares">{{ formatShares(h.hold_vol) }}</td>
              <td class="lp-price lp-dim">--</td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>

    <!-- Holdings Table -->
    <div class="lp-table-wrap" v-if="portfolio?.holdings?.length">
      <table class="lp-tbl">
        <thead>
          <tr>
            <th>#</th>
            <th>{{ locale === 'zh-CN' ? '代码' : 'Code' }}</th>
            <th>{{ locale === 'zh-CN' ? '持仓' : 'Shares' }}</th>
            <th>{{ locale === 'zh-CN' ? '收盘价' : 'Close' }}</th>
            <th>{{ locale === 'zh-CN' ? '昨收' : 'Prev Close' }}</th>
            <th>{{ locale === 'zh-CN' ? '市值' : 'Mkt Value' }}</th>
            <th>{{ locale === 'zh-CN' ? '盈亏' : 'P&L' }}</th>
            <th>{{ locale === 'zh-CN' ? '收益率' : 'Return' }}</th>
            <th>{{ locale === 'zh-CN' ? '权重' : 'Weight' }}</th>
            <th>{{ locale === 'zh-CN' ? '价格基准' : 'Basis' }}</th>
            <th>{{ locale === 'zh-CN' ? '盈亏条' : 'P&L Bar' }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(h, i) in portfolio.holdings" :key="h.code" class="lp-row">
            <td class="lp-rank">{{ i + 1 }}</td>
            <td class="lp-code">{{ normalizeSymbol(h.code) }}</td>
            <td class="lp-shares">{{ formatShares(h.hold_vol) }}</td>
            <td class="lp-price">{{ h.close?.toFixed(2) }}</td>
            <td class="lp-price lp-dim">{{ h.preclose?.toFixed(2) }}</td>
            <td class="lp-value">{{ formatNumber(h.market_value) }}</td>
            <td :class="['lp-pnl', h.pnl >= 0 ? 'lp-pos' : 'lp-neg']">
              {{ h.pnl >= 0 ? '+' : '' }}{{ formatNumber(h.pnl) }}
            </td>
            <td :class="['lp-pnl', h.pnl_pct >= 0 ? 'lp-pos' : 'lp-neg']">
              {{ h.pnl_pct >= 0 ? '+' : '' }}{{ h.pnl_pct?.toFixed(2) }}%
            </td>
            <td class="lp-weight">{{ getWeight(h.market_value) }}%</td>
            <td class="lp-basis">{{ (h.price_basis || portfolio.price_basis || '—').toString().toUpperCase() }}</td>
            <td class="lp-bar-cell">
              <div class="lp-bar-wrap">
                <div
                  class="lp-bar-fill"
                  :class="h.pnl >= 0 ? 'lp-bar-pos' : 'lp-bar-neg'"
                  :style="{ width: getBarWidth(h.pnl_pct) + '%' }"
                ></div>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Empty state: either a genuine empty book or a read failure -->
    <div v-if="!loadingInitial && !hasHoldings && !loading" class="lp-empty">
      <div class="lp-empty-icon">&#9776;</div>
      <template v-if="holdingsError">
        <h3>{{ locale === 'zh-CN' ? '无法读取服务器持仓' : 'Could not load server holdings' }}</h3>
        <p class="lp-empty-error">{{ holdingsError }}</p>
        <button class="btn btn-secondary" @click="loadExisting" :disabled="loadingInitial">
          {{ locale === 'zh-CN' ? '重试' : 'Retry' }}
        </button>
      </template>
      <template v-else>
        <h3>{{ locale === 'zh-CN' ? '暂无持仓' : 'No holdings yet' }}</h3>
        <p>
          {{ locale === 'zh-CN'
            ? '服务器上还没有持仓记录。导入包含 '
            : 'No holdings are stored on the server. Import a CSV with ' }}<code>code,hold_vol</code>{{ locale === 'zh-CN' ? ' 列的CSV文件开始实时追踪。支持 600519 和 600519.SH 两种代码格式。' : ' columns to start tracking. Both 600519 and 600519.SH code formats are accepted.' }}
        </p>
        <button class="btn btn-primary" @click="$refs.fileInput.click()">{{ locale === 'zh-CN' ? '导入CSV' : 'Import CSV' }}</button>
      </template>
    </div>

    <!-- Initial load -->
    <div v-if="loadingInitial" class="lp-loading">
      <div class="status-spinner" style="width:24px;height:24px;"></div>
      <span>{{ locale === 'zh-CN' ? '正在加载服务器持仓...' : 'Loading server-side holdings...' }}</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useI18n } from '../i18n/index.js'
import {
  getLivePortfolio,
  getPortfolioHoldings,
  importPortfolioHoldings,
  getApiErrorMessage,
} from '../api/index.js'
import { normalizeSymbol, symbolCode } from '../utils/symbols.js'

const { $t, locale } = useI18n()

const emit = defineEmits(['toast'])

const portfolio = ref(null)
// Canonical holdings kept when prices are unavailable (server import or CSV).
const holdingsOnly = ref([])
// 'server' = restored from GET /portfolio/holdings, 'csv' = imported here.
const holdingsSource = ref('')
const priceError = ref('')
// Failure to even read the server-side book: distinct from "no holdings".
const holdingsError = ref('')
const loading = ref(false)
const refreshing = ref(false)
const loadingInitial = ref(false)
// Monotonic id for /portfolio/live requests: a slow response for an older
// book (or an overlapping refresh) must not overwrite newer state.
let priceRequestSeq = 0

const hasHoldings = computed(() => (portfolio.value?.holdings?.length || 0) > 0 || holdingsOnly.value.length > 0)

const sourceLabel = computed(() => {
  if (holdingsSource.value === 'server') return locale.value === 'zh-CN' ? '服务器持仓' : 'Server holdings'
  if (holdingsSource.value === 'csv') return locale.value === 'zh-CN' ? 'CSV 导入' : 'CSV import'
  return ''
})

const priceBasis = computed(() => {
  const basis = portfolio.value?.price_basis || portfolio.value?.holdings?.[0]?.price_basis || ''
  return basis ? String(basis).toUpperCase() : ''
})

const basisTitle = computed(() => {
  const source = portfolio.value?.price_source || ''
  const zh = locale.value === 'zh-CN'
  return `${zh ? '价格基准' : 'Price basis'}: ${priceBasis.value}${source ? ` (${source})` : ''}`
})

function formatNumber(v) {
  if (v == null) return '--'
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(v)
}

function formatShares(v) {
  if (v == null) return '--'
  return new Intl.NumberFormat('en-US').format(v)
}

function getWeight(mv) {
  if (!portfolio.value?.total_value || !mv) return '0.00'
  return (mv / portfolio.value.total_value * 100).toFixed(2)
}

function getBarWidth(pnlPct) {
  if (pnlPct == null) return 0
  return Math.min(Math.abs(pnlPct) * 20, 100)
}

function applyServerHoldings(records) {
  holdingsOnly.value = (records || [])
    .map(h => ({ code: normalizeSymbol(h.code), hold_vol: Number(h.hold_vol) }))
    .filter(h => h.hold_vol > 0)
}

// On mount the server may already hold an imported book (the component is
// destroyed by v-if on every tab switch, and a page reload loses all client
// state). Restore it and try to price it; a failed price feed must leave the
// holdings visible rather than showing the old "import a CSV" empty state.
onMounted(loadExisting)

async function loadExisting() {
  loadingInitial.value = true
  holdingsError.value = ''
  try {
    const holdings = await getPortfolioHoldings()
    if (holdings?.n_stocks) {
      holdingsSource.value = 'server'
      applyServerHoldings(holdings.holdings)
      await refreshPrices({ silent: true })
    }
  } catch (e) {
    // The server-side book could not be read. Do not claim "no holdings" —
    // surface the reason with a retry path.
    holdingsError.value = getApiErrorMessage(e, locale.value === 'zh-CN' ? '无法读取服务器持仓' : 'holdings unavailable')
  } finally {
    loadingInitial.value = false
  }
}

async function refreshPrices({ silent = false } = {}) {
  if (!hasHoldings.value && !portfolio.value) {
    if (!silent) {
      emit('toast', {
        message: locale.value === 'zh-CN' ? '暂无持仓，请先导入 CSV' : 'No holdings yet — import a CSV first',
        type: 'info',
      })
    }
    return
  }
  refreshing.value = true
  const seq = ++priceRequestSeq
  try {
    const result = await getLivePortfolio()
    if (seq !== priceRequestSeq) return
    if (result?.error) throw new Error(result.error)
    portfolio.value = result
    priceError.value = ''
    if (!silent) {
      const staleNote = result.stale
        ? (locale.value === 'zh-CN' ? `（数据日期 ${result.data_date}）` : ` (data date ${result.data_date})`)
        : ''
      emit('toast', { message: (locale.value === 'zh-CN' ? '行情已更新' : 'Prices updated') + staleNote, type: result.stale ? 'warning' : 'success' })
    }
  } catch (err) {
    if (seq !== priceRequestSeq) return
    const status = err?.response?.status
    portfolio.value = null
    if (status === 409) {
      // 409 = the server holds no book at all (a genuine empty state).
      holdingsOnly.value = []
      holdingsSource.value = ''
      priceError.value = ''
      if (!silent) {
        emit('toast', { message: locale.value === 'zh-CN' ? '服务器上没有持仓，请先导入 CSV' : 'No holdings on the server; import a CSV first', type: 'info' })
      }
    } else {
      // 503 (feed down / refresh already in progress), transport errors: keep
      // the holdings table and surface the reason instead of a zero-value
      // portfolio.
      priceError.value = getApiErrorMessage(err, locale.value === 'zh-CN' ? '行情不可用' : 'prices unavailable')
      if (!silent) {
        emit('toast', {
          message: (locale.value === 'zh-CN' ? '刷新失败：' : 'Refresh failed: ') + priceError.value,
          type: 'error',
        })
      }
    }
  } finally {
    if (seq === priceRequestSeq) refreshing.value = false
  }
}

async function onFileUpload(e) {
  const file = e.target.files?.[0]
  e.target.value = ''
  if (!file) return

  // Invalidate any in-flight price refresh for the previous book so a slow
  // response cannot overwrite the imported holdings.
  priceRequestSeq++
  refreshing.value = false

  loading.value = true
  try {
    const text = (await file.text()).replace(/^﻿/, '')
    const lines = text.trim().split(/\r?\n/).filter(line => line.trim())
    if (lines.length < 2) {
      throw new Error(locale.value === 'zh-CN' ? 'CSV 为空或只有表头' : 'CSV is empty or has only a header')
    }
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase().replace(/^﻿/, ''))

    const codeIdx = headers.findIndex(h => h === 'code' || h === 'ticker' || h === 'symbol')
    const volIdx = headers.findIndex(h => h === 'hold_vol' || h === 'shares' || h === 'volume' || h === 'quantity')

    if (codeIdx === -1 || volIdx === -1) {
      throw new Error(locale.value === 'zh-CN' ? 'CSV文件必须包含code和hold_vol列' : 'CSV must have code and hold_vol columns')
    }

    const rows = []
    const seen = new Set()
    let skipped = 0
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(',')
      if (cols.length <= Math.max(codeIdx, volIdx)) { skipped++; continue }
      // Accept `600519`, `600519.SH`, `sh600519`, `sh.600519` — the same
      // formats the backend's normalize_symbol understands. The import route
      // stores bare six-digit codes, so submit that form.
      const canonical = normalizeSymbol(cols[codeIdx].trim())
      const bare = symbolCode(canonical)
      const vol = Number(String(cols[volIdx]).trim())
      if (!bare || !Number.isFinite(vol) || vol <= 0 || seen.has(bare)) {
        skipped++
        continue
      }
      seen.add(bare)
      rows.push({ bare, canonical: normalizeSymbol(bare), hold_vol: Math.trunc(vol) })
    }

    if (!rows.length) {
      throw new Error(locale.value === 'zh-CN' ? '没有可导入的有效持仓行' : 'No valid holdings rows to import')
    }

    await importPortfolioHoldings(rows.map(r => ({ code: r.bare, hold_vol: r.hold_vol })))
    holdingsSource.value = 'csv'
    holdingsError.value = ''
    holdingsOnly.value = rows.map(r => ({ code: r.canonical, hold_vol: r.hold_vol }))
    // The previous book's prices must not keep rendering under the new "CSV
    // import" label while the fresh /portfolio/live request is in flight.
    portfolio.value = null

    // Prefer the server's authoritative copy (dedupe/validation happens there).
    try {
      const holdings = await getPortfolioHoldings()
      if (holdings?.n_stocks) applyServerHoldings(holdings.holdings)
    } catch (_) {}

    // The import already succeeded; a price-feed failure must not be reported
    // as an import failure.
    await refreshPrices({ silent: true })

    const priced = portfolio.value
      ? (locale.value === 'zh-CN'
          ? `，${portfolio.value.n_with_price}/${portfolio.value.n_positions} 条有行情`
          : `, ${portfolio.value.n_with_price}/${portfolio.value.n_positions} priced`)
      : (locale.value === 'zh-CN' ? '；行情暂不可用，可点击“刷新行情”重试' : '; prices unavailable — click "Refresh Prices" to retry')
    const skippedNote = skipped
      ? (locale.value === 'zh-CN' ? `（跳过 ${skipped} 行无效/重复数据）` : ` (skipped ${skipped} invalid/duplicate rows)`)
      : ''
    emit('toast', {
      message: (locale.value === 'zh-CN'
        ? `已导入 ${rows.length} 条持仓${priced}${skippedNote}`
        : `Imported ${rows.length} holdings${priced}${skippedNote}`),
      type: 'success',
    })
  } catch (err) {
    emit('toast', {
      message: (locale.value === 'zh-CN' ? '导入失败: ' : 'Import failed: ') + getApiErrorMessage(err),
      type: 'error',
    })
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.live-portfolio {
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: 100%;
}

.lp-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}

.lp-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 8px;
  letter-spacing: 0.5px;
}

.lp-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
}

.lp-dot.alive {
  background: var(--green);
  box-shadow: 0 0 6px rgba(52,211,153,0.5);
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.lp-chip {
  font-size: 8px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  padding: 2px 6px;
  border-radius: 3px;
  background: rgba(77,166,255,0.12);
  color: var(--accent);
  border: 1px solid rgba(77,166,255,0.25);
}

.lp-chip-basis {
  background: rgba(52,211,153,0.1);
  color: var(--green);
  border-color: rgba(52,211,153,0.25);
  cursor: help;
}

.lp-actions {
  display: flex;
  gap: 8px;
  align-items: center;
}

.lp-price-date {
  font-size: 10px;
  color: var(--text-muted);
  font-family: var(--font-mono, monospace);
}

.lp-price-date.stale {
  color: var(--orange, #fb923c);
}

.lp-banner {
  flex-shrink: 0;
  font-size: 10.5px;
  line-height: 1.5;
  padding: 7px 12px;
  border-radius: 5px;
  display: flex;
  gap: 10px;
  align-items: baseline;
  flex-wrap: wrap;
}

.lp-banner-warn {
  background: rgba(251,191,36,0.08);
  border: 1px solid rgba(251,191,36,0.25);
  color: var(--orange);
}

.lp-banner-info {
  background: rgba(77,166,255,0.06);
  border: 1px solid rgba(77,166,255,0.2);
  color: var(--text-secondary);
}

.lp-banner-error {
  color: var(--red);
  font-family: var(--font-mono);
  font-size: 10px;
}

.lp-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 8px;
  flex-shrink: 0;
}

.lp-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  padding: 10px 12px;
  text-align: center;
}

.lp-card-label {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}

.lp-card-value {
  font-size: 16px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}

.lp-accent { color: var(--accent); }
.lp-pos { color: var(--green); }
.lp-neg { color: var(--red); }
.lp-green { color: var(--green); }
.lp-warn { color: var(--orange); }

.lp-table-wrap {
  flex: 1;
  overflow: auto;
  min-height: 0;
  border: 1px solid var(--border);
  border-radius: 6px;
}

.lp-tbl {
  width: 100%;
  border-collapse: collapse;
  font-size: 10.5px;
  font-variant-numeric: tabular-nums;
}

.lp-tbl th {
  padding: 6px 8px;
  font-size: 9px;
  font-weight: 700;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.6px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
  position: sticky;
  top: 0;
  z-index: 1;
}

.lp-tbl td {
  padding: 4px 8px;
  border-bottom: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  white-space: nowrap;
}

.lp-row:hover td { background: rgba(77,166,255,0.04); }

.lp-rank { text-align: center; color: var(--text-dim); font-size: 9px; }
.lp-code { font-family: var(--font-mono); font-weight: 600; color: var(--accent); }
.lp-shares { text-align: right; font-family: var(--font-mono); }
.lp-price { text-align: right; font-family: var(--font-mono); }
.lp-dim { color: var(--text-dim); }
.lp-value { text-align: right; font-family: var(--font-mono); font-weight: 600; }
.lp-pnl { text-align: right; font-family: var(--font-mono); font-weight: 700; }
.lp-weight { text-align: right; font-family: var(--font-mono); font-size: 9px; }
.lp-basis {
  font-family: var(--font-mono);
  font-size: 9px;
  text-align: center;
  color: var(--green);
}

.lp-bar-cell { width: 80px; }
.lp-bar-wrap {
  height: 4px;
  background: var(--bg-input);
  border-radius: 2px;
  overflow: hidden;
}

.lp-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.4s ease;
  min-width: 2px;
}

.lp-bar-pos { background: linear-gradient(90deg, var(--green-dim), var(--green)); }
.lp-bar-neg { background: linear-gradient(90deg, var(--red-dim), var(--red)); }

.lp-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: var(--text-muted);
}

.lp-empty-icon { font-size: 48px; opacity: 0.2; }
.lp-empty h3 { font-size: 14px; color: var(--text-secondary); }
.lp-empty p { font-size: 12px; max-width: 400px; text-align: center; line-height: 1.6; }
.lp-empty code { background: var(--bg-input); padding: 2px 6px; border-radius: 3px; font-family: var(--font-mono); }
.lp-empty-error { color: var(--red); font-family: var(--font-mono); font-size: 11px; }

.lp-loading {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: var(--text-muted);
  font-size: 12px;
}
</style>
