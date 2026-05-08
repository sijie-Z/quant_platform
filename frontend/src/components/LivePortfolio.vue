<template>
  <div class="live-portfolio">
    <!-- Header -->
    <div class="lp-header">
      <div class="lp-title">
        <span class="lp-dot alive"></span>
        LIVE PORTFOLIO TRACKER
      </div>
      <div class="lp-actions">
        <button class="btn btn-sm btn-secondary" @click="$refs.fileInput.click()">
          Import CSV
        </button>
        <button class="btn btn-sm btn-primary" @click="refreshPrices" :disabled="refreshing">
          {{ refreshing ? 'Loading...' : 'Refresh Prices' }}
        </button>
        <input ref="fileInput" type="file" accept=".csv" style="display:none" @change="onFileUpload" />
      </div>
    </div>

    <!-- Summary Cards -->
    <div class="lp-summary" v-if="portfolio">
      <div class="lp-card">
        <div class="lp-card-label">Total Value</div>
        <div class="lp-card-value lp-accent">{{ formatNumber(portfolio.total_value) }}</div>
      </div>
      <div class="lp-card">
        <div class="lp-card-label">Daily P&L</div>
        <div :class="['lp-card-value', portfolio.total_pnl >= 0 ? 'lp-pos' : 'lp-neg']">
          {{ portfolio.total_pnl >= 0 ? '+' : '' }}{{ formatNumber(portfolio.total_pnl) }}
        </div>
      </div>
      <div class="lp-card">
        <div class="lp-card-label">Daily Return</div>
        <div :class="['lp-card-value', portfolio.daily_return_pct >= 0 ? 'lp-pos' : 'lp-neg']">
          {{ portfolio.daily_return_pct >= 0 ? '+' : '' }}{{ portfolio.daily_return_pct?.toFixed(2) }}%
        </div>
      </div>
      <div class="lp-card">
        <div class="lp-card-label">Positions</div>
        <div class="lp-card-value">{{ portfolio.n_positions }}</div>
      </div>
      <div class="lp-card">
        <div class="lp-card-label">With Price</div>
        <div class="lp-card-value lp-green">{{ portfolio.n_with_price }}</div>
      </div>
      <div class="lp-card" v-if="portfolio.n_no_price > 0">
        <div class="lp-card-label">No Price</div>
        <div class="lp-card-value lp-warn">{{ portfolio.n_no_price }}</div>
      </div>
    </div>

    <!-- Holdings Table -->
    <div class="lp-table-wrap" v-if="portfolio?.holdings?.length">
      <table class="lp-tbl">
        <thead>
          <tr>
            <th>#</th>
            <th>Code</th>
            <th>Shares</th>
            <th>Close</th>
            <th>Prev Close</th>
            <th>Mkt Value</th>
            <th>P&L</th>
            <th>Return</th>
            <th>Weight</th>
            <th>P&L Bar</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(h, i) in portfolio.holdings" :key="h.code" class="lp-row">
            <td class="lp-rank">{{ i + 1 }}</td>
            <td class="lp-code">{{ h.code }}</td>
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

    <!-- Empty State -->
    <div v-if="!portfolio && !loading" class="lp-empty">
      <div class="lp-empty-icon">&#9776;</div>
      <h3>Import Holdings CSV</h3>
      <p>Upload a CSV file with <code>code,hold_vol</code> columns to track your portfolio in real-time.</p>
      <button class="btn btn-primary" @click="$refs.fileInput.click()">Import CSV</button>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="lp-loading">
      <div class="status-spinner" style="width:24px;height:24px;"></div>
      <span>Fetching real-time prices from baostock...</span>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const emit = defineEmits(['toast'])

const portfolio = ref(null)
const loading = ref(false)
const refreshing = ref(false)

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

async function onFileUpload(e) {
  const file = e.target.files[0]
  if (!file) return

  loading.value = true
  try {
    const text = await file.text()
    const lines = text.trim().split('\n')
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase())

    const codeIdx = headers.findIndex(h => h === 'code' || h === 'ticker' || h === 'symbol')
    const volIdx = headers.findIndex(h => h === 'hold_vol' || h === 'shares' || h === 'volume' || h === 'quantity')

    if (codeIdx === -1 || volIdx === -1) {
      emit('toast', { message: 'CSV must have code and hold_vol columns', type: 'error' })
      loading.value = false
      return
    }

    const data = []
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(',')
      if (cols.length > Math.max(codeIdx, volIdx)) {
        data.push({
          code: cols[codeIdx].trim(),
          hold_vol: parseInt(cols[volIdx].trim()) || 0,
        })
      }
    }

    // Import to backend
    await axios.post('/api/portfolio/import', { data })

    // Fetch live prices
    const result = await axios.get('/api/portfolio/live')
    portfolio.value = result.data

    emit('toast', { message: `Imported ${data.length} holdings, got prices for ${portfolio.value.n_with_price}`, type: 'success' })
  } catch (err) {
    emit('toast', { message: `Import failed: ${err.message}`, type: 'error' })
  } finally {
    loading.value = false
  }
}

async function refreshPrices() {
  if (!portfolio.value) return
  refreshing.value = true
  try {
    const result = await axios.get('/api/portfolio/live')
    portfolio.value = result.data
    emit('toast', { message: 'Prices updated', type: 'success' })
  } catch (err) {
    emit('toast', { message: `Refresh failed: ${err.message}`, type: 'error' })
  } finally {
    refreshing.value = false
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

.lp-actions {
  display: flex;
  gap: 8px;
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
