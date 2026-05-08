<template>
  <div class="ms-container">
    <!-- Add Strategy Form -->
    <div class="ms-add-form">
      <div class="ms-form-row">
        <input v-model="newStrat.name" placeholder="Strategy name" class="ms-input" />
        <select v-model="newStrat.optimizer" class="ms-select">
          <option value="equal_weight">Equal Weight</option>
          <option value="mean_variance">Mean-Variance</option>
          <option value="risk_parity">Risk Parity</option>
        </select>
        <select v-model="newStrat.alpha_method" class="ms-select">
          <option value="equal_weight">EW Alpha</option>
          <option value="ic_weighted">IC Alpha</option>
          <option value="icir_weighted">ICIR Alpha</option>
        </select>
        <input v-model.number="newStrat.allocation_pct" type="number" min="0" max="1" step="0.05"
          placeholder="Alloc %" class="ms-input ms-input-sm" />
        <button class="ms-btn ms-btn-add" @click="addNew">+ Add</button>
      </div>
    </div>

    <!-- Aggregate Metrics -->
    <div class="ms-agg" v-if="metrics">
      <div class="ms-agg-card">
        <div class="ms-agg-label">Total Capital</div>
        <div class="ms-agg-val">{{ fmtNum(metrics.total_capital) }}</div>
      </div>
      <div class="ms-agg-card">
        <div class="ms-agg-label">Total Value</div>
        <div class="ms-agg-val">{{ fmtNum(metrics.total_value) }}</div>
      </div>
      <div class="ms-agg-card">
        <div class="ms-agg-label">Total P&L</div>
        <div class="ms-agg-val" :class="metrics.total_pnl >= 0 ? 'pos' : 'neg'">{{ fmtNum(metrics.total_pnl) }}</div>
      </div>
      <div class="ms-agg-card">
        <div class="ms-agg-label">Aggregate Sharpe</div>
        <div class="ms-agg-val accent">{{ metrics.aggregate_sharpe?.toFixed(2) || '--' }}</div>
      </div>
      <div class="ms-agg-card">
        <div class="ms-agg-label">Aggregate DD</div>
        <div class="ms-agg-val neg">{{ (metrics.aggregate_max_dd * 100)?.toFixed(1) || '0.0' }}%</div>
      </div>
      <div class="ms-agg-card">
        <div class="ms-agg-label">Active</div>
        <div class="ms-agg-val accent">{{ metrics.n_active }}/{{ metrics.n_strategies }}</div>
      </div>
    </div>

    <!-- Strategy Table -->
    <div class="ms-table-wrap">
      <table class="ms-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Optimizer</th>
            <th>Alpha</th>
            <th>Allocation</th>
            <th>Capital</th>
            <th>Value</th>
            <th>P&L</th>
            <th>Return</th>
            <th>Sharpe</th>
            <th>Max DD</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="s in strategies" :key="s.strategy_id" :class="{ 'ms-inactive': !s.is_active }">
            <td class="ms-name">{{ s.name }}</td>
            <td><span class="ms-tag">{{ shortOpt(s.optimizer) }}</span></td>
            <td><span class="ms-tag">{{ shortAlpha(s.alpha_method) }}</span></td>
            <td class="mono">{{ (s.allocation_pct * 100).toFixed(1) }}%</td>
            <td class="mono">{{ fmtNum(s.capital_allocated) }}</td>
            <td class="mono">{{ fmtNum(s.current_value) }}</td>
            <td class="mono" :class="s.total_pnl >= 0 ? 'pos' : 'neg'">{{ fmtNum(s.total_pnl) }}</td>
            <td class="mono" :class="s.total_return >= 0 ? 'pos' : 'neg'">{{ (s.total_return * 100).toFixed(2) }}%</td>
            <td class="mono accent">{{ s.sharpe_ratio?.toFixed(2) || '0.00' }}</td>
            <td class="mono neg">{{ (s.max_drawdown * 100).toFixed(1) }}%</td>
            <td>
              <button class="ms-btn-rm" @click="remove(s.strategy_id)" title="Remove">&times;</button>
            </td>
          </tr>
          <tr v-if="!strategies.length">
            <td colspan="11" class="ms-empty">No strategies registered. Add one above.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Correlation Matrix -->
    <div class="ms-corr" v-if="metrics?.correlation_matrix && Object.keys(metrics.correlation_matrix).length > 1">
      <div class="ms-corr-title">Strategy Correlation Matrix</div>
      <div class="ms-corr-grid">
        <div v-for="(row, sid) in metrics.correlation_matrix" :key="sid" class="ms-corr-row">
          <div class="ms-corr-label">{{ getStratName(sid) }}</div>
          <div v-for="(val, sid2) in row" :key="sid2"
            class="ms-corr-cell"
            :style="{ background: corrColor(val) }"
            :title="`${getStratName(sid)} vs ${getStratName(sid2)}: ${val?.toFixed(3)}`">
            {{ val?.toFixed(2) }}
          </div>
        </div>
      </div>
    </div>

    <!-- Risk Alerts -->
    <div class="ms-alerts" v-if="alerts.length">
      <div class="ms-alert-title">Risk Alerts</div>
      <div v-for="a in alerts" :key="a.strategy_id + a.type" class="ms-alert" :class="'ms-alert-' + a.severity">
        <span class="ms-alert-icon">{{ a.severity === 'red' ? '!!' : '!' }}</span>
        <span>{{ a.message }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { addStrategy, removeStrategy, listStrategies, getStrategyMetrics, getStrategyAlerts } from '../api/index.js'

const emit = defineEmits(['toast'])

const strategies = ref([])
const metrics = ref(null)
const alerts = ref([])

const newStrat = reactive({
  name: '',
  optimizer: 'mean_variance',
  alpha_method: 'icir_weighted',
  allocation_pct: 0.2,
})

async function loadAll() {
  try {
    const [list, m, a] = await Promise.all([listStrategies(), getStrategyMetrics(), getStrategyAlerts()])
    strategies.value = list.strategies || []
    metrics.value = m
    alerts.value = a.alerts || []
  } catch (e) {
    console.error('Failed to load strategies', e)
  }
}

async function addNew() {
  if (!newStrat.name) {
    emit('toast', { message: 'Strategy name required', type: 'error' })
    return
  }
  try {
    await addStrategy({ ...newStrat })
    emit('toast', { message: `Added strategy: ${newStrat.name}`, type: 'success' })
    newStrat.name = ''
    await loadAll()
  } catch (e) {
    emit('toast', { message: 'Failed to add strategy', type: 'error' })
  }
}

async function remove(sid) {
  try {
    await removeStrategy(sid)
    emit('toast', { message: 'Strategy removed', type: 'success' })
    await loadAll()
  } catch (e) {
    emit('toast', { message: 'Failed to remove strategy', type: 'error' })
  }
}

function getStratName(sid) {
  const s = strategies.value.find(x => x.strategy_id === sid)
  return s ? s.name : sid?.slice(0, 6) || '?'
}

function shortOpt(v) {
  return { equal_weight: 'EW', mean_variance: 'MV', risk_parity: 'RP' }[v] || v
}

function shortAlpha(v) {
  return { equal_weight: 'EW', ic_weighted: 'IC', icir_weighted: 'ICIR' }[v] || v
}

function fmtNum(v) {
  if (!v && v !== 0) return '--'
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + 'M'
  if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + 'K'
  return v.toFixed(0)
}

function corrColor(v) {
  if (v == null) return 'transparent'
  const abs = Math.abs(v)
  if (v > 0) return `rgba(34,197,94,${abs * 0.4})`
  return `rgba(239,68,68,${abs * 0.4})`
}

onMounted(loadAll)

defineExpose({ loadAll })
</script>

<style scoped>
.ms-container { display: flex; flex-direction: column; gap: 10px; font-size: 11px; }
.ms-add-form { background: var(--bg-card, #111827); border: 1px solid var(--border, #1e2a3a); border-radius: 6px; padding: 8px; }
.ms-form-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.ms-input, .ms-select {
  background: var(--bg-base, #0a0e17); border: 1px solid var(--border, #1e2a3a); border-radius: 4px;
  color: var(--text-primary, #e6edf3); padding: 4px 8px; font-size: 11px; font-family: inherit;
}
.ms-input { flex: 1; min-width: 100px; }
.ms-input-sm { max-width: 80px; }
.ms-btn { padding: 4px 12px; border-radius: 4px; border: none; cursor: pointer; font-size: 11px; font-weight: 600; }
.ms-btn-add { background: #4da6ff; color: #0a0e17; }
.ms-btn-add:hover { background: #6db8ff; }
.ms-btn-rm { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 14px; padding: 0 4px; }

.ms-agg { display: flex; gap: 6px; flex-wrap: wrap; }
.ms-agg-card {
  background: var(--bg-card, #111827); border: 1px solid var(--border, #1e2a3a); border-radius: 6px;
  padding: 6px 10px; flex: 1; min-width: 100px; text-align: center;
}
.ms-agg-label { font-size: 8px; font-weight: 700; color: #6b7a8d; text-transform: uppercase; letter-spacing: 0.5px; }
.ms-agg-val { font-size: 14px; font-weight: 700; font-family: 'SF Mono', 'Fira Code', monospace; color: #e6edf3; margin-top: 2px; }
.ms-agg-val.pos { color: #22c55e; }
.ms-agg-val.neg { color: #ef4444; }
.ms-agg-val.accent { color: #4da6ff; }

.ms-table-wrap { overflow-x: auto; }
.ms-table { width: 100%; border-collapse: collapse; }
.ms-table th {
  padding: 4px 8px; font-size: 8px; font-weight: 700; color: #6b7a8d; text-transform: uppercase;
  letter-spacing: 0.5px; text-align: left; border-bottom: 1px solid #1e2a3a; background: #0d1117;
}
.ms-table td { padding: 4px 8px; border-bottom: 1px solid #161b22; color: #c9d1d9; }
.ms-table tr:hover td { background: rgba(77,166,255,0.04); }
.ms-inactive td { opacity: 0.5; }
.ms-name { font-weight: 600; color: #4da6ff; }
.ms-tag { font-size: 9px; padding: 1px 5px; border-radius: 3px; background: rgba(77,166,255,0.1); color: #4da6ff; }
.ms-empty { text-align: center; color: #6b7a8d; padding: 16px !important; }
.mono { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 10px; }
.pos { color: #22c55e; }
.neg { color: #ef4444; }
.accent { color: #4da6ff; }

.ms-corr { background: var(--bg-card, #111827); border: 1px solid var(--border, #1e2a3a); border-radius: 6px; padding: 8px; }
.ms-corr-title { font-size: 9px; font-weight: 700; color: #6b7a8d; text-transform: uppercase; margin-bottom: 6px; }
.ms-corr-grid { display: flex; flex-direction: column; gap: 2px; }
.ms-corr-row { display: flex; gap: 2px; align-items: center; }
.ms-corr-label { width: 80px; font-size: 9px; color: #6b7a8d; text-align: right; padding-right: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ms-corr-cell {
  width: 50px; height: 24px; display: flex; align-items: center; justify-content: center;
  font-size: 9px; font-family: monospace; color: #c9d1d9; border-radius: 3px;
}

.ms-alerts { display: flex; flex-direction: column; gap: 4px; }
.ms-alert-title { font-size: 9px; font-weight: 700; color: #6b7a8d; text-transform: uppercase; margin-bottom: 2px; }
.ms-alert {
  padding: 6px 10px; border-radius: 4px; font-size: 10px; display: flex; gap: 6px; align-items: center;
}
.ms-alert-red { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }
.ms-alert-orange { background: rgba(251,191,36,0.1); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
.ms-alert-icon { font-weight: 700; }
</style>
