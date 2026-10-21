<template>
  <div class="ms-container">
    <!-- Add Strategy Form -->
    <div class="ms-add-form">
      <div class="ms-form-row">
        <input v-model="newStrat.name" :placeholder="locale === 'zh-CN' ? '策略名称' : 'Strategy name'" class="ms-input" />
        <select v-model="newStrat.optimizer" class="ms-select">
          <option value="equal_weight">{{ locale === 'zh-CN' ? '等权' : 'Equal Weight' }}</option>
          <option value="mean_variance">{{ locale === 'zh-CN' ? '均值方差' : 'Mean-Variance' }}</option>
          <option value="risk_parity">{{ locale === 'zh-CN' ? '风险平价' : 'Risk Parity' }}</option>
        </select>
        <select v-model="newStrat.alpha_method" class="ms-select">
          <option value="equal_weight">{{ locale === 'zh-CN' ? '等权Alpha' : 'EW Alpha' }}</option>
          <option value="ic_weighted">{{ locale === 'zh-CN' ? 'IC加权Alpha' : 'IC Alpha' }}</option>
          <option value="icir_weighted">{{ locale === 'zh-CN' ? 'ICIR加权Alpha' : 'ICIR Alpha' }}</option>
        </select>
        <input v-model.number="newStrat.allocation_pct" type="number" min="0" max="1" step="0.05"
          :placeholder="locale === 'zh-CN' ? '比例 0–1' : 'Alloc 0–1'" class="ms-input ms-input-sm" />
        <button class="ms-btn ms-btn-add" @click="addNew">{{ locale === 'zh-CN' ? '+ 添加' : '+ Add' }}</button>
      </div>
    </div>

    <!-- Aggregate Metrics -->
    <div class="ms-agg" v-if="metrics">
      <div class="ms-agg-card">
        <div class="ms-agg-label">{{ locale === 'zh-CN' ? '总资本' : 'Total Capital' }}</div>
        <div class="ms-agg-val">{{ fmtNum(metrics.total_capital) }}</div>
      </div>
      <div class="ms-agg-card">
        <div class="ms-agg-label">{{ locale === 'zh-CN' ? '总价值' : 'Total Value' }}</div>
        <div class="ms-agg-val">{{ fmtNum(metrics.total_value) }}</div>
      </div>
      <div class="ms-agg-card">
        <div class="ms-agg-label">{{ locale === 'zh-CN' ? '总盈亏' : 'Total P&L' }}</div>
        <div class="ms-agg-val" :class="metrics.total_pnl >= 0 ? 'pos' : 'neg'">{{ fmtNum(metrics.total_pnl) }}</div>
      </div>
      <div class="ms-agg-card">
        <div class="ms-agg-label">{{ locale === 'zh-CN' ? '综合夏普' : 'Aggregate Sharpe' }}</div>
        <div class="ms-agg-val accent">{{ metrics.aggregate_sharpe?.toFixed(2) || '--' }}</div>
      </div>
      <div class="ms-agg-card">
        <div class="ms-agg-label">{{ locale === 'zh-CN' ? '综合回撤' : 'Aggregate DD' }}</div>
        <div class="ms-agg-val neg">{{ (metrics.aggregate_max_dd * 100)?.toFixed(1) || '0.0' }}%</div>
      </div>
      <div class="ms-agg-card">
        <div class="ms-agg-label">{{ locale === 'zh-CN' ? '活跃数' : 'Active' }}</div>
        <div class="ms-agg-val accent">{{ metrics.n_active }}/{{ metrics.n_strategies }}</div>
      </div>
    </div>

    <!-- Strategy Table -->
    <div class="ms-table-wrap">
      <table class="ms-table">
        <thead>
          <tr>
            <th>{{ locale === 'zh-CN' ? '名称' : 'Name' }}</th>
            <th>{{ locale === 'zh-CN' ? '优化器' : 'Optimizer' }}</th>
            <th>{{ locale === 'zh-CN' ? 'Alpha' : 'Alpha' }}</th>
            <th>{{ locale === 'zh-CN' ? '分配比例' : 'Allocation' }}</th>
            <th>{{ locale === 'zh-CN' ? '资本' : 'Capital' }}</th>
            <th>{{ locale === 'zh-CN' ? '价值' : 'Value' }}</th>
            <th>{{ locale === 'zh-CN' ? '盈亏' : 'P&L' }}</th>
            <th>{{ locale === 'zh-CN' ? '收益率' : 'Return' }}</th>
            <th>{{ locale === 'zh-CN' ? '夏普' : 'Sharpe' }}</th>
            <th>{{ locale === 'zh-CN' ? '最大回撤' : 'Max DD' }}</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="s in strategies" :key="s.strategy_id" :class="{ 'ms-inactive': !s.is_active }">
            <td class="ms-name">{{ s.name }}</td>
            <td><span class="ms-tag">{{ shortOpt(s.optimizer) }}</span></td>
            <td><span class="ms-tag">{{ shortAlpha(s.alpha_method) }}</span></td>
            <td class="mono ms-alloc-cell">
              <input
                v-model.number="allocDraft[s.strategy_id]"
                type="number"
                min="0"
                max="100"
                step="0.5"
                class="ms-input ms-alloc-input"
                :aria-label="locale === 'zh-CN' ? '分配比例(%)' : 'Allocation (%)'"
              />%
            </td>
            <td class="mono">{{ fmtNum(s.capital_allocated) }}</td>
            <td class="mono">{{ fmtNum(s.current_value) }}</td>
            <td class="mono" :class="s.total_pnl >= 0 ? 'pos' : 'neg'">{{ fmtNum(s.total_pnl) }}</td>
            <td class="mono" :class="s.total_return >= 0 ? 'pos' : 'neg'">{{ (s.total_return * 100).toFixed(2) }}%</td>
            <td class="mono accent">{{ s.sharpe_ratio?.toFixed(2) || '0.00' }}</td>
            <td class="mono neg">{{ (s.max_drawdown * 100).toFixed(1) }}%</td>
            <td>
              <button class="ms-btn-rm" @click="remove(s.strategy_id)" :title="locale === 'zh-CN' ? '删除' : 'Remove'">&times;</button>
            </td>
          </tr>
          <tr v-if="!strategies.length">
            <td colspan="11" class="ms-empty">{{ loading ? (locale === 'zh-CN' ? '正在加载策略...' : 'Loading strategies...') : (locale === 'zh-CN' ? '尚未注册策略。请在上方添加。' : 'No strategies registered. Add one above.') }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Allocation editor: the backend applies a *full* weight map (omitted
         strategies are zeroed, sum must be <= 1), so edits are collected for
         every registered strategy and submitted together. -->
    <div class="ms-alloc-bar" v-if="strategies.length">
      <span class="ms-alloc-total" :class="{ over: !!allocError }">
        {{ locale === 'zh-CN' ? '合计分配' : 'Total allocated' }}: {{ allocSum.toFixed(1) }}%
      </span>
      <span v-if="allocError" class="ms-alloc-error">{{ allocError }}</span>
      <span v-else class="ms-alloc-hint">{{ locale === 'zh-CN' ? '按总资本的绝对比例提交完整权重表；未列出的策略将被归零' : 'Submits an absolute weight for every strategy; omitted ids are zeroed' }}</span>
      <button
        class="ms-btn ms-btn-apply"
        @click="applyAllocation"
        :disabled="!!allocError || allocating"
      >{{ allocating ? (locale === 'zh-CN' ? '提交中...' : 'Applying...') : (locale === 'zh-CN' ? '应用分配' : 'Apply Allocation') }}</button>
    </div>

    <!-- Correlation Matrix -->
    <div class="ms-corr" v-if="metrics?.correlation_matrix && Object.keys(metrics.correlation_matrix).length > 1">
      <div class="ms-corr-title">{{ locale === 'zh-CN' ? '策略相关性矩阵' : 'Strategy Correlation Matrix' }}</div>
      <div class="ms-corr-grid">
        <div v-for="(row, sid) in metrics.correlation_matrix" :key="sid" class="ms-corr-row">
          <div class="ms-corr-label">{{ getStratName(sid) }}</div>
          <div v-for="(val, sid2) in row" :key="sid2"
            class="ms-corr-cell"
            :style="{ background: corrColor(val) }"
            :title="locale === 'zh-CN' ? `${getStratName(sid)} 对比 ${getStratName(sid2)}：${val?.toFixed(3)}` : `${getStratName(sid)} vs ${getStratName(sid2)}: ${val?.toFixed(3)}`">
            {{ val?.toFixed(2) }}
          </div>
        </div>
      </div>
    </div>

    <!-- Risk Alerts -->
    <div class="ms-alerts" v-if="alerts.length">
      <div class="ms-alert-title">{{ locale === 'zh-CN' ? '风险告警' : 'Risk Alerts' }}</div>
      <div v-for="a in alerts" :key="a.strategy_id + a.type" class="ms-alert" :class="'ms-alert-' + a.severity">
        <span class="ms-alert-icon">{{ a.severity === 'red' ? '!!' : '!' }}</span>
        <span>{{ a.message }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import {
  addStrategy,
  removeStrategy,
  listStrategies,
  allocateCapital,
  getStrategyMetrics,
  getStrategyAlerts,
  getApiErrorMessage,
} from '../api/index.js'
import { useI18n } from '../i18n/index.js'

const { $t, locale } = useI18n()

const emit = defineEmits(['toast'])

const strategies = ref([])
const metrics = ref(null)
const alerts = ref([])
const loading = ref(false)
const allocating = ref(false)
// strategy_id -> allocation in percent (0-100), one entry per registered
// strategy so the submitted map is always complete.
const allocDraft = reactive({})

const newStrat = reactive({
  name: '',
  optimizer: 'mean_variance',
  alpha_method: 'icir_weighted',
  allocation_pct: 0.2,
})

const allocSum = computed(() => strategies.value.reduce((sum, s) => {
  const v = Number(allocDraft[s.strategy_id])
  return sum + (Number.isFinite(v) ? v : 0)
}, 0))

const allocError = computed(() => {
  for (const s of strategies.value) {
    const v = Number(allocDraft[s.strategy_id])
    if (!Number.isFinite(v) || v < 0) {
      return locale.value === 'zh-CN' ? '分配比例必须为非负数' : 'Allocation must be a non-negative number'
    }
  }
  if (allocSum.value > 100 + 1e-7) {
    return locale.value === 'zh-CN'
      ? `合计 ${allocSum.value.toFixed(1)}% 超过 100%`
      : `Total ${allocSum.value.toFixed(1)}% exceeds 100%`
  }
  return ''
})

function syncAllocDraft() {
  const liveIds = new Set(strategies.value.map(s => s.strategy_id))
  for (const id of Object.keys(allocDraft)) {
    if (!liveIds.has(id)) delete allocDraft[id]
  }
  for (const s of strategies.value) {
    allocDraft[s.strategy_id] = Number(((Number(s.allocation_pct) || 0) * 100).toFixed(2))
  }
}

async function loadAll() {
  loading.value = true
  try {
    // One failing endpoint used to discard the other two successful responses
    // and leave the table reading "No strategies registered".
    const [list, m, a] = await Promise.allSettled([
      listStrategies(), getStrategyMetrics(), getStrategyAlerts(),
    ])
    if (list.status === 'fulfilled') {
      strategies.value = list.value.strategies || []
      syncAllocDraft()
    }
    if (m.status === 'fulfilled') metrics.value = m.value
    if (a.status === 'fulfilled') alerts.value = a.value.alerts || []
    const failed = [list, m, a].filter(result => result.status === 'rejected')
    if (failed.length) {
      emit('toast', {
        message: locale.value === 'zh-CN'
          ? `部分策略数据加载失败（${failed.length}/3）`
          : `Some strategy data failed to load (${failed.length}/3)`,
        type: 'warning',
      })
    }
  } finally {
    loading.value = false
  }
}

async function applyAllocation() {
  if (allocError.value) {
    emit('toast', { message: allocError.value, type: 'error' })
    return
  }
  if (!strategies.value.length) return
  allocating.value = true
  try {
    // Full-map semantics: include every strategy id. Values are absolute
    // fractions of total_capital.
    const weights = {}
    for (const s of strategies.value) {
      const pct = Number(allocDraft[s.strategy_id])
      weights[s.strategy_id] = Math.max(0, pct) / 100
    }
    await allocateCapital(weights)
    emit('toast', {
      message: locale.value === 'zh-CN'
        ? `已应用完整权重表（合计 ${allocSum.value.toFixed(1)}%）`
        : `Full allocation applied (total ${allocSum.value.toFixed(1)}%)`,
      type: 'success',
    })
    await loadAll()
  } catch (e) {
    emit('toast', {
      message: (locale.value === 'zh-CN' ? '资本分配失败: ' : 'Allocation failed: ') + getApiErrorMessage(e),
      type: 'error',
    })
  } finally {
    allocating.value = false
  }
}

async function addNew() {
  if (!newStrat.name) {
    emit('toast', { message: locale.value === 'zh-CN' ? '需要策略名称' : 'Strategy name required', type: 'error' })
    return
  }
  const pct = Number(newStrat.allocation_pct)
  if (!Number.isFinite(pct) || pct < 0) {
    emit('toast', { message: locale.value === 'zh-CN' ? '分配比例必须为非负数' : 'Allocation must be a non-negative number', type: 'error' })
    return
  }
  // The add endpoint does not validate the sum; reject >100% client-side so
  // the table cannot claim six 20% strategies against one capital pool.
  const used = strategies.value.reduce((sum, s) => sum + (Number(s.allocation_pct) || 0), 0)
  // The server accepts sums up to 1.0 + 1e-9; mirror that tolerance exactly
  // instead of a looser 100.0001% that would pass here and 422 there.
  if (used + pct > 1 + 1e-9) {
    const msg = locale.value === 'zh-CN'
      ? `总分配将超过 100%（已用 ${(used * 100).toFixed(1)}%，本次 ${(pct * 100).toFixed(1)}%）`
      : `Total allocation would exceed 100% (used ${(used * 100).toFixed(1)}%, new ${(pct * 100).toFixed(1)}%)`
    emit('toast', { message: msg, type: 'error' })
    return
  }
  try {
    await addStrategy({ ...newStrat })
    emit('toast', { message: locale.value === 'zh-CN' ? `已添加策略：${newStrat.name}` : `Added strategy: ${newStrat.name}`, type: 'success' })
    newStrat.name = ''
    await loadAll()
  } catch (e) {
    emit('toast', { message: (locale.value === 'zh-CN' ? '添加策略失败: ' : 'Failed to add strategy: ') + getApiErrorMessage(e), type: 'error' })
  }
}

async function remove(sid) {
  try {
    await removeStrategy(sid)
    emit('toast', { message: locale.value === 'zh-CN' ? '策略已移除' : 'Strategy removed', type: 'success' })
    await loadAll()
  } catch (e) {
    emit('toast', { message: (locale.value === 'zh-CN' ? '移除策略失败: ' : 'Failed to remove strategy: ') + getApiErrorMessage(e), type: 'error' })
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

.ms-alloc-cell { white-space: nowrap; }
.ms-alloc-input {
  width: 64px;
  min-width: 0;
  max-width: 64px;
  padding: 2px 6px;
  font-size: 10px;
  text-align: right;
}
.ms-alloc-bar {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  background: var(--bg-card, #111827);
  border: 1px solid var(--border, #1e2a3a);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 10px;
}
.ms-alloc-total { font-weight: 700; font-family: monospace; color: #22c55e; }
.ms-alloc-total.over { color: #ef4444; }
.ms-alloc-error { color: #ef4444; font-weight: 600; }
.ms-alloc-hint { color: #6b7a8d; }
.ms-btn-apply {
  margin-left: auto;
  background: #22c55e;
  color: #0a0e17;
}
.ms-btn-apply:hover { background: #45d27e; }
.ms-btn-apply:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
