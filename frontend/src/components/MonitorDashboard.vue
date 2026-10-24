<template>
  <div class="monitor-dashboard">
    <!-- Top KPI Strip -->
    <div class="metrics-grid monitor-kpi">
      <div class="metric-card">
        <div class="metric-label">{{ locale === 'zh-CN' ? '总资产净值' : 'Total NAV' }}</div>
        <div class="metric-value text-accent" :title="riskReason">{{ riskAvailable ? formatNum(riskOverview.portfolio_value, 0) : '—' }}</div>
        <div v-if="!riskAvailable" class="metric-sub" :title="riskReason">{{ shortReason(riskReason) }}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">{{ locale === 'zh-CN' ? '日盈亏' : 'Daily P&L' }}</div>
        <div class="metric-value" :class="!riskAvailable ? '' : (riskOverview.daily_pnl >= 0 ? 'text-green' : 'text-red')" :title="riskReason">
          {{ riskAvailable ? formatNum(riskOverview.daily_pnl, 0) : '—' }}
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-label">{{ locale === 'zh-CN' ? '风险等级' : 'Risk Level' }}</div>
        <div class="metric-value" :style="{ color: riskAvailable ? riskColor(riskOverview.risk_level) : 'var(--text-dim)' }" :title="riskReason">
          {{ riskAvailable ? (riskOverview.risk_level || (locale === 'zh-CN' ? '无' : 'N/A')) : 'N/A' }}
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-label">{{ locale === 'zh-CN' ? '回撤' : 'Drawdown' }}</div>
        <div class="metric-value text-orange" :title="riskReason">
          {{ riskAvailable ? '-' + (Number(riskOverview.current_drawdown || 0) * 100).toFixed(2) + '%' : '—' }}
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-label">{{ locale === 'zh-CN' ? '容量使用率' : 'Capacity Use' }}</div>
        <div class="metric-value" :class="!capacityAvailable ? '' : (capacityGauge.usage_pct > 80 ? 'text-red' : 'text-green')" :title="capacityReason">
          {{ capacityAvailable ? formatPct1(capacityGauge.usage_pct) : '—' }}
        </div>
        <div v-if="capacityAvailable && capacityGauge.as_of" class="metric-sub">{{ locale === 'zh-CN' ? '截至' : 'as of' }} {{ capacityGauge.as_of }}</div>
        <div v-else-if="!capacityAvailable" class="metric-sub" :title="capacityReason">{{ shortReason(capacityReason) }}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">{{ locale === 'zh-CN' ? '平均IS成本' : 'Avg IS Cost' }}</div>
        <div class="metric-value text-purple" :title="tcaReason">
          <template v-if="tcaAvailable">{{ formatBps(tcaSummary.mean_is_bps) }} <span class="metric-unit">bps</span></template>
          <template v-else>—</template>
        </div>
        <div v-if="tcaAvailable && tcaSummary.as_of" class="metric-sub">{{ locale === 'zh-CN' ? '截至' : 'as of' }} {{ tcaSummary.as_of }}</div>
        <div v-else-if="!tcaAvailable" class="metric-sub" :title="tcaReason">{{ shortReason(tcaReason) }}</div>
      </div>
    </div>

    <!-- Main Grid: 3 columns -->
    <div class="monitor-grid">
      <!-- Panel 1: Risk Dashboard (left) -->
      <div class="card monitor-card">
        <div class="card-header">
          <span class="card-title"><span class="card-title-dot" style="background:#4da6ff"></span> {{ locale === 'zh-CN' ? '风险仪表盘' : 'Risk Dashboard' }}</span>
          <span class="tag" :style="riskAvailable ? { background: riskColor(riskOverview.risk_level) + '22', color: riskColor(riskOverview.risk_level) } : { background: 'rgba(139,157,192,0.12)', color: '#8b9dc0' }">
            {{ riskAvailable ? (riskOverview.risk_level || (locale === 'zh-CN' ? '无' : 'UNKNOWN')) : 'N/A' }}
          </span>
        </div>
        <div class="card-body">
          <div v-if="riskOverview.available === false" class="chart-unavailable" style="min-height:auto;justify-content:flex-start;text-align:left;padding-bottom:0">
            {{ riskOverview.reason || (locale === 'zh-CN' ? '风控状态不可用' : 'Risk state unavailable') }}
          </div>
          <!-- v-show (not v-if) keeps the chart DOM alive so the cached ECharts
               instances stay valid when availability toggles. -->
          <div v-show="riskOverview.available !== false">
            <!-- Factor Exposure Radar -->
            <div class="chart-title">{{ locale === 'zh-CN' ? 'Barra因子暴露' : 'Barra Factor Exposures' }}</div>
            <div
              v-if="riskOverview.factor_exposures_available === false"
              class="chart-unavailable"
            >{{ riskOverview.factor_exposures_reason || (locale === 'zh-CN' ? '暂无因子暴露数据' : 'No factor exposures available') }}</div>
            <div v-show="riskOverview.factor_exposures_available !== false" ref="riskRadarRef" class="chart-container-sm"></div>
            <!-- Sector Concentration -->
            <div class="chart-title mt-2">{{ locale === 'zh-CN' ? '行业集中度' : 'Sector Concentration' }}</div>
            <div ref="sectorBarRef" class="chart-container-sm"></div>
          </div>
        </div>
      </div>

      <!-- Panel 2: TCA Monitor (center) -->
      <div class="card monitor-card">
        <div class="card-header">
          <span class="card-title"><span class="card-title-dot" style="background:#fb923c"></span> {{ locale === 'zh-CN' ? '交易成本监控' : 'TCA Monitor' }}</span>
          <span class="tag tag-orange">
            {{ tcaAvailable ? tcaSummary.n_orders : '—' }} {{ locale === 'zh-CN' ? '笔订单' : 'orders' }}
            <template v-if="tcaAvailable && tcaSummary.as_of">· {{ locale === 'zh-CN' ? '截至' : 'as of' }} {{ tcaSummary.as_of }}</template>
          </span>
        </div>
        <div class="card-body">
          <div v-if="tcaSummary.available === false" class="chart-unavailable">
            {{ tcaSummary.reason || (locale === 'zh-CN' ? '暂无成交记录，无法计算 TCA' : 'No trades recorded; TCA unavailable') }}
          </div>
          <div v-show="tcaSummary.available !== false">
            <div class="chart-title">{{ locale === 'zh-CN' ? '实现缺口趋势' : 'Implementation Shortfall Trend' }}</div>
            <div ref="tcaLineRef" class="chart-container-sm"></div>
            <div class="chart-title mt-2">{{ locale === 'zh-CN' ? '成本分解' : 'Cost Decomposition' }}</div>
            <div ref="tcaBarRef" class="chart-container-sm"></div>
          </div>
        </div>
      </div>

      <!-- Panel 3: Factor Monitor (right) -->
      <div class="card monitor-card">
        <div class="card-header">
          <span class="card-title"><span class="card-title-dot" style="background:#a78bfa"></span> {{ locale === 'zh-CN' ? '因子监控' : 'Factor Monitor' }}</span>
          <span v-if="factorStatus.decay_alerts.length" class="tag tag-red">
            {{ factorStatus.decay_alerts.length }} {{ locale === 'zh-CN' ? '条告警' : 'alerts' }}
          </span>
        </div>
        <div class="card-body">
          <div v-if="factorStatus.available === false" class="chart-unavailable">
            {{ factorStatus.reason || (locale === 'zh-CN' ? '暂无因子 IC 数据' : 'No factor IC data available') }}
          </div>
          <div v-show="factorStatus.available !== false">
            <div class="chart-title">{{ locale === 'zh-CN' ? '滚动IC（60天）' : 'Rolling IC (60d)' }}</div>
            <div ref="factorICRef" class="chart-container-sm"></div>
            <div class="chart-title mt-2">{{ locale === 'zh-CN' ? '因子归因（bps）' : 'Factor Attribution (bps)' }}</div>
            <div ref="factorAttrRef" class="chart-container-sm"></div>
          </div>
        </div>
      </div>

      <!-- Panel 4: Capacity Gauge (left bottom) -->
      <div class="card monitor-card">
        <div class="card-header">
          <span class="card-title"><span class="card-title-dot" style="background:#34d399"></span> {{ locale === 'zh-CN' ? '容量仪表盘' : 'Capacity Gauge' }}</span>
          <span class="tag tag-green">
            {{ capacityAvailable ? formatNum(capacityGauge.current_aum, 0) : '—' }} {{ locale === 'zh-CN' ? '管理规模' : 'AUM' }}
            <template v-if="capacityAvailable && capacityGauge.as_of">· {{ locale === 'zh-CN' ? '截至' : 'as of' }} {{ capacityGauge.as_of }}</template>
          </span>
        </div>
        <div class="card-body">
          <div v-if="capacityGauge.available === false" class="chart-unavailable">
            {{ capacityGauge.reason || (locale === 'zh-CN' ? '暂无持仓数据，无法估算容量' : 'No positions; capacity unavailable') }}
          </div>
          <div v-show="capacityGauge.available !== false">
            <div class="chart-title">{{ locale === 'zh-CN' ? '容量使用率' : 'Capacity Usage' }}</div>
            <div ref="capacityGaugeRef" class="chart-container-sm"></div>
            <div class="chart-title mt-2">{{ locale === 'zh-CN' ? '规模 vs 夏普衰减' : 'AUM vs Sharpe Decay' }}</div>
            <div ref="capacityCurveRef" class="chart-container-sm"></div>
          </div>
        </div>
      </div>

      <!-- Panel 5: Config UI (center bottom) -->
      <div class="card monitor-card">
        <div class="card-header">
          <span class="card-title"><span class="card-title-dot" style="background:#fbbf24"></span> {{ locale === 'zh-CN' ? '风险配置' : 'Risk Configuration' }}</span>
        </div>
        <div class="card-body">
          <div class="config-form">
            <div class="form-row">
              <label class="config-label">{{ locale === 'zh-CN' ? '最大仓位（小数 0–1）' : 'Max Position (fraction 0–1)' }}</label>
              <input type="number" v-model.number="configForm.max_position_pct" min="0.01" max="0.20" step="0.01" class="config-input" @input="configDirty = true" />
            </div>
            <div class="form-row">
              <label class="config-label">{{ locale === 'zh-CN' ? '最大行业（小数 0–1）' : 'Max Sector (fraction 0–1)' }}</label>
              <input type="number" v-model.number="configForm.max_sector_pct" min="0.05" max="0.50" step="0.05" class="config-input" @input="configDirty = true" />
            </div>
            <div class="form-row">
              <label class="config-label">{{ locale === 'zh-CN' ? '回撤暂停（小数 0–1）' : 'Drawdown Halt (fraction 0–1)' }}</label>
              <input type="number" v-model.number="configForm.max_drawdown_pct" min="0.01" max="0.30" step="0.01" class="config-input" @input="configDirty = true" />
            </div>
            <div class="form-row">
              <label class="config-label">{{ locale === 'zh-CN' ? '日亏损限额（小数 0–1）' : 'Daily Loss Limit (fraction 0–1)' }}</label>
              <input type="number" v-model.number="configForm.max_daily_loss_pct" min="0.005" max="0.10" step="0.005" class="config-input" @input="configDirty = true" />
            </div>
            <div class="config-actions">
              <button class="btn btn-primary btn-sm" @click="submitConfig" :disabled="configSaving">
                {{ configSaving ? (locale === 'zh-CN' ? '保存中...' : 'Saving...') : (locale === 'zh-CN' ? '更新限额' : 'Update Limits') }}
              </button>
            </div>
          </div>
          <!-- Kill Switch -->
          <div class="kill-switch-section mt-3">
            <div class="kill-switch-label">{{ locale === 'zh-CN' ? '紧急熔断开关' : 'Emergency Kill Switch' }}</div>
            <button
              class="btn btn-danger kill-btn"
              @click="confirmKillSwitch"
            >
              {{ killSwitchActive ? (locale === 'zh-CN' ? '解除熔断开关' : 'DEACTIVATE KILL SWITCH') : (locale === 'zh-CN' ? '激活熔断开关' : 'ACTIVATE KILL SWITCH') }}
            </button>
          </div>
          <!-- Config change log -->
          <div class="config-log mt-3">
            <div class="chart-title">{{ locale === 'zh-CN' ? '最近变更' : 'Recent Changes' }}</div>
            <div v-if="configLog.length === 0" class="text-muted text-sm">{{ locale === 'zh-CN' ? '暂无最近变更' : 'No recent changes' }}</div>
            <div v-for="(log, i) in configLog" :key="i" class="config-log-item">
              <span class="text-muted text-xs">{{ log.time }}</span>
              <span class="text-sm">{{ log.message }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Panel 6: Factor Alert Table (right bottom) -->
      <div class="card monitor-card">
        <div class="card-header">
          <span class="card-title"><span class="card-title-dot" style="background:#f87171"></span> {{ locale === 'zh-CN' ? '因子告警' : 'Factor Alerts' }}</span>
        </div>
        <div class="card-body">
          <div v-if="factorStatus.available === false" class="chart-unavailable" style="min-height:auto;justify-content:flex-start;text-align:left;padding-bottom:0">
            {{ factorStatus.reason || (locale === 'zh-CN' ? '暂无因子 IC 数据' : 'No factor IC data available') }}
          </div>
          <div v-else class="table-container">
            <table>
              <thead>
                <tr>
                  <th>{{ locale === 'zh-CN' ? '因子' : 'Factor' }}</th>
                  <th>IC</th>
                  <th>ICIR</th>
                  <th>{{ locale === 'zh-CN' ? '趋势' : 'Trend' }}</th>
                  <th>{{ locale === 'zh-CN' ? '状态' : 'Status' }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="f in factorStatus.factors" :key="f.name">
                  <td class="text-mono">{{ f.name }}</td>
                  <td :class="factorIc(f) > 0 ? 'text-green' : 'text-red'">{{ formatIc(factorIc(f)) }}</td>
                  <td>{{ formatIcir(f.icir) }}</td>
                  <td>
                    <span :class="f.trend === 'up' ? 'text-green' : 'text-red'">
                      {{ f.trend === 'up' ? '&#9650;' : '&#9660;' }}
                    </span>
                  </td>
                  <td>
                    <span class="tag" :class="alertTagClass(f.alert)">{{ f.alert }}</span>
                  </td>
                </tr>
                <tr v-if="!factorStatus.factors.length">
                  <td colspan="5" class="text-muted text-sm" style="text-align:center;padding:12px;">
                    {{ locale === 'zh-CN' ? '无因子告警记录' : 'No factor alerts' }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <!-- Disabled factors -->
          <div v-if="factorStatus.disabled_factors.length" class="mt-2">
            <div class="text-red text-sm font-bold">{{ locale === 'zh-CN' ? '自动降权已禁用：' : 'Auto-Decay Disabled:' }}</div>
            <span v-for="name in factorStatus.disabled_factors" :key="name" class="tag tag-red mr-1">{{ name }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Kill Switch Confirmation Modal -->
    <div v-if="showKillModal" class="modal-overlay" @click.self="showKillModal = false">
      <div class="modal-content">
        <div class="modal-title text-red">{{ locale === 'zh-CN' ? '确认熔断开关' : 'CONFIRM KILL SWITCH' }}</div>
        <p class="text-secondary">
          {{ locale === 'zh-CN' ? '这将立即' : 'This will immediately' }}
          {{ killSwitchActive ? (locale === 'zh-CN' ? '解除' : 'deactivate') : (locale === 'zh-CN' ? '激活' : 'ACTIVATE') }}
          {{ locale === 'zh-CN' ? '熔断开关，' : ' the Kill Switch,' }}
          {{ killSwitchActive ? (locale === 'zh-CN' ? '恢复' : 'resuming') : (locale === 'zh-CN' ? '停止所有' : 'STOPPING ALL') }}
          {{ locale === 'zh-CN' ? '订单提交。' : ' order submissions.' }}
        </p>
        <div class="form-group mt-2">
          <label class="config-label">{{ locale === 'zh-CN' ? '原因' : 'Reason' }}</label>
          <input type="text" v-model="killReason" class="config-input" :placeholder="locale === 'zh-CN' ? '操作原因' : 'Reason for action'" />
        </div>
        <div class="modal-actions mt-3">
          <button class="btn btn-ghost" @click="showKillModal = false">{{ locale === 'zh-CN' ? '取消' : 'Cancel' }}</button>
          <button class="btn btn-danger" @click="executeKillSwitch">
            {{ killSwitchActive ? (locale === 'zh-CN' ? '解除' : 'Deactivate') : (locale === 'zh-CN' ? '激活' : 'ACTIVATE') }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'
import {
  getMonitorRiskOverview,
  getMonitorTCASummary,
  getMonitorFactorStatus,
  getMonitorCapacityGauge,
  updateMonitorConfig,
  triggerMonitorKillSwitch,
  getApiErrorMessage,
} from '../api/index.js'
import { useI18n } from '../i18n/index.js'

const emit = defineEmits(['toast'])
const { $t, locale } = useI18n()

// ── State ──
const riskOverview = reactive({
  available: null, reason: '',
  factor_exposures: {}, sector_concentration: {}, current_drawdown: 0,
  volatility: 0, var_95: 0, cvar_95: 0, risk_level: 'UNKNOWN',
  portfolio_value: 0, daily_pnl: 0, n_positions: 0,
  kill_switch: false, limits: {},
  factor_exposures_available: null, factor_exposures_reason: '',
})

const tcaSummary = reactive({
  available: null, reason: '',
  n_orders: 0, mean_is_bps: 0, mean_delay_bps: 0, mean_impact_bps: 0,
  mean_timing_bps: 0, mean_arrival_bps: 0, median_is_bps: 0,
  daily_trend: [], cost_breakdown: [], by_ticker: {},
})

const factorStatus = reactive({
  available: null, reason: '',
  factors: [], rolling_ic: {}, ic_dates: [], attribution: [],
  decay_alerts: [], disabled_factors: [],
})

const capacityGauge = reactive({
  available: null, reason: '',
  current_aum: 0, capacity_aum: 0, usage_pct: 0,
  participation_rate: 0, sharpe_at_capacity: 0, aum_curve: [],
})

const configForm = reactive({
  max_position_pct: 0.05, max_sector_pct: 0.30,
  max_drawdown_pct: 0.15, max_daily_loss_pct: 0.03,
})

// Transport-level failures per endpoint. `available === false` covers the
// backend saying "I cannot compute this"; these cover "the request itself
// failed". Both must render as —/N/A with a reason, never as measured zeros.
const loadErrors = reactive({ risk: '', tca: '', factor: '', capacity: '' })

const riskAvailable = computed(() => riskOverview.available === true)
const tcaAvailable = computed(() => tcaSummary.available === true)
const capacityAvailable = computed(() => capacityGauge.available === true)

const riskReason = computed(() => loadErrors.risk || riskOverview.reason || '')
const tcaReason = computed(() => loadErrors.tca || tcaSummary.reason || '')
const capacityReason = computed(() => loadErrors.capacity || capacityGauge.reason || '')

function shortReason(reason) {
  const text = String(reason || '').trim()
  if (!text) return ''
  return text.length > 48 ? `${text.slice(0, 45)}…` : text
}

const configSaving = ref(false)
// While the operator is editing, a 15 s poll must not overwrite the form with
// the server limits (edits used to be silently reverted mid-typing).
const configDirty = ref(false)
const configLog = ref([])
const showKillModal = ref(false)
const killSwitchActive = ref(false)
const killReason = ref('Manual activation from monitor dashboard')

// Chart refs
const riskRadarRef = ref(null)
const sectorBarRef = ref(null)
const tcaLineRef = ref(null)
const tcaBarRef = ref(null)
const factorICRef = ref(null)
const factorAttrRef = ref(null)
const capacityGaugeRef = ref(null)
const capacityCurveRef = ref(null)

let chartInstances = []
const chartsByEl = new Map()
let resizeObservers = []
let pollTimer = null

// ── Chart Styling ──
const tooltipBase = {
  backgroundColor: 'rgba(15, 24, 41, 0.95)',
  borderColor: '#1c2d4a',
  textStyle: { color: '#e8edf5', fontSize: 11 },
  extraCssText: 'backdrop-filter: blur(8px);',
}
const gridBase = { left: 48, right: 14, top: 24, bottom: 36, containLabel: false }
const colors = ['#4da6ff', '#34d399', '#f87171', '#fb923c', '#a78bfa', '#22d3ee', '#fbbf24']

function initChart(refEl, opts = {}) {
  if (!refEl) return null
  // The dashboard polls every 15s, and every render used to call
  // `echarts.init` again on the same DOM node. Each call leaves the previous
  // instance alive and un-disposed, so one open tab accumulated a new set of
  // eight charts (and their canvases) every poll. Reuse the instance for the
  // element; the DOM ref changes only when v-show/v-if recreates the subtree.
  const cached = chartsByEl.get(refEl)
  if (cached && !cached.isDisposed() && cached.getDom() === refEl) {
    return cached
  }
  const chart = echarts.init(refEl, null, { renderer: 'canvas' })
  chartsByEl.set(refEl, chart)
  chartInstances.push(chart)
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(refEl)
    resizeObservers.push(ro)
  }
  return chart
}

// When a successful poll returns an empty series, render functions that just
// `return` leave the previous chart pixels on screen as if current. Clear the
// cached instance instead of showing stale bars/lines.
function clearChart(refEl) {
  if (!refEl) return
  const cached = chartsByEl.get(refEl)
  if (cached && !cached.isDisposed()) cached.clear()
}

// ── Data Fetching ──
async function fetchAll() {
  // Settle each endpoint independently: a failing request must mark its own
  // panel unavailable (with a reason) instead of being swallowed into `{}`,
  // which left the reactive defaults (zeros) on screen as if measured.
  const [riskR, tcaR, factorR, capR] = await Promise.allSettled([
    getMonitorRiskOverview(),
    getMonitorTCASummary(),
    getMonitorFactorStatus(),
    getMonitorCapacityGauge(),
  ])

  if (riskR.status === 'fulfilled') {
    const risk = riskR.value || {}
    loadErrors.risk = ''
    // The backend emits lowercase enum values (`green`..`kill`); the colour
    // map and KPI tags are keyed uppercase, so RED/KILL books used to render
    // grey. Normalize at the boundary.
    if (risk.risk_level) risk.risk_level = String(risk.risk_level).toUpperCase()
    Object.assign(riskOverview, risk)
    // Seed the kill switch and the live limits from the shared RiskMonitor;
    // otherwise the dashboard always shows ACTIVATE and default limits even
    // after the engine tripped the switch.
    if (risk.available === true) {
      if (risk.kill_switch !== undefined) killSwitchActive.value = !!risk.kill_switch
      if (!configDirty.value && risk.limits && Object.keys(risk.limits).length) {
        configForm.max_position_pct = risk.limits.max_single_position ?? configForm.max_position_pct
        configForm.max_sector_pct = risk.limits.max_sector ?? configForm.max_sector_pct
        configForm.max_drawdown_pct = risk.limits.max_drawdown ?? configForm.max_drawdown_pct
        configForm.max_daily_loss_pct = risk.limits.max_daily_loss ?? configForm.max_daily_loss_pct
      }
    }
  } else {
    loadErrors.risk = getApiErrorMessage(riskR.reason, 'request failed')
    riskOverview.available = false
    riskOverview.reason = `${locale.value === 'zh-CN' ? '请求失败' : 'request failed'}: ${loadErrors.risk}`
    riskOverview.risk_level = 'UNKNOWN'
  }

  if (tcaR.status === 'fulfilled') {
    loadErrors.tca = ''
    Object.assign(tcaSummary, tcaR.value || {})
  } else {
    loadErrors.tca = getApiErrorMessage(tcaR.reason, 'request failed')
    tcaSummary.available = false
    tcaSummary.reason = `${locale.value === 'zh-CN' ? '请求失败' : 'request failed'}: ${loadErrors.tca}`
  }

  if (factorR.status === 'fulfilled') {
    loadErrors.factor = ''
    Object.assign(factorStatus, factorR.value || {})
  } else {
    loadErrors.factor = getApiErrorMessage(factorR.reason, 'request failed')
    factorStatus.available = false
    factorStatus.reason = `${locale.value === 'zh-CN' ? '请求失败' : 'request failed'}: ${loadErrors.factor}`
    factorStatus.factors = []
    factorStatus.decay_alerts = []
    factorStatus.disabled_factors = []
  }

  if (capR.status === 'fulfilled') {
    loadErrors.capacity = ''
    Object.assign(capacityGauge, capR.value || {})
  } else {
    loadErrors.capacity = getApiErrorMessage(capR.reason, 'request failed')
    capacityGauge.available = false
    capacityGauge.reason = `${locale.value === 'zh-CN' ? '请求失败' : 'request failed'}: ${loadErrors.capacity}`
  }

  nextTick(() => renderAllCharts())
}

// ── Chart Rendering ──
function renderAllCharts() {
  renderRiskRadar()
  renderSectorBar()
  renderTCALine()
  renderTCABar()
  renderFactorIC()
  renderFactorAttr()
  renderCapacityGauge()
  renderCapacityCurve()
  echarts.connect(chartInstances)
}

function renderRiskRadar() {
  if (riskOverview.available === false) { clearChart(riskRadarRef.value); return }
  const factors = Object.keys(riskOverview.factor_exposures)
  const values = Object.values(riskOverview.factor_exposures)
  if (riskOverview.factor_exposures_available === false) { clearChart(riskRadarRef.value); return }
  const chart = initChart(riskRadarRef.value)
  if (!chart) return
  if (!factors.length) { chart.clear(); return }
  chart.setOption({
    tooltip: { ...tooltipBase },
    radar: {
      indicator: factors.map(f => ({ name: f, max: 1, min: -1 })),
      shape: 'polygon',
      splitNumber: 4,
      axisName: { color: '#8b9dc0', fontSize: 9 },
      splitLine: { lineStyle: { color: '#1c2d4a' } },
      splitArea: { areaStyle: { color: ['rgba(15,24,41,0.3)', 'rgba(15,24,41,0.1)'] } },
      axisLine: { lineStyle: { color: '#243756' } },
    },
    series: [{
      type: 'radar',
      data: [{
        value: values,
        name: 'Exposure',
        areaStyle: { color: 'rgba(77, 166, 255, 0.15)' },
        lineStyle: { color: '#4da6ff', width: 2 },
        itemStyle: { color: '#4da6ff' },
      }],
    }],
  })
}

function renderSectorBar() {
  // Fraction weights (market_value / total_mv) scaled to percent. The old
  // xAxis max of 1 clipped every sector >= 1% at the right edge, and the value
  // labels landed outside the grid.
  if (riskOverview.available === false) { clearChart(sectorBarRef.value); return }
  const sectors = Object.entries(riskOverview.sector_concentration || {})
    .map(([name, weight]) => [name, Number(weight) * 100])
    .filter(([, pct]) => Number.isFinite(pct))
  const chart = initChart(sectorBarRef.value)
  if (!chart) return
  if (!sectors.length) { chart.clear(); return }
  chart.setOption({
    tooltip: { ...tooltipBase, trigger: 'axis', valueFormatter: v => `${Number(v).toFixed(1)}%`, axisPointer: { type: 'shadow' } },
    grid: { ...gridBase, left: 80, right: 44 },
    xAxis: {
      type: 'value',
      max: 100,
      axisLabel: { color: '#556882', formatter: v => v + '%' },
      splitLine: { lineStyle: { color: '#152035' } },
    },
    yAxis: { type: 'category', data: sectors.map(s => s[0]).reverse(), axisLabel: { color: '#8b9dc0', fontSize: 10 }, axisLine: { lineStyle: { color: '#1c2d4a' } } },
    series: [{
      type: 'bar',
      data: sectors.map(s => Number(s[1].toFixed(1))).reverse(),
      itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: '#4da6ff' }, { offset: 1, color: '#22d3ee' }]), borderRadius: [0, 3, 3, 0] },
      barWidth: 14,
      label: { show: true, position: 'right', color: '#8b9dc0', fontSize: 10, formatter: '{c}%' },
    }],
  })
}

function renderTCALine() {
  if (tcaSummary.available === false) { clearChart(tcaLineRef.value); return }
  const trend = tcaSummary.daily_trend
  const chart = initChart(tcaLineRef.value)
  if (!chart) return
  if (!trend.length) { chart.clear(); return }
  chart.setOption({
    tooltip: { ...tooltipBase, trigger: 'axis' },
    grid: gridBase,
    xAxis: { type: 'category', data: trend.map(d => d.date.slice(5)), axisLabel: { color: '#556882', fontSize: 9 }, axisLine: { lineStyle: { color: '#1c2d4a' } } },
    yAxis: { type: 'value', name: 'bps', nameTextStyle: { color: '#556882' }, axisLabel: { color: '#556882' }, splitLine: { lineStyle: { color: '#152035' } } },
    series: [{
      type: 'line',
      data: trend.map(d => d.is_bps),
      smooth: true,
      lineStyle: { color: '#fb923c', width: 2 },
      itemStyle: { color: '#fb923c' },
      areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(251,146,60,0.2)' }, { offset: 1, color: 'rgba(251,146,60,0)' }]) },
      symbol: 'none',
    }],
  })
}

function renderTCABar() {
  if (tcaSummary.available === false) { clearChart(tcaBarRef.value); return }
  const cb = tcaSummary.cost_breakdown
  const chart = initChart(tcaBarRef.value)
  if (!chart) return
  if (!cb.length) { chart.clear(); return }
  const last10 = cb.slice(-10)
  chart.setOption({
    tooltip: { ...tooltipBase, trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: gridBase,
    legend: { data: ['Delay', 'Impact', 'Timing'], textStyle: { color: '#8b9dc0', fontSize: 10 }, top: 0, right: 60 },
    xAxis: { type: 'category', data: last10.map(d => d.date.slice(5)), axisLabel: { color: '#556882', fontSize: 9 }, axisLine: { lineStyle: { color: '#1c2d4a' } } },
    yAxis: { type: 'value', name: 'bps', nameTextStyle: { color: '#556882' }, axisLabel: { color: '#556882' }, splitLine: { lineStyle: { color: '#152035' } } },
    series: [
      { name: 'Delay', type: 'bar', stack: 'cost', data: last10.map(d => d.delay), itemStyle: { color: '#f87171' }, barWidth: 16 },
      { name: 'Impact', type: 'bar', stack: 'cost', data: last10.map(d => d.impact), itemStyle: { color: '#fb923c' } },
      { name: 'Timing', type: 'bar', stack: 'cost', data: last10.map(d => d.timing), itemStyle: { color: '#fbbf24' } },
    ],
  })
}

function renderFactorIC() {
  if (factorStatus.available === false) { clearChart(factorICRef.value); return }
  const dates = factorStatus.ic_dates.map(d => d.slice(5))
  const factors = Object.keys(factorStatus.rolling_ic)
  const chart = initChart(factorICRef.value)
  if (!chart) return
  if (!factors.length || !dates.length) { chart.clear(); return }
  chart.setOption({
    tooltip: { ...tooltipBase, trigger: 'axis' },
    grid: gridBase,
    legend: { data: factors.slice(0, 5), textStyle: { color: '#8b9dc0', fontSize: 9 }, top: 0, right: 10, type: 'scroll' },
    xAxis: { type: 'category', data: dates, axisLabel: { color: '#556882', fontSize: 9 }, axisLine: { lineStyle: { color: '#1c2d4a' } } },
    yAxis: { type: 'value', name: 'IC', nameTextStyle: { color: '#556882' }, axisLabel: { color: '#556882' }, splitLine: { lineStyle: { color: '#152035' } } },
    series: factors.slice(0, 5).map((f, i) => ({
      name: f, type: 'line', data: factorStatus.rolling_ic[f],
      smooth: true, lineStyle: { color: colors[i], width: 1.5 }, itemStyle: { color: colors[i] }, symbol: 'none',
    })),
  }, { notMerge: true })
}

function renderFactorAttr() {
  if (factorStatus.available === false) { clearChart(factorAttrRef.value); return }
  const attr = factorStatus.attribution
  const chart = initChart(factorAttrRef.value)
  if (!chart) return
  if (!attr.length) { chart.clear(); return }
  const sorted = [...attr].sort((a, b) => b.contribution_bps - a.contribution_bps)
  chart.setOption({
    tooltip: { ...tooltipBase, trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { ...gridBase, left: 100 },
    xAxis: { type: 'value', name: 'bps', nameTextStyle: { color: '#556882' }, axisLabel: { color: '#556882' }, splitLine: { lineStyle: { color: '#152035' } } },
    yAxis: { type: 'category', data: sorted.map(a => a.factor).reverse(), axisLabel: { color: '#8b9dc0', fontSize: 10 }, axisLine: { lineStyle: { color: '#1c2d4a' } } },
    series: [{
      type: 'bar',
      data: sorted.map(a => a.contribution_bps).reverse(),
      itemStyle: {
        color: p => p.value >= 0 ? '#34d399' : '#f87171',
        borderRadius: p => p.value >= 0 ? [0, 3, 3, 0] : [3, 0, 0, 3],
      },
      barWidth: 14,
    }],
  })
}

function renderCapacityGauge() {
  if (capacityGauge.available === false) { clearChart(capacityGaugeRef.value); return }
  const chart = initChart(capacityGaugeRef.value)
  if (!chart) return
  chart.setOption({
    tooltip: { ...tooltipBase },
    series: [{
      type: 'gauge',
      startAngle: 220, endAngle: -40,
      min: 0, max: 100,
      progress: { show: true, width: 18, itemStyle: { color: capacityGauge.usage_pct > 80 ? '#f87171' : '#34d399' } },
      axisLine: { lineStyle: { width: 18, color: [[0.8, '#1c2d4a'], [1, '#2a1a1a']] } },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { show: false },
      pointer: { show: false },
      title: { offsetCenter: [0, '30%'], color: '#8b9dc0', fontSize: 12 },
      detail: {
        valueAnimation: true, offsetCenter: [0, '-10%'],
        fontSize: 28, fontWeight: 'bold',
        color: capacityGauge.usage_pct > 80 ? '#f87171' : '#34d399',
        formatter: v => v.toFixed(1) + '%',
      },
      data: [{ value: capacityGauge.usage_pct, name: locale.value === 'zh-CN' ? '已用容量' : 'Capacity Used' }],
    }],
  })
}

function renderCapacityCurve() {
  if (capacityGauge.available === false) { clearChart(capacityCurveRef.value); return }
  const curve = capacityGauge.aum_curve
  const chart = initChart(capacityCurveRef.value)
  if (!chart) return
  if (!curve.length) { chart.clear(); return }
  chart.setOption({
    tooltip: { ...tooltipBase, trigger: 'axis' },
    grid: gridBase,
    xAxis: {
      type: 'value', name: 'AUM',
      nameTextStyle: { color: '#556882' },
      axisLabel: { color: '#556882', fontSize: 9, formatter: v => (v / 1e6).toFixed(0) + 'M' },
      splitLine: { lineStyle: { color: '#152035' } },
    },
    yAxis: { type: 'value', name: 'Sharpe', nameTextStyle: { color: '#556882' }, axisLabel: { color: '#556882' }, splitLine: { lineStyle: { color: '#152035' } } },
    series: [
      {
        type: 'line', data: curve.map(p => [p.aum, p.sharpe]),
        smooth: true, lineStyle: { color: '#34d399', width: 2 }, itemStyle: { color: '#34d399' }, symbol: 'circle', symbolSize: 6,
        areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(52,211,153,0.15)' }, { offset: 1, color: 'rgba(52,211,153,0)' }]) },
        markLine: {
          silent: true,
          lineStyle: { color: '#f87171', type: 'dashed' },
          data: [{ yAxis: 0.5, label: { formatter: locale.value === 'zh-CN' ? '最小夏普' : 'Min Sharpe', color: '#f87171', fontSize: 10 } }],
        },
      },
    ],
  })
}

// ── Actions ──
async function submitConfig() {
  configSaving.value = true
  try {
    const result = await updateMonitorConfig(configForm)
    configLog.value.unshift({
      time: new Date().toLocaleTimeString(),
      message: (locale.value === 'zh-CN' ? '已更新：' : 'Updated: ') + result.updated.join(', '),
    })
    if (configLog.value.length > 10) configLog.value.pop()
    configDirty.value = false
    emit('toast', { message: locale.value === 'zh-CN' ? '配置已更新' : 'Config updated', type: 'success' })
  } catch (e) {
    emit('toast', { message: locale.value === 'zh-CN' ? '配置更新失败' : 'Config update failed', type: 'error' })
  } finally {
    configSaving.value = false
  }
}

function confirmKillSwitch() {
  showKillModal.value = true
}

async function executeKillSwitch() {
  try {
    const result = await triggerMonitorKillSwitch({
      activate: !killSwitchActive.value,
      reason: killReason.value,
    })
    killSwitchActive.value = result.active
    showKillModal.value = false
    configLog.value.unshift({
      time: new Date().toLocaleTimeString(),
      message: result.message,
    })
    emit('toast', { message: result.message, type: result.active ? 'error' : 'success' })
  } catch (e) {
    emit('toast', { message: locale.value === 'zh-CN' ? '熔断操作失败' : 'Kill switch action failed', type: 'error' })
  }
}

// ── Helpers ──
function isFiniteNumber(v) {
  return v != null && v !== '' && Number.isFinite(Number(v))
}

function formatNum(v, decimals = 2) {
  if (!isFiniteNumber(v)) return '—'
  const n = Number(v)
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K'
  return n.toFixed(decimals)
}

function formatPct1(v) {
  if (!isFiniteNumber(v)) return '—'
  return Number(v).toFixed(1) + '%'
}

function formatBps(v) {
  if (!isFiniteNumber(v)) return '—'
  return Number(v).toFixed(1)
}

// Factor rows are documented as {name, ic, icir, trend, alert}; an older
// shape used current_ic. Accept both, and never call .toFixed on undefined.
function factorIc(f) {
  if (f == null) return NaN
  const value = f.ic ?? f.current_ic
  return isFiniteNumber(value) ? Number(value) : NaN
}

function formatIc(v) {
  return Number.isFinite(v) ? Number(v).toFixed(4) : '—'
}

function formatIcir(v) {
  return isFiniteNumber(v) ? Number(v).toFixed(3) : '—'
}

function riskColor(level) {
  const map = { GREEN: '#34d399', YELLOW: '#fbbf24', ORANGE: '#fb923c', RED: '#f87171', KILL: '#ff0040', UNKNOWN: '#8b9dc0' }
  return map[level] || '#8b9dc0'
}

function alertTagClass(alert) {
  return { green: 'tag-green', yellow: 'tag-orange', red: 'tag-red' }[alert] || 'tag-accent'
}

// ── Lifecycle ──
let unmounted = false

onMounted(async () => {
  await fetchAll()
  if (unmounted) return
  pollTimer = setInterval(fetchAll, 15000)
})

onBeforeUnmount(() => {
  unmounted = true
  if (pollTimer) clearInterval(pollTimer)
  chartInstances.forEach(c => { try { c.dispose() } catch (_) {} })
  chartInstances = []
  chartsByEl.clear()
  resizeObservers.forEach(o => { try { o.disconnect() } catch (_) {} })
  resizeObservers = []
})
</script>

<style scoped>
.monitor-dashboard {
  padding-bottom: var(--space-8);
}

.monitor-kpi {
  margin-bottom: var(--space-4);
}

.metric-sub {
  font-size: 8px;
  color: var(--text-dim);
  margin-top: 2px;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Utility classes used by this component's template but never defined in
   style.css (pre-existing gaps: Avg IS Cost was not purple, unit/space/bold
   hints were ignored). */
.text-purple { color: #a78bfa; }
.metric-unit { font-size: 9px; color: var(--text-dim); }
.font-bold { font-weight: 700; }
.mr-1 { margin-right: 4px; }

.monitor-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-4);
}

.monitor-card {
  display: flex;
  flex-direction: column;
  min-height: 420px;
}

.monitor-card .card-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: var(--space-3) var(--space-4);
  overflow: hidden;
}

.chart-container-sm {
  width: 100%;
  height: 160px;
  flex-shrink: 0;
}

.chart-unavailable {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 120px;
  padding: var(--space-3);
  color: var(--text-muted);
  font-size: var(--text-sm);
  text-align: center;
}

.chart-title {
  font-size: var(--text-xs);
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: var(--space-1);
}

/* Config form */
.config-form {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.config-label {
  font-size: var(--text-sm);
  color: var(--text-secondary);
  min-width: 120px;
}

.config-input {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  color: var(--text-primary);
  padding: 6px 10px;
  border-radius: 4px;
  font-size: var(--text-sm);
  font-family: var(--font-mono);
  width: 100px;
  text-align: right;
}

.config-input:focus {
  outline: none;
  border-color: var(--accent);
}

.config-actions {
  margin-top: var(--space-2);
}

/* Kill Switch */
.kill-switch-section {
  border-top: 1px solid var(--border);
  padding-top: var(--space-3);
}

.kill-switch-label {
  font-size: var(--text-sm);
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: var(--space-2);
}

.kill-btn {
  width: 100%;
  padding: 10px;
  font-size: var(--text-base);
  font-weight: 700;
  letter-spacing: 1px;
}

/* Config log */
.config-log-item {
  display: flex;
  gap: var(--space-2);
  padding: 4px 0;
  border-bottom: 1px solid var(--border-subtle);
  font-size: var(--text-xs);
}

/* Modal */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: var(--bg-card);
  border: 1px solid var(--border-accent);
  border-radius: 8px;
  padding: var(--space-6);
  max-width: 400px;
  width: 90%;
}

.modal-title {
  font-size: var(--text-lg);
  font-weight: 700;
  letter-spacing: 2px;
  margin-bottom: var(--space-3);
}

.modal-actions {
  display: flex;
  gap: var(--space-3);
  justify-content: flex-end;
}

/* Responsive: 2 cols on smaller screens */
@media (max-width: 1200px) {
  .monitor-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .monitor-grid {
    grid-template-columns: 1fr;
  }
}
</style>
