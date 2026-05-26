<template>
  <div class="lt-container">
    <!-- Engine Control Bar -->
    <div class="lt-control">
      <div class="lt-control-left">
        <span class="lt-status-dot" :class="engineStatus"></span>
        <span class="lt-status-text">{{ engineStatus.toUpperCase() }}</span>
        <span v-if="stateMachine" class="lt-sm-badge" :class="'sm-' + stateMachine">{{ stateMachine }}</span>
        <span v-if="marketStatus" class="lt-market-badge">{{ marketStatus }}</span>
        <span v-if="riskLevel" class="lt-risk-badge" :class="'risk-' + riskLevel">{{ riskLevel }}</span>
        <span v-if="state.started_at" class="lt-started">{{ locale === 'zh-CN' ? '启动于: ' : 'Started: ' }}{{ fmtTime(state.started_at) }}</span>
      </div>
      <div class="lt-control-right">
        <button v-if="engineStatus !== 'running'" class="lt-btn lt-btn-start" @click="start">
          {{ locale === 'zh-CN' ? '▶ 启动交易' : '▶ Start Trading' }}
        </button>
        <button v-else class="lt-btn lt-btn-stop" @click="stop">
          {{ locale === 'zh-CN' ? '■ 停止' : '■ Stop' }}
        </button>
        <button v-if="riskLevel === 'kill'" class="lt-btn lt-btn-kill" @click="deactivateKill">
          {{ locale === 'zh-CN' ? '解除熔断' : 'Unlock Kill Switch' }}
        </button>
        <button v-else-if="engineStatus === 'running'" class="lt-btn lt-btn-kill-on" @click="activateKill">
          {{ locale === 'zh-CN' ? '紧急熔断' : 'Kill Switch' }}
        </button>
        <button class="lt-btn lt-btn-refresh" @click="refresh" :disabled="refreshing">
          {{ locale === 'zh-CN' ? '↻ 刷新' : '↻ Refresh' }}
        </button>
      </div>
    </div>

    <!-- Engine Config (when idle) -->
    <div v-if="engineStatus === 'idle' || engineStatus === 'no_engine'" class="lt-config">
      <div class="lt-config-row">
        <div class="lt-field">
          <label>{{ locale === 'zh-CN' ? '券商' : 'Broker' }}</label>
          <select v-model="config.broker">
            <option value="simulated">{{ locale === 'zh-CN' ? '模拟交易' : 'Paper Trading (Simulated)' }}</option>
            <option value="qmt">{{ locale === 'zh-CN' ? 'QMT/xtquant (实盘)' : 'QMT/xtquant (Live)' }}</option>
          </select>
        </div>
        <div class="lt-field">
          <label>{{ locale === 'zh-CN' ? '初始资金' : 'Initial Cash' }}</label>
          <input v-model.number="config.initial_cash" type="number" step="100000" />
        </div>
        <div class="lt-field">
          <label>{{ locale === 'zh-CN' ? '持仓数量' : 'Portfolio Size' }}</label>
          <input v-model.number="config.n_stocks" type="number" step="10" />
        </div>
        <div class="lt-field">
          <label>{{ locale === 'zh-CN' ? '再平衡(秒)' : 'Rebalance (sec)' }}</label>
          <input v-model.number="config.rebalance_interval" type="number" step="60" />
        </div>
      </div>
      <div v-if="config.broker === 'qmt'" class="lt-config-row">
        <div class="lt-field lt-field-wide">
          <label>{{ locale === 'zh-CN' ? 'QMT路径' : 'QMT Path' }}</label>
          <input v-model="config.qmt_path" placeholder="C:\国金证券QMT\UserData_mini" />
        </div>
        <div class="lt-field">
          <label>{{ locale === 'zh-CN' ? '账户ID' : 'Account ID' }}</label>
          <input v-model="config.account_id" :placeholder="locale === 'zh-CN' ? '请输入券商账户ID' : 'Your broker account ID'" />
        </div>
      </div>
    </div>

    <!-- Live Metrics (when running) -->
    <div v-if="engineStatus === 'running' || engineStatus === 'error'" class="lt-metrics">
      <div class="lt-metric">
        <div class="lt-metric-label">{{ locale === 'zh-CN' ? '组合价值' : 'Portfolio Value' }}</div>
        <div class="lt-metric-val">{{ fmtMoney(state.portfolio_value) }}</div>
      </div>
      <div class="lt-metric">
        <div class="lt-metric-label">{{ locale === 'zh-CN' ? '现金' : 'Cash' }}</div>
        <div class="lt-metric-val">{{ fmtMoney(state.cash) }}</div>
      </div>
      <div class="lt-metric">
        <div class="lt-metric-label">{{ locale === 'zh-CN' ? '总盈亏' : 'Total P&L' }}</div>
        <div class="lt-metric-val" :class="state.total_pnl >= 0 ? 'pos' : 'neg'">
          {{ fmtMoney(state.total_pnl) }} ({{ (state.total_pnl_pct * 100).toFixed(2) }}%)
        </div>
      </div>
      <div class="lt-metric">
        <div class="lt-metric-label">{{ locale === 'zh-CN' ? '持仓数' : 'Positions' }}</div>
        <div class="lt-metric-val">{{ state.n_positions }}</div>
      </div>
      <div class="lt-metric">
        <div class="lt-metric-label">{{ locale === 'zh-CN' ? '运行周期' : 'Cycles' }}</div>
        <div class="lt-metric-val">{{ state.n_cycles }}</div>
      </div>
      <div class="lt-metric">
        <div class="lt-metric-label">{{ locale === 'zh-CN' ? '交易次数' : 'Trades' }}</div>
        <div class="lt-metric-val">{{ state.total_trades }}</div>
      </div>
    </div>

    <!-- Positions Table -->
    <div v-if="positions.length" class="lt-section">
      <div class="lt-section-title">{{ locale === 'zh-CN' ? '实时持仓' : 'Live Positions' }}</div>
      <table class="lt-table">
        <thead>
          <tr>
            <th>{{ locale === 'zh-CN' ? '代码' : 'Code' }}</th>
            <th>{{ locale === 'zh-CN' ? '名称' : 'Name' }}</th>
            <th>{{ locale === 'zh-CN' ? '数量' : 'Qty' }}</th>
            <th>{{ locale === 'zh-CN' ? '均价' : 'Avg Cost' }}</th>
            <th>{{ locale === 'zh-CN' ? '现价' : 'Price' }}</th>
            <th>{{ locale === 'zh-CN' ? '市值' : 'Market Value' }}</th>
            <th>{{ locale === 'zh-CN' ? '盈亏' : 'P&L' }}</th>
            <th>{{ locale === 'zh-CN' ? '盈亏%' : 'P&L %' }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="p in positions" :key="p.code">
            <td class="lt-code">{{ p.code }}</td>
            <td>{{ p.name }}</td>
            <td class="mono">{{ p.quantity }}</td>
            <td class="mono">{{ p.avg_cost?.toFixed(2) }}</td>
            <td class="mono">{{ p.current_price?.toFixed(2) }}</td>
            <td class="mono">{{ fmtMoney(p.market_value) }}</td>
            <td class="mono" :class="p.unrealized_pnl >= 0 ? 'pos' : 'neg'">
              {{ fmtMoney(p.unrealized_pnl) }}
            </td>
            <td class="mono" :class="p.unrealized_pnl_pct >= 0 ? 'pos' : 'neg'">
              {{ (p.unrealized_pnl_pct * 100).toFixed(2) }}%
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Recent Cycles -->
    <div v-if="cycles.length" class="lt-section">
      <div class="lt-section-title">{{ locale === 'zh-CN' ? '近期交易周期' : 'Recent Trading Cycles' }}</div>
      <div class="lt-cycles">
        <div v-for="c in cycles" :key="c.cycle_id" class="lt-cycle">
          <div class="lt-cycle-head">
            <span class="lt-cycle-id">#{{ c.cycle_id }}</span>
            <span class="lt-cycle-time">{{ fmtTime(c.timestamp) }}</span>
            <span class="lt-cycle-signals">{{ c.signals?.length || 0 }}{{ locale === 'zh-CN' ? '个信号' : ' signals' }}</span>
            <span class="lt-cycle-orders">{{ c.orders?.length || 0 }}{{ locale === 'zh-CN' ? '个订单' : ' orders' }}</span>
            <span class="lt-cycle-eq">{{ fmtMoney(c.portfolio_value) }}</span>
          </div>
          <div v-if="c.orders?.length" class="lt-cycle-orders-detail">
            <div v-for="o in c.orders" :key="o.order_id" class="lt-order"
              :class="o.side === 'buy' ? 'lt-order-buy' : 'lt-order-sell'">
              <span class="lt-order-side">{{ o.side?.toUpperCase() }}</span>
              <span class="lt-order-code">{{ o.code }}</span>
              <span class="lt-order-qty">{{ o.quantity }} @ {{ o.filled_price?.toFixed(2) }}</span>
              <span class="lt-order-status">{{ o.status }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Risk Monitor -->
    <div v-if="riskStatus" class="lt-section">
      <div class="lt-section-title">
        {{ locale === 'zh-CN' ? '风险监控' : 'Risk Monitor' }}
        <span class="lt-risk-level" :class="'risk-' + (riskStatus.risk_level || 'green')">
          {{ (riskStatus.risk_level || 'green').toUpperCase() }}
        </span>
      </div>
      <div class="lt-risk-grid">
        <div class="lt-risk-item">
          <span class="lt-risk-label">{{ locale === 'zh-CN' ? '紧急熔断' : 'Kill Switch' }}</span>
          <span :class="riskStatus.kill_switch_active ? 'neg' : 'pos'">
            {{ riskStatus.kill_switch_active ? (locale === 'zh-CN' ? '已激活' : 'ACTIVE') : (locale === 'zh-CN' ? '关闭' : 'OFF') }}
          </span>
        </div>
        <div class="lt-risk-item">
          <span class="lt-risk-label">{{ locale === 'zh-CN' ? '组合价值' : 'Portfolio Value' }}</span>
          <span class="mono">{{ fmtMoney(riskStatus.portfolio_value) }}</span>
        </div>
        <div class="lt-risk-item">
          <span class="lt-risk-label">{{ locale === 'zh-CN' ? '峰值' : 'Peak Value' }}</span>
          <span class="mono">{{ fmtMoney(riskStatus.peak_value) }}</span>
        </div>
        <div class="lt-risk-item">
          <span class="lt-risk-label">{{ locale === 'zh-CN' ? '日盈亏' : 'Daily P&L' }}</span>
          <span class="mono" :class="(riskStatus.daily_pnl || 0) >= 0 ? 'pos' : 'neg'">
            {{ fmtMoney(riskStatus.daily_pnl) }}
          </span>
        </div>
        <div class="lt-risk-item">
          <span class="lt-risk-label">{{ locale === 'zh-CN' ? '最大回撤限额' : 'Max Drawdown Limit' }}</span>
          <span class="mono">{{ ((riskStatus.limits?.max_drawdown_pct || 0.15) * 100).toFixed(0) }}%</span>
        </div>
        <div class="lt-risk-item">
          <span class="lt-risk-label">{{ locale === 'zh-CN' ? '仓位上限' : 'Max Position Limit' }}</span>
          <span class="mono">{{ ((riskStatus.limits?.max_single_position_pct || 0.05) * 100).toFixed(0) }}%</span>
        </div>
      </div>
      <div v-if="riskStatus.breaches?.length" class="lt-breaches">
        <div v-for="(b, i) in riskStatus.breaches.slice(-5)" :key="i" class="lt-breach"
          :class="'breach-' + (b.severity || 'yellow')">
          <span class="lt-breach-type">{{ b.breach_type }}</span>
          <span class="lt-breach-msg">{{ b.message }}</span>
          <span class="lt-breach-time">{{ fmtTime(b.timestamp) }}</span>
        </div>
      </div>
    </div>

    <!-- State Machine + Audit + Events (3-column layout) -->
    <div class="lt-triple">
      <!-- State Machine History -->
      <div class="lt-section lt-triple-col">
        <div class="lt-section-title">{{ locale === 'zh-CN' ? '状态机' : 'State Machine' }}</div>
        <div class="lt-sm-history">
          <div v-for="(t, i) in stateHistory" :key="i" class="lt-sm-transition">
            <span class="lt-sm-from" :class="'sm-' + t.from">{{ t.from }}</span>
            <span class="lt-sm-arrow">→</span>
            <span class="lt-sm-to" :class="'sm-' + t.to">{{ t.to }}</span>
            <span class="lt-sm-reason">{{ t.reason }}</span>
            <span class="lt-sm-time">{{ fmtTime(t.time) }}</span>
          </div>
          <div v-if="!stateHistory.length" class="lt-empty">{{ locale === 'zh-CN' ? '暂无状态转换' : 'No transitions yet' }}</div>
        </div>
      </div>

      <!-- Audit Log -->
      <div class="lt-section lt-triple-col">
        <div class="lt-section-title">{{ locale === 'zh-CN' ? '审计日志' : 'Audit Trail' }}</div>
        <div class="lt-audit-list">
          <div v-for="(e, i) in auditEvents" :key="i" class="lt-audit-item">
            <span class="lt-audit-action" :class="'audit-' + (e.data?.action || '')">
              {{ e.data?.action || 'unknown' }}
            </span>
            <span class="lt-audit-comp">{{ e.data?.component || '' }}</span>
            <span class="lt-audit-reason">{{ e.data?.reason || '' }}</span>
          </div>
          <div v-if="!auditEvents.length" class="lt-empty">{{ locale === 'zh-CN' ? '暂无审计事件' : 'No audit events' }}</div>
        </div>
      </div>

      <!-- Event Stream -->
      <div class="lt-section lt-triple-col">
        <div class="lt-section-title">
          {{ locale === 'zh-CN' ? '事件流' : 'Event Stream' }}
          <span class="lt-event-count">{{ wsEvents.length }}</span>
        </div>
        <div class="lt-event-list">
          <div v-for="(e, i) in wsEvents" :key="i" class="lt-event-item">
            <span class="lt-event-topic">{{ e.event }}</span>
            <span class="lt-event-data">{{ summarizeEvent(e.data) }}</span>
          </div>
          <div v-if="!wsEvents.length" class="lt-empty">{{ locale === 'zh-CN' ? '暂无事件' : 'No events yet' }}</div>
        </div>
      </div>
    </div>

    <!-- Market Snapshot -->
    <div class="lt-section">
      <div class="lt-section-title">
        {{ locale === 'zh-CN' ? '实时行情' : 'Real-Time Market' }}
        <span class="lt-market-time" v-if="marketTime">{{ marketTime }}</span>
      </div>
      <div v-if="marketStocks.length" class="lt-market-grid">
        <div v-for="s in marketStocks" :key="s['代码']" class="lt-stock"
          :class="(s['涨跌幅'] || 0) >= 0 ? 'lt-stock-up' : 'lt-stock-down'">
          <div class="lt-stock-name">{{ s['名称'] }}</div>
          <div class="lt-stock-price">{{ (s['最新价'] || 0).toFixed(2) }}</div>
          <div class="lt-stock-change">{{ (s['涨跌幅'] || 0).toFixed(2) }}%</div>
        </div>
      </div>
      <div v-else class="lt-empty">{{ locale === 'zh-CN' ? '加载行情数据...' : 'Loading market data...' }}</div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onBeforeUnmount } from 'vue'
import { useI18n } from '../i18n/index.js'
import {
  startTrading, stopTrading, getTradingStatus,
  getTradingPositions, getTradingAccount, getTradingCycles,
  runTradingCycle, getMarketSnapshot,
  getCoreState, getCoreAudit, getCoreRisk, toggleCoreKillSwitch,
  createStatusSocket
} from '../api/index.js'

const { $t, locale } = useI18n()

const emit = defineEmits(['toast'])

const engineStatus = ref('no_engine')
const state = reactive({
  started_at: '', n_cycles: 0, total_trades: 0, total_signals: 0,
  portfolio_value: 0, cash: 0, n_positions: 0, total_pnl: 0, total_pnl_pct: 0,
  state_machine: '', market_status: '',
})
const positions = ref([])
const cycles = ref([])
const marketStocks = ref([])
const marketTime = ref('')
const refreshing = ref(false)

// Core architecture data
const stateMachine = ref('')
const marketStatus = ref('')
const riskLevel = ref('')
const riskStatus = ref(null)
const stateHistory = ref([])
const auditEvents = ref([])
const wsEvents = ref([])

const config = reactive({
  broker: 'simulated',
  initial_cash: 1000000,
  n_stocks: 30,
  rebalance_interval: 300,
  qmt_path: '',
  account_id: '',
})

let pollTimer = null
let ws = null

async function start() {
  try {
    await startTrading(config)
    engineStatus.value = 'running'
    emit('toast', { message: locale.value === 'zh-CN' ? '实盘交易引擎已启动' : 'Live trading engine started', type: 'success' })
    startPolling()
    connectWS()
  } catch (e) {
    emit('toast', { message: locale.value === 'zh-CN' ? '启动失败: ' + (e.response?.data?.detail || e.message) : 'Failed to start: ' + (e.response?.data?.detail || e.message), type: 'error' })
  }
}

async function stop() {
  try {
    await stopTrading()
    engineStatus.value = 'idle'
    emit('toast', { message: locale.value === 'zh-CN' ? '引擎已停止' : 'Engine stopped', type: 'success' })
    stopPolling()
  } catch (e) {
    emit('toast', { message: locale.value === 'zh-CN' ? '停止失败' : 'Failed to stop', type: 'error' })
  }
}

async function activateKill() {
  try {
    await toggleCoreKillSwitch(true, 'Manual UI activation')
    emit('toast', { message: locale.value === 'zh-CN' ? '紧急熔断已激活 — 所有订单已被阻止' : 'Kill switch activated — all orders blocked', type: 'error' })
    await fetchRisk()
  } catch (e) {
    emit('toast', { message: locale.value === 'zh-CN' ? '激活熔断失败' : 'Failed to activate kill switch', type: 'error' })
  }
}

async function deactivateKill() {
  try {
    await toggleCoreKillSwitch(false)
    emit('toast', { message: locale.value === 'zh-CN' ? '紧急熔断已解除' : 'Kill switch deactivated', type: 'success' })
    await fetchRisk()
  } catch (e) {
    emit('toast', { message: locale.value === 'zh-CN' ? '解除熔断失败' : 'Failed to deactivate kill switch', type: 'error' })
  }
}

async function refresh() {
  refreshing.value = true
  try {
    await Promise.all([fetchStatus(), fetchPositions(), fetchCycles(), fetchMarket(), fetchState(), fetchAudit(), fetchRisk()])
  } finally {
    refreshing.value = false
  }
}

async function fetchStatus() {
  try {
    const s = await getTradingStatus()
    if (s.status === 'no_engine') {
      engineStatus.value = 'no_engine'
    } else {
      engineStatus.value = s.status
      Object.assign(state, s)
      stateMachine.value = s.state_machine || ''
      marketStatus.value = s.market_status || ''
      if (s.risk) {
        riskLevel.value = s.risk.risk_level || 'green'
      }
    }
  } catch (_) {}
}

async function fetchPositions() {
  try {
    const r = await getTradingPositions()
    positions.value = r.positions || []
  } catch (_) {}
}

async function fetchCycles() {
  try {
    const r = await getTradingCycles()
    cycles.value = (r.cycles || []).reverse()
  } catch (_) {}
}

async function fetchMarket() {
  try {
    const r = await getMarketSnapshot()
    marketStocks.value = (r.stocks || []).slice(0, 30)
    marketTime.value = r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : ''
  } catch (_) {}
}

async function fetchState() {
  try {
    const r = await getCoreState()
    stateHistory.value = (r.history || []).reverse()
    stateMachine.value = r.current_state || stateMachine.value
  } catch (_) {}
}

async function fetchAudit() {
  try {
    const r = await getCoreAudit('', 30)
    auditEvents.value = r.events || []
  } catch (_) {}
}

async function fetchRisk() {
  try {
    const r = await getCoreRisk()
    riskStatus.value = r
    riskLevel.value = r.risk_level || 'green'
  } catch (_) {}
}

function connectWS() {
  if (ws) return
  try {
    ws = createStatusSocket(
      (data) => {
        if (data.type === 'event') {
          wsEvents.value.unshift(data)
          if (wsEvents.value.length > 100) wsEvents.value = wsEvents.value.slice(0, 50)
        }
      },
      () => {}
    )
  } catch (_) {}
}

function startPolling() {
  stopPolling()
  pollTimer = setInterval(async () => {
    await Promise.all([fetchStatus(), fetchPositions(), fetchAudit(), fetchRisk()])
  }, 5000)
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

function summarizeEvent(data) {
  if (!data) return ''
  if (data.code) return `${data.code} ${data.side || ''} ${data.quantity || ''}`
  if (data.equity) return `equity=${fmtMoney(data.equity)}`
  if (data.error) return data.error
  return JSON.stringify(data).slice(0, 60)
}

function fmtMoney(v) {
  if (!v && v !== 0) return '--'
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(2) + 'M'
  if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + 'K'
  return v.toFixed(0)
}

function fmtTime(t) {
  if (!t) return ''
  return new Date(t).toLocaleTimeString()
}

onMounted(async () => {
  await fetchStatus()
  await fetchMarket()
  await fetchState()
  await fetchAudit()
  await fetchRisk()
  if (engineStatus.value === 'running') {
    startPolling()
    connectWS()
  }
})

onBeforeUnmount(() => {
  stopPolling()
  if (ws) { ws.close(); ws = null }
})
</script>

<style scoped>
.lt-container { display: flex; flex-direction: column; gap: 12px; font-size: 11px; }

.lt-control {
  display: flex; justify-content: space-between; align-items: center;
  background: var(--bg-card, #111827); border: 1px solid var(--border, #1e2a3a);
  border-radius: 6px; padding: 10px 14px;
}
.lt-control-left { display: flex; align-items: center; gap: 8px; }
.lt-control-right { display: flex; gap: 6px; }

.lt-status-dot {
  width: 8px; height: 8px; border-radius: 50%;
}
.lt-status-dot.running { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.5); animation: pulse 2s infinite; }
.lt-status-dot.idle, .lt-status-dot.no_engine { background: #6b7a8d; }
.lt-status-dot.error { background: #ef4444; }
.lt-status-dot.stopping { background: #fbbf24; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }

.lt-status-text { font-weight: 700; color: #e6edf3; letter-spacing: 0.5px; }
.lt-started { color: #6b7a8d; font-size: 10px; }

/* Badges */
.lt-sm-badge, .lt-market-badge, .lt-risk-badge {
  font-size: 9px; font-weight: 700; padding: 2px 8px; border-radius: 3px;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.sm-init { background: rgba(107,122,141,0.2); color: #6b7a8d; }
.sm-ready { background: rgba(77,166,255,0.15); color: #4da6ff; }
.sm-pre_market { background: rgba(251,191,36,0.15); color: #fbbf24; }
.sm-trading { background: rgba(34,197,94,0.15); color: #22c55e; }
.sm-rebalancing { background: rgba(168,85,247,0.15); color: #a855f7; }
.sm-post_market { background: rgba(107,122,141,0.2); color: #6b7a8d; }
.sm-halted { background: rgba(239,68,68,0.2); color: #ef4444; }
.sm-error { background: rgba(239,68,68,0.2); color: #ef4444; }
.lt-market-badge { background: rgba(77,166,255,0.1); color: #4da6ff; }
.risk-green { background: rgba(34,197,94,0.15); color: #22c55e; }
.risk-yellow { background: rgba(251,191,36,0.15); color: #fbbf24; }
.risk-orange { background: rgba(249,115,22,0.15); color: #f97316; }
.risk-red { background: rgba(239,68,68,0.2); color: #ef4444; }
.risk-kill { background: rgba(239,68,68,0.3); color: #ef4444; animation: pulse 1s infinite; }

.lt-btn {
  padding: 5px 14px; border-radius: 4px; border: 1px solid; font-size: 11px;
  font-weight: 600; cursor: pointer; transition: all 0.15s;
}
.lt-btn-start { border-color: #22c55e; color: #22c55e; background: transparent; }
.lt-btn-start:hover { background: rgba(34,197,94,0.1); }
.lt-btn-stop { border-color: #ef4444; color: #ef4444; background: transparent; }
.lt-btn-stop:hover { background: rgba(239,68,68,0.1); }
.lt-btn-kill { border-color: #22c55e; color: #22c55e; background: transparent; }
.lt-btn-kill:hover { background: rgba(34,197,94,0.1); }
.lt-btn-kill-on { border-color: #ef4444; color: #ef4444; background: transparent; }
.lt-btn-kill-on:hover { background: rgba(239,68,68,0.15); }
.lt-btn-refresh { border-color: #4da6ff; color: #4da6ff; background: transparent; }
.lt-btn-refresh:disabled { opacity: 0.4; }

.lt-config {
  background: var(--bg-card, #111827); border: 1px solid var(--border, #1e2a3a);
  border-radius: 6px; padding: 12px;
}
.lt-config-row { display: flex; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }
.lt-config-row:last-child { margin-bottom: 0; }
.lt-field { flex: 1; min-width: 120px; }
.lt-field-wide { flex: 2; }
.lt-field label { display: block; font-size: 9px; font-weight: 700; color: #6b7a8d; text-transform: uppercase; margin-bottom: 4px; }
.lt-field input, .lt-field select {
  width: 100%; padding: 6px 8px; background: var(--bg-base, #0a0e17);
  border: 1px solid var(--border, #1e2a3a); border-radius: 4px;
  color: #e6edf3; font-size: 11px; font-family: inherit;
}

.lt-metrics {
  display: flex; gap: 6px; flex-wrap: wrap;
}
.lt-metric {
  flex: 1; min-width: 100px; background: var(--bg-card, #111827);
  border: 1px solid var(--border, #1e2a3a); border-radius: 6px;
  padding: 8px 10px; text-align: center;
}
.lt-metric-label { font-size: 8px; font-weight: 700; color: #6b7a8d; text-transform: uppercase; }
.lt-metric-val { font-size: 15px; font-weight: 700; font-family: monospace; color: #e6edf3; margin-top: 2px; }
.pos { color: #22c55e; }
.neg { color: #ef4444; }

.lt-section {
  background: var(--bg-card, #111827); border: 1px solid var(--border, #1e2a3a);
  border-radius: 6px; padding: 10px;
}
.lt-section-title {
  font-size: 9px; font-weight: 700; color: #6b7a8d; text-transform: uppercase;
  letter-spacing: 0.5px; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
}
.lt-section-title::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: #4da6ff; }
.lt-market-time { margin-left: auto; font-size: 9px; color: #6b7a8d; font-weight: 400; }

.lt-table { width: 100%; border-collapse: collapse; }
.lt-table th {
  padding: 4px 8px; font-size: 8px; font-weight: 700; color: #6b7a8d;
  text-transform: uppercase; text-align: left; border-bottom: 1px solid #1e2a3a;
}
.lt-table td { padding: 4px 8px; border-bottom: 1px solid #161b22; color: #c9d1d9; }
.lt-table tr:hover td { background: rgba(77,166,255,0.04); }
.lt-code { font-weight: 600; color: #4da6ff; }
.mono { font-family: monospace; font-size: 10px; }

.lt-cycles { display: flex; flex-direction: column; gap: 6px; max-height: 300px; overflow-y: auto; }
.lt-cycle {
  border: 1px solid #1e2a3a; border-radius: 4px; padding: 6px 8px;
}
.lt-cycle-head { display: flex; gap: 8px; align-items: center; font-size: 10px; }
.lt-cycle-id { font-weight: 700; color: #4da6ff; }
.lt-cycle-time { color: #6b7a8d; }
.lt-cycle-signals { color: #fbbf24; }
.lt-cycle-orders { color: #22c55e; }
.lt-cycle-eq { margin-left: auto; font-family: monospace; color: #e6edf3; }

.lt-cycle-orders-detail { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.lt-order {
  display: flex; gap: 4px; align-items: center; font-size: 9px;
  padding: 2px 6px; border-radius: 3px;
}
.lt-order-buy { background: rgba(34,197,94,0.1); color: #22c55e; }
.lt-order-sell { background: rgba(239,68,68,0.1); color: #ef4444; }
.lt-order-side { font-weight: 700; }
.lt-order-code { color: #e6edf3; }
.lt-order-status { color: #6b7a8d; }

/* Risk Monitor */
.lt-risk-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 8px; }
.lt-risk-item { display: flex; justify-content: space-between; align-items: center; font-size: 10px; padding: 4px 6px; border: 1px solid #1e2a3a; border-radius: 3px; }
.lt-risk-label { color: #6b7a8d; }
.lt-risk-level { margin-left: auto; font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 3px; }
.lt-breaches { display: flex; flex-direction: column; gap: 4px; }
.lt-breach { display: flex; gap: 8px; align-items: center; font-size: 9px; padding: 4px 8px; border-radius: 3px; }
.breach-yellow { background: rgba(251,191,36,0.08); border-left: 3px solid #fbbf24; }
.breach-orange { background: rgba(249,115,22,0.08); border-left: 3px solid #f97316; }
.breach-red { background: rgba(239,68,68,0.08); border-left: 3px solid #ef4444; }
.breach-kill { background: rgba(239,68,68,0.15); border-left: 3px solid #ef4444; }
.lt-breach-type { font-weight: 700; color: #fbbf24; min-width: 80px; }
.lt-breach-msg { color: #c9d1d9; flex: 1; }
.lt-breach-time { color: #6b7a8d; font-size: 8px; }

/* Triple column: State Machine + Audit + Events */
.lt-triple { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
.lt-triple-col { max-height: 250px; overflow-y: auto; }

.lt-sm-history { display: flex; flex-direction: column; gap: 3px; }
.lt-sm-transition { display: flex; gap: 4px; align-items: center; font-size: 9px; }
.lt-sm-from, .lt-sm-to { padding: 1px 5px; border-radius: 2px; font-weight: 600; font-size: 8px; }
.lt-sm-arrow { color: #6b7a8d; }
.lt-sm-reason { color: #6b7a8d; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lt-sm-time { color: #6b7a8d; font-size: 8px; }

.lt-audit-list { display: flex; flex-direction: column; gap: 3px; }
.lt-audit-item { display: flex; gap: 6px; align-items: center; font-size: 9px; }
.lt-audit-action { padding: 1px 5px; border-radius: 2px; font-weight: 600; font-size: 8px; background: rgba(77,166,255,0.1); color: #4da6ff; }
.audit-order_filled { background: rgba(34,197,94,0.1); color: #22c55e; }
.audit-order_rejected { background: rgba(239,68,68,0.1); color: #ef4444; }
.audit-signal_generated { background: rgba(251,191,36,0.1); color: #fbbf24; }
.audit-engine_start { background: rgba(34,197,94,0.1); color: #22c55e; }
.audit-engine_stop { background: rgba(239,68,68,0.1); color: #ef4444; }
.audit-risk_breach { background: rgba(239,68,68,0.15); color: #ef4444; }
.lt-audit-comp { color: #6b7a8d; min-width: 50px; }
.lt-audit-reason { color: #c9d1d9; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.lt-event-list { display: flex; flex-direction: column; gap: 3px; }
.lt-event-item { display: flex; gap: 6px; align-items: center; font-size: 9px; }
.lt-event-topic { font-weight: 600; color: #a855f7; min-width: 100px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lt-event-data { color: #c9d1d9; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lt-event-count { margin-left: auto; font-size: 8px; color: #6b7a8d; }

/* Market */
.lt-market-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 4px; }
.lt-stock {
  padding: 6px 8px; border-radius: 4px; border: 1px solid #1e2a3a;
}
.lt-stock-up { border-left: 3px solid #22c55e; }
.lt-stock-down { border-left: 3px solid #ef4444; }
.lt-stock-name { font-size: 9px; color: #6b7a8d; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lt-stock-price { font-size: 12px; font-weight: 700; font-family: monospace; color: #e6edf3; }
.lt-stock-change { font-size: 10px; font-family: monospace; }
.lt-stock-up .lt-stock-change { color: #22c55e; }
.lt-stock-down .lt-stock-change { color: #ef4444; }

.lt-empty { text-align: center; color: #6b7a8d; padding: 16px; }
</style>
