<template>
  <div class="rm-view">
    <div class="rm-header">
      <div class="rm-title">
        <span :class="['rm-dot', `rm-${status?.risk_level || 'green'}`]"></span>
        {{ locale === 'zh-CN' ? '风险监控' : 'RISK MONITOR' }}
      </div>
      <div class="rm-actions">
        <button
          :class="['btn', 'btn-sm', status?.kill_switch ? 'btn-danger' : 'btn-secondary']"
          @click="toggleKill"
        >
          {{ status?.kill_switch ? (locale === 'zh-CN' ? '解除熔断开关' : 'Deactivate Kill Switch') : (locale === 'zh-CN' ? '熔断开关' : 'Kill Switch') }}
        </button>
        <button class="btn btn-sm btn-secondary" @click="refresh" :disabled="loading">
          {{ loading ? '...' : (locale === 'zh-CN' ? '刷新' : 'Refresh') }}
        </button>
      </div>
    </div>

    <template v-if="status">
      <!-- Risk Level Banner -->
      <div :class="['rm-banner', `rm-banner-${status.risk_level}`]">
        <span class="rm-banner-icon">
          {{ status.risk_level === 'green' ? '&#10003;' : status.risk_level === 'kill' ? '&#9888;' : '&#9679;' }}
        </span>
        <span class="rm-banner-text">
          {{ status.risk_level === 'green' ? (locale === 'zh-CN' ? '一切正常' : 'ALL CLEAR') : status.risk_level === 'kill' ? (locale === 'zh-CN' ? '熔断开关已触发' : 'KILL SWITCH ACTIVE') : (locale === 'zh-CN' ? '检测到风险违规' : 'RISK BREACH DETECTED') }}
        </span>
        <span class="rm-banner-detail" v-if="status.n_breaches_today">
          {{ locale === 'zh-CN' ? `今日${status.n_breaches_today}次违规` : `${status.n_breaches_today} breach(es) today` }}
        </span>
      </div>

      <!-- Risk Metrics -->
      <div class="rm-metrics">
        <div class="rm-metric">
          <div class="rm-metric-label">{{ locale === 'zh-CN' ? '组合价值' : 'Portfolio Value' }}</div>
          <div class="rm-metric-value">{{ formatNum(status.portfolio_value) }}</div>
        </div>
        <div class="rm-metric">
          <div class="rm-metric-label">{{ locale === 'zh-CN' ? '峰值' : 'Peak Value' }}</div>
          <div class="rm-metric-value">{{ formatNum(status.peak_value) }}</div>
        </div>
        <div class="rm-metric">
          <div class="rm-metric-label">{{ locale === 'zh-CN' ? '当前回撤' : 'Current DD' }}</div>
          <div :class="['rm-metric-value', status.current_drawdown > 0.1 ? 'rm-neg' : '']">
            {{ (status.current_drawdown * 100)?.toFixed(2) }}%
          </div>
        </div>
        <div class="rm-metric">
          <div class="rm-metric-label">{{ locale === 'zh-CN' ? '日盈亏' : 'Daily P&L' }}</div>
          <div :class="['rm-metric-value', status.daily_pnl >= 0 ? 'rm-pos' : 'rm-neg']">
            {{ status.daily_pnl >= 0 ? '+' : '' }}{{ formatNum(status.daily_pnl) }}
          </div>
        </div>
        <div class="rm-metric">
          <div class="rm-metric-label">{{ locale === 'zh-CN' ? '日盈亏%' : 'Daily P&L %' }}</div>
          <div :class="['rm-metric-value', status.daily_pnl_pct >= 0 ? 'rm-pos' : 'rm-neg']">
            {{ (status.daily_pnl_pct * 100)?.toFixed(2) }}%
          </div>
        </div>
        <div class="rm-metric">
          <div class="rm-metric-label">{{ locale === 'zh-CN' ? '持仓数' : 'Positions' }}</div>
          <div class="rm-metric-value">{{ status.n_positions }}</div>
        </div>
      </div>

      <!-- Limits -->
      <div class="rm-limits">
        <div class="rm-limit-title">{{ locale === 'zh-CN' ? '风险限额' : 'RISK LIMITS' }}</div>
        <div class="rm-limit-row" v-for="(val, key) in status.limits" :key="key">
          <span class="rm-limit-name">{{ formatLimitName(key) }}</span>
          <div class="rm-limit-bar-wrap">
            <div class="rm-limit-bar-bg"></div>
            <div class="rm-limit-bar-fill" :style="{ width: getLimitFill(key, val) + '%' }"></div>
          </div>
          <span class="rm-limit-value">{{ (val * 100).toFixed(1) }}%</span>
        </div>
      </div>

      <!-- Recent Breaches -->
      <div class="rm-breaches" v-if="status.recent_breaches?.length">
        <div class="rm-breach-title">{{ locale === 'zh-CN' ? '近期违规' : 'RECENT BREACHES' }}</div>
        <div v-for="b in status.recent_breaches" :key="b.timestamp" :class="['rm-breach-item', `rm-breach-${b.severity}`]">
          <span class="rm-breach-severity">{{ b.severity }}</span>
          <span class="rm-breach-msg">{{ b.message }}</span>
          <span class="rm-breach-time">{{ b.timestamp?.substring(11, 19) }}</span>
        </div>
      </div>
    </template>

    <div v-if="!status && !loading" class="rm-empty">
      <div class="rm-empty-icon">&#9888;</div>
      <h3>{{ locale === 'zh-CN' ? '风险监控' : 'Risk Monitor' }}</h3>
      <p>{{ locale === 'zh-CN' ? '实时风险监控，包括仓位限制、亏损限制、回撤熔断和紧急熔断开关。' : 'Real-time risk monitoring with position limits, loss limits, drawdown circuit breakers, and emergency kill switch.' }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getRiskStatus, toggleKillSwitch } from '../api/index.js'
import { useI18n } from '../i18n/index.js'

const { $t, locale } = useI18n()

const emit = defineEmits(['toast'])
const loading = ref(false)
const status = ref(null)

const limitNameMap = {
  max_position_pct: { en: 'Position', zh: '仓位' },
  max_drawdown_pct: { en: 'Drawdown', zh: '回撤' },
  max_kill_drawdown_pct: { en: 'Kill Drawdown', zh: '熔断回撤' },
  max_daily_loss_pct: { en: 'Daily Loss', zh: '日亏损' },
  max_leverage: { en: 'Leverage', zh: '杠杆' },
  max_sector_pct: { en: 'Sector', zh: '行业集中度' },
  max_order_freq: { en: 'Order Freq', zh: '订单频率' },
}

function formatNum(v) {
  if (v == null) return '--'
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(v)
}

function formatLimitName(key) {
  const known = limitNameMap[key]
  if (known) {
    return locale.value === 'zh-CN' ? known.zh : known.en
  }
  return key.replace(/_/g, ' ').replace(/max /, '').replace(/pct/, '').trim()
}

function getLimitFill(key, val) {
  const current = status.value?.current_drawdown || 0
  const dailyLoss = status.value?.daily_pnl_pct || 0
  if (key.includes('drawdown') && !key.includes('kill')) return Math.min(current / val * 100, 100)
  if (key.includes('daily')) return Math.min(Math.abs(dailyLoss) / val * 100, 100)
  return 0
}

async function refresh() {
  loading.value = true
  try {
    status.value = await getRiskStatus()
  } catch {
    // ignore
  } finally {
    loading.value = false
  }
}

async function toggleKill() {
  const action = status.value?.kill_switch ? 'deactivate' : 'activate'
  const reason = locale.value === 'zh-CN' ? '从界面手动触发' : 'Manual toggle from UI'
  try {
    await toggleKillSwitch({ action, reason })
    await refresh()
    emit('toast', { message: locale.value === 'zh-CN' ? `熔断开关已${action === 'activate' ? '触发' : '解除'}` : `Kill switch ${action}d`, type: action === 'activate' ? 'error' : 'success' })
  } catch (e) {
    emit('toast', { message: locale.value === 'zh-CN' ? `操作失败：${e.message}` : `Failed: ${e.message}`, type: 'error' })
  }
}

onMounted(refresh)
</script>

<style scoped>
.rm-view { display: flex; flex-direction: column; gap: 10px; height: 100%; }
.rm-header { display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
.rm-title { font-size: 12px; font-weight: 600; color: var(--text-secondary); display: flex; align-items: center; gap: 8px; letter-spacing: 0.5px; }
.rm-dot { width: 8px; height: 8px; border-radius: 50%; }
.rm-dot.green { background: var(--green); box-shadow: 0 0 6px rgba(52,211,153,0.5); }
.rm-dot.yellow { background: var(--orange); }
.rm-dot.orange { background: var(--orange); box-shadow: 0 0 6px rgba(251,191,36,0.5); }
.rm-dot.red { background: var(--red); box-shadow: 0 0 6px rgba(239,68,68,0.5); }
.rm-dot.kill { background: var(--red); animation: pulse 0.5s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
.rm-actions { display: flex; gap: 8px; }

.rm-banner { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; flex-shrink: 0; }
.rm-banner-green { background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.2); color: var(--green); }
.rm-banner-yellow, .rm-banner-orange { background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.2); color: var(--orange); }
.rm-banner-red, .rm-banner-kill { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); color: var(--red); }
.rm-banner-icon { font-size: 14px; }
.rm-banner-text { flex: 1; }
.rm-banner-detail { font-size: 9px; opacity: 0.7; }

.rm-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; flex-shrink: 0; }
.rm-metric { background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 4px; padding: 6px 8px; text-align: center; }
.rm-metric-label { font-size: 8px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
.rm-metric-value { font-size: 13px; font-weight: 700; font-family: var(--font-mono); color: var(--text-primary); margin-top: 2px; }
.rm-pos { color: var(--green); }
.rm-neg { color: var(--red); }

.rm-limits { flex-shrink: 0; }
.rm-limit-title { font-size: 9px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.rm-limit-row { display: flex; align-items: center; gap: 8px; padding: 3px 0; }
.rm-limit-name { font-size: 10px; color: var(--text-dim); width: 80px; text-transform: capitalize; }
.rm-limit-bar-wrap { flex: 1; height: 4px; background: var(--bg-input); border-radius: 2px; position: relative; overflow: hidden; }
.rm-limit-bar-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--green), var(--orange)); transition: width 0.3s; }
.rm-limit-value { font-size: 10px; font-family: var(--font-mono); color: var(--text-secondary); width: 40px; text-align: right; }

.rm-breaches { flex: 1; overflow: auto; min-height: 0; }
.rm-breach-title { font-size: 9px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.rm-breach-item { display: flex; align-items: center; gap: 8px; padding: 4px 8px; border-radius: 4px; margin-bottom: 4px; font-size: 10px; }
.rm-breach-green { background: rgba(52,211,153,0.05); }
.rm-breach-yellow, .rm-breach-orange { background: rgba(251,191,36,0.05); }
.rm-breach-red, .rm-breach-kill { background: rgba(239,68,68,0.05); }
.rm-breach-severity { font-size: 8px; font-weight: 700; text-transform: uppercase; padding: 1px 4px; border-radius: 2px; background: var(--bg-input); color: var(--text-dim); }
.rm-breach-msg { flex: 1; color: var(--text-secondary); }
.rm-breach-time { font-size: 9px; font-family: var(--font-mono); color: var(--text-dim); }

.rm-empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; color: var(--text-muted); }
.rm-empty-icon { font-size: 48px; opacity: 0.2; }
.rm-empty h3 { font-size: 14px; color: var(--text-secondary); }
.rm-empty p { font-size: 12px; max-width: 400px; text-align: center; line-height: 1.6; }
</style>
