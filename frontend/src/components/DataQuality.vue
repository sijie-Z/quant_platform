<template>
  <div class="dq-container">
    <div class="dq-header">
      <button class="dq-btn" @click="runCheck" :disabled="loading">
        <span v-if="loading" class="dq-spinner"></span>
        <span v-else>{{ locale === 'zh-CN' ? '运行质量检查' : 'Run Quality Check' }}</span>
      </button>
      <span class="dq-status" v-if="report">
        <span :class="report.overall_status === 'PASS' ? 'dq-pass' : 'dq-fail'">
          {{ report.overall_status === 'PASS' ? (locale === 'zh-CN' ? '通过' : 'PASS') : (locale === 'zh-CN' ? '失败' : 'FAIL') }}
        </span>
        <span class="dq-rate">{{ locale === 'zh-CN' ? `${(report.pass_rate * 100).toFixed(0)}% 通过率` : `${(report.pass_rate * 100).toFixed(0)}% pass rate` }}</span>
      </span>
    </div>

    <!-- Summary Cards -->
    <div class="dq-summary" v-if="report">
      <div class="dq-card">
        <div class="dq-card-label">{{ locale === 'zh-CN' ? '总检查项' : 'Total Checks' }}</div>
        <div class="dq-card-val">{{ report.total_checks }}</div>
      </div>
      <div class="dq-card dq-card-pass">
        <div class="dq-card-label">{{ locale === 'zh-CN' ? '通过' : 'Passed' }}</div>
        <div class="dq-card-val pos">{{ report.passed }}</div>
      </div>
      <div class="dq-card dq-card-fail">
        <div class="dq-card-label">{{ locale === 'zh-CN' ? '失败' : 'Failed' }}</div>
        <div class="dq-card-val neg">{{ report.failed }}</div>
      </div>
      <div class="dq-card" v-for="(count, sev) in report.by_severity" :key="sev">
        <div class="dq-card-label">{{ locale === 'zh-CN' ? (sev === 'info' ? '信息' : sev === 'warn' ? '警告' : sev === 'error' ? '错误' : sev === 'critical' ? '严重' : sev) : sev }}</div>
        <div class="dq-card-val" :class="sevClass(sev)">{{ count }}</div>
      </div>
    </div>

    <!-- Check Details -->
    <div class="dq-checks" v-if="report">
      <div v-for="c in report.checks" :key="c.name" class="dq-check" :class="c.passed ? 'dq-check-pass' : 'dq-check-fail'">
        <div class="dq-check-head">
          <span class="dq-check-icon">{{ c.passed ? '&#10003;' : '&#10007;' }}</span>
          <span class="dq-check-name">{{ c.name }}</span>
          <span class="dq-check-sev" :class="sevClass(c.severity)">{{ locale === 'zh-CN' ? (c.severity === 'info' ? '信息' : c.severity === 'warn' ? '警告' : c.severity === 'error' ? '错误' : c.severity === 'critical' ? '严重' : c.severity) : c.severity }}</span>
        </div>
        <div class="dq-check-msg">{{ c.message }}</div>
      </div>
    </div>

    <div v-if="!report && !loading" class="dq-empty">
      {{ locale === 'zh-CN' ? '点击"运行质量检查"以分析数据完整性。' : 'Click "Run Quality Check" to analyze data integrity.' }}
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { runDataQuality } from '../api/index.js'
import { useI18n } from '../i18n/index.js'

const { $t, locale } = useI18n()

const props = defineProps({ runId: { type: String, default: '' } })
const emit = defineEmits(['toast'])

const loading = ref(false)
const report = ref(null)

async function runCheck() {
  loading.value = true
  try {
    report.value = await runDataQuality(props.runId)
    const statusLabel = report.value.overall_status === 'PASS' ? (locale.value === 'zh-CN' ? '通过' : 'PASS') : (locale.value === 'zh-CN' ? '失败' : 'FAIL')
    emit('toast', { message: locale.value === 'zh-CN' ? `质量检查：${statusLabel}（${report.value.passed}/${report.value.total_checks}）` : `Quality check: ${statusLabel} (${report.value.passed}/${report.value.total_checks})`, type: report.value.overall_status === 'PASS' ? 'success' : 'error' })
  } catch (e) {
    emit('toast', { message: locale.value === 'zh-CN' ? `质量检查失败：${e.response?.data?.detail || e.message}` : 'Quality check failed: ' + (e.response?.data?.detail || e.message), type: 'error' })
  } finally {
    loading.value = false
  }
}

function sevClass(sev) {
  return { info: 'sev-info', warn: 'sev-warn', error: 'sev-error', critical: 'sev-critical' }[sev] || ''
}
</script>

<style scoped>
.dq-container { display: flex; flex-direction: column; gap: 8px; font-size: 11px; }
.dq-header { display: flex; align-items: center; gap: 10px; }
.dq-btn {
  padding: 5px 14px; border-radius: 4px; border: 1px solid #4da6ff; background: transparent;
  color: #4da6ff; font-size: 11px; font-weight: 600; cursor: pointer; transition: all 0.15s;
}
.dq-btn:hover { background: rgba(77,166,255,0.1); }
.dq-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.dq-spinner {
  display: inline-block; width: 12px; height: 12px; border: 2px solid #4da6ff; border-top-color: transparent;
  border-radius: 50%; animation: spin 0.6s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.dq-status { display: flex; gap: 6px; align-items: center; }
.dq-pass { color: #22c55e; font-weight: 700; }
.dq-fail { color: #ef4444; font-weight: 700; }
.dq-rate { color: #6b7a8d; font-size: 10px; }

.dq-summary { display: flex; gap: 6px; flex-wrap: wrap; }
.dq-card {
  background: var(--bg-card, #111827); border: 1px solid var(--border, #1e2a3a); border-radius: 6px;
  padding: 6px 10px; text-align: center; min-width: 70px; flex: 1;
}
.dq-card-label { font-size: 8px; font-weight: 700; color: #6b7a8d; text-transform: uppercase; letter-spacing: 0.5px; }
.dq-card-val { font-size: 16px; font-weight: 700; font-family: monospace; color: #e6edf3; margin-top: 2px; }
.dq-card-val.pos { color: #22c55e; }
.dq-card-val.neg { color: #ef4444; }

.dq-checks { display: flex; flex-direction: column; gap: 4px; }
.dq-check {
  border-radius: 4px; padding: 6px 10px;
  background: var(--bg-card, #111827); border: 1px solid var(--border, #1e2a3a);
}
.dq-check-pass { border-left: 3px solid #22c55e; }
.dq-check-fail { border-left: 3px solid #ef4444; }
.dq-check-head { display: flex; align-items: center; gap: 6px; }
.dq-check-icon { font-size: 12px; }
.dq-check-pass .dq-check-icon { color: #22c55e; }
.dq-check-fail .dq-check-icon { color: #ef4444; }
.dq-check-name { font-weight: 600; color: #e6edf3; }
.dq-check-sev {
  font-size: 8px; padding: 1px 5px; border-radius: 3px; font-weight: 700; text-transform: uppercase;
}
.sev-info { color: #4da6ff; background: rgba(77,166,255,0.1); }
.sev-warn { color: #fbbf24; background: rgba(251,191,36,0.1); }
.sev-error { color: #ef4444; background: rgba(239,68,68,0.1); }
.sev-critical { color: #ff0000; background: rgba(255,0,0,0.15); }
.dq-check-msg { font-size: 10px; color: #6b7a8d; margin-top: 2px; }
.dq-empty { text-align: center; color: #6b7a8d; padding: 20px; }
</style>
