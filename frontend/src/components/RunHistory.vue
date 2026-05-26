<template>
  <div>
    <div class="section-header">
      <div>
        <div class="section-title">{{ locale === 'zh-CN' ? '运行历史' : 'Run History' }}</div>
        <div class="section-subtitle">{{ locale === 'zh-CN' ? '查看和检查过去的流水线运行记录' : 'View and inspect past pipeline runs' }}</div>
      </div>
      <button class="btn btn-secondary btn-sm" @click="refresh" :disabled="loading">
        <span :class="loading ? 'status-spinner' : ''" style="display:inline-block;width:12px;height:12px;"></span>
        {{ locale === 'zh-CN' ? '刷新' : 'Refresh' }}
      </button>
    </div>

    <div v-if="error" role="alert" class="alert alert-error">
      <span aria-hidden="true">&#10007;</span> {{ error }}
    </div>

    <div v-if="loading && !runs.length" class="run-list">
      <div class="skeleton" v-for="i in 5" :key="i" style="height:60px;border-radius:10px;"></div>
    </div>

    <div v-else-if="!runs.length" class="empty-state">
      <div class="empty-icon">&#9201;</div>
      <h3>{{ locale === 'zh-CN' ? '暂无运行记录' : 'No runs yet' }}</h3>
      <p>{{ locale === 'zh-CN' ? '请先在仪表盘标签页运行流水线，查看结果。' : 'Start a pipeline run from the Dashboard tab to see results here.' }}</p>
    </div>

    <div v-else class="run-list">
      <div
        v-for="run in runs"
        :key="run.run_id"
        class="run-item"
        @click="inspectRun(run)"
        tabindex="0"
        @keydown.enter="inspectRun(run)"
      >
        <span class="run-item-id text-mono">{{ run.run_id }}</span>
        <span :class="['status-badge', run.status]">
          <span v-if="run.status === 'running'" class="status-spinner"></span>
          <span v-else-if="run.status === 'completed'" aria-hidden="true">&#10003;</span>
          <span v-else aria-hidden="true">&#10007;</span>
          {{ run.status }}
        </span>
        <span class="run-item-stage">{{ run.stage }}</span>
        <span class="run-item-time">{{ formatTime(run.started_at) }}</span>
        <span v-if="run.status === 'running'" class="text-mono text-accent">{{ run.progress }}%</span>
        <span v-else-if="run.completed_at" class="text-dim text-xs">
          {{ formatDuration(run.started_at, run.completed_at) }}
        </span>
      </div>
    </div>

    <!-- Detail Panel -->
    <Transition name="tab-content">
      <div v-if="selectedRun" class="card mt-4">
        <div class="card-header">
          <div class="card-title">
            <span class="card-title-dot"></span>
            {{ locale === 'zh-CN' ? '运行详情：' : 'Run Detail: ' }}{{ selectedRun.run_id }}
          </div>
          <button class="btn btn-ghost btn-xs" @click="selectedRun = null">&#10005; {{ locale === 'zh-CN' ? '关闭' : 'Close' }}</button>
        </div>

        <div v-if="detailLoading" class="text-muted">{{ locale === 'zh-CN' ? '正在加载结果...' : 'Loading result...' }}</div>
        <div v-else-if="detailError" class="alert alert-error">{{ detailError }}</div>
        <div v-else-if="detailData">
          <!-- Metrics -->
          <div v-if="detailData.performance" class="metrics-grid mb-4">
            <div class="metric-card" v-for="m in detailMetrics" :key="m.label">
              <div :class="['metric-value', m.color]">{{ m.value }}</div>
              <div class="metric-label">{{ m.label }}</div>
            </div>
          </div>

          <!-- Factors -->
          <div v-if="detailData.factors?.length" class="mt-3">
            <div class="text-sm text-muted mb-2">{{ locale === 'zh-CN' ? '按ICIR排序的顶级因子' : 'Top Factors by ICIR' }}</div>
            <div class="table-container">
              <table>
                <thead>
                  <tr><th>#</th><th>{{ locale === 'zh-CN' ? '因子' : 'Factor' }}</th><th>{{ locale === 'zh-CN' ? '平均IC' : 'Mean IC' }}</th><th>ICIR</th><th>{{ locale === 'zh-CN' ? 'IC>0%' : 'IC>0 %' }}</th></tr>
                </thead>
                <tbody>
                  <tr v-for="(f, i) in detailData.factors.slice(0, 8)" :key="f.name">
                    <td>{{ i + 1 }}</td>
                    <td class="highlight-number">{{ f.name }}</td>
                    <td>{{ Number(f.mean_ic).toFixed(4) }}</td>
                    <td :class="Math.abs(f.icir) > 0.2 ? 'positive' : ''">{{ Number(f.icir).toFixed(2) }}</td>
                    <td>{{ (Number(f.ic_positive_ratio) * 100).toFixed(1) }}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { getRuns, getRunResult } from '../api/index.js'
import { useI18n } from '../i18n/index.js'

const emit = defineEmits(['toast'])
const { $t, locale } = useI18n()

const runs = ref([])
const loading = ref(false)
const error = ref(null)
const selectedRun = ref(null)
const detailData = ref(null)
const detailLoading = ref(false)
const detailError = ref(null)

const detailMetrics = computed(() => {
  if (!detailData.value?.performance) return []
  const p = detailData.value.performance
  return [
    { label: locale.value === 'zh-CN' ? '总收益' : 'Total Return', value: (p.total_return * 100).toFixed(1) + '%', color: p.total_return >= 0 ? 'positive' : 'negative' },
    { label: locale.value === 'zh-CN' ? '夏普比' : 'Sharpe', value: Number(p.sharpe_ratio).toFixed(2), color: p.sharpe_ratio >= 1 ? 'positive' : p.sharpe_ratio >= 0 ? 'neutral' : 'negative' },
    { label: locale.value === 'zh-CN' ? '最大回撤' : 'Max DD', value: (p.max_drawdown * 100).toFixed(1) + '%', color: 'negative' },
    { label: locale.value === 'zh-CN' ? '胜率' : 'Win Rate', value: (p.win_rate * 100).toFixed(1) + '%', color: p.win_rate >= 0.5 ? 'positive' : 'negative' },
    { label: locale.value === 'zh-CN' ? '再平衡次数' : 'Rebalances', value: p.n_rebalances, color: 'accent' },
  ]
})

function formatTime(iso) {
  if (!iso) return '-'
  const d = new Date(iso)
  return d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function formatDuration(start, end) {
  if (!start || !end) return ''
  const ms = new Date(end) - new Date(start)
  if (ms < 1000) return ms + 'ms'
  return (ms / 1000).toFixed(1) + 's'
}

async function refresh() {
  loading.value = true
  error.value = null
  try {
    const data = await getRuns()
    runs.value = data.runs || []
  } catch (e) {
    error.value = e.response?.data?.detail || e.message
  } finally {
    loading.value = false
  }
}

async function inspectRun(run) {
  if (run.status !== 'completed') {
    emit('toast', { message: locale.value === 'zh-CN' ? '只能查看已完成的运行' : 'Can only inspect completed runs', type: 'info' })
    return
  }
  selectedRun.value = run
  detailLoading.value = true
  detailError.value = null
  detailData.value = null
  try {
    detailData.value = await getRunResult(run.run_id)
  } catch (e) {
    detailError.value = e.response?.data?.detail || e.message
  } finally {
    detailLoading.value = false
  }
}

onMounted(refresh)
</script>
