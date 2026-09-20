<template>
  <div class="ic-decay">
    <div class="decay-header">
      <span class="decay-title">IC DECAY CURVE</span>
      <div v-if="curves.length" class="decay-tabs">
        <button
          v-for="c in curves"
          :key="c.factor"
          :class="['decay-tab', { active: selectedFactor === c.factor }]"
          @click="selectedFactor = c.factor"
        >{{ c.factor }}</button>
      </div>
    </div>
    <div ref="chartRef" class="decay-chart"></div>
    <div v-if="!curves.length" class="decay-empty">{{ message }}</div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, computed, nextTick } from 'vue'
import * as echarts from 'echarts'
import { getICDecay } from '../api/index.js'

/**
 * The curve comes from `/api/analysis/ic-decay`. It used to be generated here,
 * in the browser: `Math.exp(-lag * 0.15) * baseIC` with a baseIC picked by
 * substring-matching the factor name and noise from a seeded LCG. That was the
 * same invention the endpoint itself was removed for (BUG-45), one layer up --
 * the panel never called the API at all.
 */
const chartRef = ref(null)
const selectedFactor = ref('')
const curves = ref([])
const message = ref('Loading IC decay...')
let chart = null
let resizeObs = null

const selectedCurve = computed(
  () => curves.value.find(c => c.factor === selectedFactor.value) || curves.value[0],
)

async function load() {
  try {
    const data = await getICDecay()
    if (data?.available === false) {
      curves.value = []
      message.value = data.reason || 'IC decay is not available for this run.'
      return
    }
    curves.value = data?.factors ?? []
    if (!curves.value.length) {
      message.value = 'No IC decay data for this run.'
    } else if (!selectedFactor.value) {
      selectedFactor.value = curves.value[0].factor
    }
  } catch (e) {
    curves.value = []
    message.value = `IC decay request failed: ${e?.message || e}`
  }
}

function render() {
  const curve = selectedCurve.value
  if (!chartRef.value || !curve?.lags?.length) return

  if (!chart) {
    chart = echarts.init(chartRef.value, null, { renderer: 'canvas' })
  }

  chart.setOption({
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const p = params[0]
        return `Lag ${p.name} days<br/>IC: <b>${p.value.toFixed(4)}</b>`
      },
    },
    grid: { top: 16, right: 20, bottom: 28, left: 50 },
    xAxis: {
      type: 'category',
      data: curve.lags,
      name: 'Lag (days)',
      nameLocation: 'center',
      nameGap: 18,
      nameTextStyle: { color: '#6b7a8d', fontSize: 9 },
      axisLabel: { color: '#8892a4', fontSize: 9 },
      axisLine: { lineStyle: { color: '#1e2a3a' } },
    },
    yAxis: {
      type: 'value',
      name: 'Rank IC',
      nameTextStyle: { color: '#6b7a8d', fontSize: 9 },
      axisLabel: { color: '#8892a4', fontSize: 9, formatter: v => v.toFixed(3) },
      splitLine: { lineStyle: { color: '#1e2a3a' } },
      axisLine: { lineStyle: { color: '#1e2a3a' } },
    },
    series: [
      {
        type: 'line',
        data: curve.ics,
        smooth: 0.3,
        symbol: 'circle',
        symbolSize: 5,
        lineStyle: { color: '#4da6ff', width: 2 },
        itemStyle: { color: '#4da6ff' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(77,166,255,0.25)' },
            { offset: 1, color: 'rgba(77,166,255,0.02)' },
          ]),
        },
        markLine: {
          silent: true,
          data: [{ yAxis: 0, lineStyle: { color: '#3a4a5e', type: 'dashed' } }],
          label: { show: false },
        },
      },
    ],
  })
}

watch(selectedFactor, () => nextTick(render))

onMounted(async () => {
  await load()
  await nextTick()
  render()
  resizeObs = new ResizeObserver(() => chart?.resize())
  if (chartRef.value) resizeObs.observe(chartRef.value)
})

onBeforeUnmount(() => {
  resizeObs?.disconnect()
  chart?.dispose()
})
</script>

<style scoped>
.ic-decay {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 6px;
  position: relative;
}

.decay-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  gap: 8px;
}

.decay-title {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

.decay-tabs {
  display: flex;
  gap: 2px;
  overflow-x: auto;
  flex-shrink: 0;
}

.decay-tab {
  padding: 2px 6px;
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  background: transparent;
  border: 1px solid transparent;
  border-radius: 3px;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.15s;
}

.decay-tab:hover { color: var(--text-secondary); }
.decay-tab.active {
  color: var(--accent);
  background: rgba(77,166,255,0.1);
  border-color: rgba(77,166,255,0.2);
}

.decay-chart {
  flex: 1;
  min-height: 0;
}

.decay-empty {
  position: absolute;
  inset: 24px 0 0 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 16px;
  text-align: center;
  font-size: 11px;
  line-height: 1.6;
  color: var(--text-dim);
  pointer-events: none;
}
</style>
