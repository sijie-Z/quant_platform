<template>
  <div class="factor-scatter">
    <div v-if="!scatterData || !scatterData.length" class="fs-empty">No scatter data</div>
    <template v-else>
      <div class="fs-tabs">
        <button
          v-for="(item, i) in scatterData"
          :key="item.factor_name"
          :class="['fs-tab', { active: activeIdx === i }]"
          @click="activeIdx = i"
        >
          {{ item.factor_name }}
          <span :class="['fs-ic-badge', icClass(item.icir)]">{{ item.icir?.toFixed(2) }}</span>
        </button>
      </div>
      <div ref="chartRef" class="fs-chart"></div>
      <div class="fs-stats" v-if="activeItem">
        <div class="fs-stat">
          <span class="fs-stat-label">IC</span>
          <span class="fs-stat-value">{{ activeItem.ic?.toFixed(4) }}</span>
        </div>
        <div class="fs-stat">
          <span class="fs-stat-label">ICIR</span>
          <span :class="['fs-stat-value', icClass(activeItem.icir)]">{{ activeItem.icir?.toFixed(3) }}</span>
        </div>
        <div class="fs-stat">
          <span class="fs-stat-label">t-stat</span>
          <span :class="['fs-stat-value', tStatClass(activeItem.t_stat)]">{{ activeItem.t_stat?.toFixed(1) }}</span>
        </div>
        <div class="fs-stat">
          <span class="fs-stat-label">Significance</span>
          <span :class="['fs-stat-value', sigClass(activeItem.t_stat)]">{{ significance(activeItem.t_stat) }}</span>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  scatterData: { type: Array, default: () => [] },
})

const chartRef = ref(null)
const activeIdx = ref(0)
let chartInstance = null
let resizeObserver = null

const activeItem = computed(() => props.scatterData?.[activeIdx.value])

function icClass(icir) {
  if (icir > 0.2) return 'fs-pos'
  if (icir < -0.2) return 'fs-neg'
  return ''
}

function tStatClass(t) {
  if (Math.abs(t) > 2.58) return 'fs-pos'
  if (Math.abs(t) > 1.96) return 'fs-warn'
  return ''
}

function significance(t) {
  const a = Math.abs(t)
  if (a > 2.58) return '***'
  if (a > 1.96) return '**'
  if (a > 1.65) return '*'
  return 'ns'
}

function renderChart() {
  if (!chartRef.value || !activeItem.value) return
  if (chartInstance) chartInstance.dispose()

  chartInstance = echarts.init(chartRef.value, null, { renderer: 'canvas' })
  const item = activeItem.value
  const points = item.points || []

  chartInstance.setOption({
    textStyle: { color: '#8b9dc0', fontFamily: 'Inter, -apple-system, sans-serif', fontSize: 10 },
    grid: { left: 44, right: 14, top: 10, bottom: 28 },
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(15, 24, 41, 0.95)',
      borderColor: '#1c2d4a',
      borderWidth: 1,
      textStyle: { color: '#e8edf5', fontSize: 11 },
      extraCssText: 'backdrop-filter: blur(8px); border-radius: 6px;',
      formatter: p => `Factor: ${p.value[0]?.toFixed(3)}<br/>Return: ${p.value[1]?.toFixed(2)}%`,
    },
    xAxis: {
      type: 'value',
      name: item.factor_name,
      nameLocation: 'center',
      nameGap: 18,
      nameTextStyle: { color: '#556882', fontSize: 9 },
      axisLine: { lineStyle: { color: '#1c2d4a' } },
      axisLabel: { color: '#556882', fontSize: 9 },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: '#152035', type: 'dashed' } },
    },
    yAxis: {
      type: 'value',
      name: 'Fwd Return %',
      nameLocation: 'center',
      nameGap: 34,
      nameTextStyle: { color: '#556882', fontSize: 9 },
      axisLine: { show: false },
      axisLabel: { color: '#556882', fontSize: 9, formatter: v => v.toFixed(1) },
      splitLine: { lineStyle: { color: '#152035', type: 'dashed' } },
      axisTick: { show: false },
    },
    series: [
      {
        type: 'scatter',
        data: points.map(p => [p.x, p.y]),
        symbolSize: 4,
        itemStyle: {
          color: new echarts.graphic.RadialGradient(0.5, 0.5, 0.5, [
            { offset: 0, color: 'rgba(77,166,255,0.6)' },
            { offset: 1, color: 'rgba(77,166,255,0.15)' },
          ]),
        },
        emphasis: {
          itemStyle: {
            color: '#4da6ff',
            borderColor: '#fff',
            borderWidth: 1,
            shadowBlur: 6,
            shadowColor: 'rgba(77,166,255,0.5)',
          },
        },
      },
      {
        type: 'line',
        data: computeRegressionLine(points),
        lineStyle: { color: item.icir > 0 ? '#34d399' : '#f87171', width: 1.5, type: 'dashed' },
        symbol: 'none',
        silent: true,
      },
    ],
  })

  if (window.ResizeObserver) {
    resizeObserver = new ResizeObserver(() => chartInstance?.resize())
    resizeObserver.observe(chartRef.value)
  }
}

function computeRegressionLine(points) {
  if (points.length < 2) return []
  const n = points.length
  let sx = 0, sy = 0, sxy = 0, sx2 = 0
  for (const p of points) {
    sx += p.x; sy += p.y; sxy += p.x * p.y; sx2 += p.x * p.x
  }
  const slope = (n * sxy - sx * sy) / (n * sxy - sx * sx || 1)
  const intercept = (sy - slope * sx) / n
  const minX = Math.min(...points.map(p => p.x))
  const maxX = Math.max(...points.map(p => p.x))
  return [[minX, slope * minX + intercept], [maxX, slope * maxX + intercept]]
}

watch(activeIdx, () => nextTick(renderChart))
watch(() => props.scatterData, () => { activeIdx.value = 0; nextTick(renderChart) }, { deep: true })
onMounted(() => { if (props.scatterData?.length) nextTick(renderChart) })
onBeforeUnmount(() => {
  if (chartInstance) chartInstance.dispose()
  if (resizeObserver) resizeObserver.disconnect()
})
</script>

<style scoped>
.factor-scatter {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 4px;
}

.fs-tabs {
  display: flex;
  gap: 3px;
  flex-wrap: wrap;
  flex-shrink: 0;
}

.fs-tab {
  padding: 2px 7px;
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  background: var(--bg-secondary);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s;
  display: flex;
  align-items: center;
  gap: 4px;
  white-space: nowrap;
}

.fs-tab:hover { color: var(--text-secondary); border-color: var(--border); }
.fs-tab.active { color: var(--accent); border-color: var(--accent); background: rgba(77,166,255,0.08); }

.fs-ic-badge {
  font-family: var(--font-mono);
  font-size: 8px;
  padding: 0 3px;
  border-radius: 2px;
  background: var(--bg-input);
}

.fs-chart {
  flex: 1;
  min-height: 0;
  width: 100%;
}

.fs-stats {
  display: flex;
  gap: 12px;
  flex-shrink: 0;
  padding: 4px 0;
}

.fs-stat {
  display: flex;
  align-items: center;
  gap: 4px;
}

.fs-stat-label {
  font-size: 8px;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.fs-stat-value {
  font-size: 10px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}

.fs-pos { color: var(--green); }
.fs-neg { color: var(--red); }
.fs-warn { color: var(--orange); }

.fs-empty {
  color: var(--text-dim);
  font-size: 11px;
  text-align: center;
  padding: 20px;
}
</style>
