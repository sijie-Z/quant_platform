<template>
  <div class="kpi-strip">
    <div
      v-for="m in metrics"
      :key="m.key"
      :class="['kpi-item', m.tone]"
    >
      <div class="kpi-label">{{ m.label }}</div>
      <div class="kpi-value">{{ m.display }}</div>
      <div v-if="m.sub" class="kpi-sub">{{ m.sub }}</div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  performance: { type: Object, default: null },
  risk: { type: Object, default: null },
})

const metrics = computed(() => {
  if (!props.performance) return placeholderMetrics
  const p = props.performance
  const r = props.risk || {}

  return [
    { key: 'ret',    label: 'TOTAL RET',   display: pct(p.total_return),      tone: p.total_return >= 0 ? 'pos' : 'neg' },
    { key: 'ann',    label: 'ANN. RET',    display: pct(p.annual_return),     tone: p.annual_return >= 0 ? 'pos' : 'neg' },
    { key: 'vol',    label: 'ANN. VOL',    display: pct(p.annual_volatility), tone: 'neu' },
    { key: 'sharpe', label: 'SHARPE',      display: num(p.sharpe_ratio),      tone: p.sharpe_ratio >= 1 ? 'pos' : p.sharpe_ratio >= 0 ? 'neu' : 'neg' },
    { key: 'sort',   label: 'SORTINO',     display: num(p.sortino_ratio),     tone: p.sortino_ratio >= 0 ? 'pos' : 'neg' },
    { key: 'dd',     label: 'MAX DD',      display: pct(p.max_drawdown),      tone: 'neg' },
    { key: 'var',    label: 'VAR 95%',     display: r.historical_var != null ? pct(r.historical_var) : '--', tone: 'neg' },
    { key: 'win',    label: 'WIN RATE',    display: pct(p.win_rate),          tone: p.win_rate >= 0.5 ? 'pos' : 'neu', sub: `${p.n_rebalances || 0} rebal` },
  ]
})

const placeholderMetrics = [
  { key: 'ret',    label: 'TOTAL RET',   display: '--', tone: 'dim' },
  { key: 'ann',    label: 'ANN. RET',    display: '--', tone: 'dim' },
  { key: 'vol',    label: 'ANN. VOL',    display: '--', tone: 'dim' },
  { key: 'sharpe', label: 'SHARPE',      display: '--', tone: 'dim' },
  { key: 'sort',   label: 'SORTINO',     display: '--', tone: 'dim' },
  { key: 'dd',     label: 'MAX DD',      display: '--', tone: 'dim' },
  { key: 'var',    label: 'VAR 95%',     display: '--', tone: 'dim' },
  { key: 'win',    label: 'WIN RATE',    display: '--', tone: 'dim' },
]

function pct(v) {
  if (v == null) return '--'
  return (Number(v) * 100).toFixed(2) + '%'
}

function num(v) {
  if (v == null) return '--'
  return Number(v).toFixed(2)
}
</script>

<style scoped>
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  gap: 1px;
  background: var(--border-subtle);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  flex-shrink: 0;
}

.kpi-item {
  background: var(--bg-card);
  padding: 8px 10px;
  text-align: center;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.kpi-label {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  letter-spacing: 0.8px;
  text-transform: uppercase;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.kpi-value {
  font-size: 16px;
  font-weight: 700;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.5px;
  line-height: 1.2;
}

.kpi-sub {
  font-size: 8px;
  color: var(--text-dim);
  letter-spacing: 0.3px;
}

.kpi-item.pos .kpi-value { color: var(--green); }
.kpi-item.neg .kpi-value { color: var(--red); }
.kpi-item.neu .kpi-value { color: var(--orange); }
.kpi-item.dim .kpi-value { color: var(--text-dim); }

@media (max-width: 1200px) {
  .kpi-strip {
    grid-template-columns: repeat(4, 1fr);
  }
}

@media (max-width: 600px) {
  .kpi-strip {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
