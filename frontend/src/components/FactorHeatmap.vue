<template>
  <div class="factor-heatmap">
    <div v-if="!sorted.length" class="fh-empty">No factor data</div>
    <div v-else class="fh-grid">
      <div
        v-for="f in sorted"
        :key="f.name"
        :class="['fh-cell', cellClass(f)]"
        :title="`${f.name}\nIC: ${Number(f.mean_ic).toFixed(4)}\nICIR: ${Number(f.icir).toFixed(2)}\nIC>0: ${(Number(f.ic_positive_ratio) * 100).toFixed(0)}%`"
      >
        <div class="fh-name">{{ shortName(f.name) }}</div>
        <div class="fh-icir">
          <span class="fh-dir" v-if="Number(f.icir) > 0">+</span>{{ Number(f.icir).toFixed(2) }}
        </div>
        <div class="fh-ic">{{ Number(f.mean_ic).toFixed(3) }}</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  factors: { type: Array, default: () => [] },
})

const sorted = computed(() => {
  return [...props.factors].sort((a, b) => Math.abs(Number(b.icir)) - Math.abs(Number(a.icir)))
})

function shortName(name) {
  return name
    .replace('momentum_', 'mom')
    .replace('volatility_', 'vol')
    .replace('turnover_', 'turn')
    .replace('_14d', '')
    .replace('_20d', '')
    .replace('_60d', '')
    .replace('_1m', '1m')
    .replace('_3m', '3m')
    .replace('_6m', '6m')
    .replace('_12m', '12m')
    .replace('log_market_cap', 'lncap')
    .replace('pb_ratio', 'pb')
    .replace('pe_ratio', 'pe')
    .replace('asset_growth', 'asset')
    .replace('amplitude_', 'amp')
    .replace('rsi', 'rsi')
    .replace('macd', 'macd')
}

function cellClass(f) {
  const icir = Number(f.icir)
  const absIcir = Math.abs(icir)
  const direction = icir >= 0 ? 'pos' : 'neg'
  if (absIcir >= 0.3) return `strong-${direction}`
  if (absIcir >= 0.2) return `good-${direction}`
  if (absIcir >= 0.1) return `weak-${direction}`
  return 'flat'
}
</script>

<style scoped>
.factor-heatmap {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.fh-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(68px, 1fr));
  gap: 3px;
  width: 100%;
}

.fh-cell {
  border-radius: 4px;
  padding: 5px 4px;
  text-align: center;
  cursor: default;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-height: 46px;
  justify-content: center;
}

.fh-cell:hover {
  transform: scale(1.08);
  box-shadow: 0 2px 10px rgba(0,0,0,0.4);
  z-index: 1;
}

/* Positive ICIR cells (green) */
.fh-cell.strong-pos {
  background: rgba(52, 211, 153, 0.22);
  border: 1px solid rgba(52, 211, 153, 0.4);
}

.fh-cell.good-pos {
  background: rgba(52, 211, 153, 0.1);
  border: 1px solid rgba(52, 211, 153, 0.2);
}

.fh-cell.weak-pos {
  background: rgba(52, 211, 153, 0.04);
  border: 1px solid rgba(52, 211, 153, 0.1);
}

/* Negative ICIR cells (red) */
.fh-cell.strong-neg {
  background: rgba(248, 113, 113, 0.22);
  border: 1px solid rgba(248, 113, 113, 0.4);
}

.fh-cell.good-neg {
  background: rgba(248, 113, 113, 0.1);
  border: 1px solid rgba(248, 113, 113, 0.2);
}

.fh-cell.weak-neg {
  background: rgba(248, 113, 113, 0.04);
  border: 1px solid rgba(248, 113, 113, 0.1);
}

.fh-cell.flat {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-subtle);
}

.fh-name {
  font-size: 8.5px;
  font-weight: 600;
  color: var(--text-secondary);
  letter-spacing: 0.3px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.fh-icir {
  font-size: 13px;
  font-weight: 700;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
}

.fh-dir {
  font-size: 10px;
  opacity: 0.7;
}

.fh-cell.strong-pos .fh-icir { color: var(--green-bright); }
.fh-cell.good-pos .fh-icir { color: var(--green); }
.fh-cell.weak-pos .fh-icir { color: var(--green-dim); }
.fh-cell.strong-neg .fh-icir { color: var(--red-bright); }
.fh-cell.good-neg .fh-icir { color: var(--red); }
.fh-cell.weak-neg .fh-icir { color: var(--red-dim); }
.fh-cell.flat .fh-icir { color: var(--text-dim); }

.fh-ic {
  font-size: 8px;
  font-family: var(--font-mono);
  color: var(--text-dim);
  font-variant-numeric: tabular-nums;
}

.fh-empty {
  color: var(--text-dim);
  font-size: 11px;
  text-align: center;
  padding: 20px;
}
</style>
