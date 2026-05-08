<template>
  <div class="holdings-table">
    <div v-if="!holdings || !holdings.length" class="ht-empty">No holdings data</div>
    <div v-else class="ht-scroll">
      <table class="ht-tbl">
        <thead>
          <tr>
            <th class="ht-rank">#</th>
            <th class="ht-ticker">Ticker</th>
            <th class="ht-sector">Sector</th>
            <th class="ht-weight">Weight</th>
            <th class="ht-bar">Allocation</th>
            <th class="ht-pnl">1D P&L</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(h, i) in holdings" :key="h.ticker" class="ht-row">
            <td class="ht-rank">{{ i + 1 }}</td>
            <td class="ht-ticker">
              <span class="ht-ticker-code">{{ h.ticker }}</span>
            </td>
            <td class="ht-sector">
              <span class="ht-sector-tag" :style="{ background: sectorColor(h.sector) }">
                {{ h.sector || '--' }}
              </span>
            </td>
            <td class="ht-weight">{{ (h.weight * 100).toFixed(2) }}%</td>
            <td class="ht-bar">
              <div class="ht-bar-wrap">
                <div
                  class="ht-bar-fill"
                  :style="{ width: (h.weight / maxWeight * 100) + '%' }"
                ></div>
              </div>
            </td>
            <td :class="['ht-pnl', pnlClass(h.pnl_pct)]">
              {{ h.pnl_pct != null ? (h.pnl_pct >= 0 ? '+' : '') + h.pnl_pct.toFixed(2) + '%' : '--' }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  holdings: { type: Array, default: () => [] },
})

const maxWeight = computed(() => {
  if (!props.holdings?.length) return 1
  return Math.max(...props.holdings.map(h => h.weight), 0.001)
})

const sectorColors = {
  '银行': '#4da6ff', '食品饮料': '#34d399', '电子': '#a78bfa',
  '医药生物': '#fb923c', '电力设备': '#22d3ee', '汽车': '#fbbf24',
  '非银金融': '#f87171', '计算机': '#818cf8', '机械设备': '#2dd4bf',
  '化工': '#e879f9', '有色金属': '#facc15', '公用事业': '#94a3b8',
  '商贸零售': '#f472b6', '钢铁': '#78716c', '建筑材料': '#a3a3a3',
}

function sectorColor(s) {
  if (!s) return 'var(--bg-input)'
  return sectorColors[s] || '#3a4d6a'
}

function pnlClass(v) {
  if (v == null) return ''
  return v > 0 ? 'ht-pos' : v < 0 ? 'ht-neg' : ''
}
</script>

<style scoped>
.holdings-table {
  height: 100%;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.ht-scroll {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
}

.ht-scroll::-webkit-scrollbar {
  width: 4px;
}

.ht-scroll::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 2px;
}

.ht-tbl {
  width: 100%;
  border-collapse: collapse;
  font-size: 10px;
  font-variant-numeric: tabular-nums;
}

.ht-tbl thead {
  position: sticky;
  top: 0;
  z-index: 1;
}

.ht-tbl th {
  padding: 4px 6px;
  font-size: 8px;
  font-weight: 700;
  color: var(--text-dim);
  letter-spacing: 0.8px;
  text-transform: uppercase;
  text-align: left;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
  white-space: nowrap;
}

.ht-tbl td {
  padding: 3px 6px;
  border-bottom: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  white-space: nowrap;
}

.ht-row:hover td {
  background: rgba(77, 166, 255, 0.04);
}

.ht-rank {
  width: 24px;
  text-align: center;
  color: var(--text-dim);
  font-size: 9px;
}

.ht-ticker-code {
  font-family: var(--font-mono);
  font-weight: 600;
  font-size: 10px;
  color: var(--accent);
}

.ht-sector-tag {
  display: inline-block;
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 8px;
  font-weight: 600;
  color: #fff;
  opacity: 0.85;
}

.ht-weight {
  font-family: var(--font-mono);
  font-weight: 600;
  text-align: right;
}

.ht-bar {
  width: 80px;
}

.ht-bar-wrap {
  height: 4px;
  background: var(--bg-input);
  border-radius: 2px;
  overflow: hidden;
}

.ht-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-dim), var(--accent));
  border-radius: 2px;
  transition: width 0.4s ease;
  min-width: 2px;
}

.ht-pnl {
  text-align: right;
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 10px;
}

.ht-pos { color: var(--green); }
.ht-neg { color: var(--red); }

.ht-empty {
  color: var(--text-dim);
  font-size: 11px;
  text-align: center;
  padding: 20px;
}
</style>
