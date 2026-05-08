<template>
  <div class="holdings-panel">
    <div v-if="!exposure" class="hp-empty">No exposure data</div>
    <template v-else>
      <!-- Summary Stats -->
      <div class="hp-stats">
        <div class="hp-stat">
          <span class="hp-stat-val">{{ exposure.n_assets }}</span>
          <span class="hp-stat-lbl">Holdings</span>
        </div>
        <div class="hp-stat">
          <span class="hp-stat-val">{{ exposure.effective_n?.toFixed(0) }}</span>
          <span class="hp-stat-lbl">Eff. N</span>
        </div>
        <div class="hp-stat">
          <span class="hp-stat-val">{{ (exposure.top5_concentration * 100).toFixed(1) }}%</span>
          <span class="hp-stat-lbl">Top 5</span>
        </div>
        <div class="hp-stat">
          <span class="hp-stat-val">{{ (exposure.top10_concentration * 100).toFixed(1) }}%</span>
          <span class="hp-stat-lbl">Top 10</span>
        </div>
      </div>

      <!-- Sector Exposure -->
      <div class="hp-sectors" v-if="sectorList.length">
        <div class="hp-sectors-title">SECTOR EXPOSURE</div>
        <div
          v-for="s in sectorList"
          :key="s.name"
          class="hp-sector-row"
        >
          <div class="hp-sector-name">{{ s.name }}</div>
          <div class="hp-sector-bar-wrap">
            <div
              class="hp-sector-bar"
              :style="{ width: (s.pct / maxSectorPct * 100) + '%' }"
            ></div>
          </div>
          <div class="hp-sector-pct">{{ s.pct.toFixed(1) }}%</div>
        </div>
      </div>

      <!-- Concentration Indicator -->
      <div class="hp-concentration">
        <div class="hp-conc-title">CONCENTRATION</div>
        <div class="hp-conc-bar">
          <div
            class="hp-conc-fill"
            :style="{ width: (exposure.top5_concentration * 100) + '%' }"
            :class="exposure.top5_concentration > 0.3 ? 'high' : exposure.top5_concentration > 0.2 ? 'med' : 'low'"
          ></div>
          <div class="hp-conc-markers">
            <div class="hp-conc-marker" style="left: 20%"><span>20%</span></div>
            <div class="hp-conc-marker" style="left: 30%"><span>30%</span></div>
            <div class="hp-conc-marker" style="left: 50%"><span>50%</span></div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  exposure: { type: Object, default: null },
})

const sectorList = computed(() => {
  if (!props.exposure?.sectors) return []
  const sectors = props.exposure.sectors
  return Object.entries(sectors)
    .map(([name, pct]) => ({ name, pct: Number(pct) * 100 }))
    .sort((a, b) => b.pct - a.pct)
    .slice(0, 8)
})

const maxSectorPct = computed(() => {
  if (!sectorList.value.length) return 1
  return Math.max(...sectorList.value.map(s => s.pct), 1)
})
</script>

<style scoped>
.holdings-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  height: 100%;
}

.hp-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
}

.hp-stat {
  background: var(--bg-secondary);
  border: 1px solid var(--border-subtle);
  border-radius: 5px;
  padding: 6px 8px;
  text-align: center;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.hp-stat-val {
  font-size: 14px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}

.hp-stat-lbl {
  font-size: 8px;
  font-weight: 600;
  color: var(--text-dim);
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.hp-sectors {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 5px;
  min-height: 0;
}

.hp-sectors-title {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  letter-spacing: 0.8px;
  text-transform: uppercase;
}

.hp-sector-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.hp-sector-name {
  font-size: 10px;
  color: var(--text-secondary);
  min-width: 60px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.hp-sector-bar-wrap {
  flex: 1;
  height: 5px;
  background: var(--bg-input);
  border-radius: 3px;
  overflow: hidden;
}

.hp-sector-bar {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-dim), var(--accent));
  border-radius: 3px;
  transition: width 0.5s ease;
  min-width: 2px;
}

.hp-sector-pct {
  font-size: 10px;
  font-weight: 600;
  font-family: var(--font-mono);
  color: var(--text-muted);
  min-width: 36px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.hp-concentration {
  margin-top: auto;
}

.hp-conc-title {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  letter-spacing: 0.8px;
  text-transform: uppercase;
  margin-bottom: 5px;
}

.hp-conc-bar {
  position: relative;
  height: 8px;
  background: var(--bg-input);
  border-radius: 4px;
  overflow: visible;
}

.hp-conc-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease;
}

.hp-conc-fill.low { background: linear-gradient(90deg, var(--green-dim), var(--green)); }
.hp-conc-fill.med { background: linear-gradient(90deg, var(--orange), var(--gold)); }
.hp-conc-fill.high { background: linear-gradient(90deg, var(--red-dim), var(--red)); }

.hp-conc-markers {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.hp-conc-marker {
  position: absolute;
  top: -2px;
  bottom: -2px;
  width: 1px;
  background: var(--border-light);
}

.hp-conc-marker span {
  position: absolute;
  top: -14px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 8px;
  color: var(--text-dim);
  white-space: nowrap;
}

.hp-empty {
  color: var(--text-dim);
  font-size: 11px;
  text-align: center;
  padding: 20px;
}
</style>
