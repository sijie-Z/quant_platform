<template>
  <div class="risk-gauges">
    <div v-if="!risk" class="rg-empty">No risk data</div>
    <template v-else>
      <!-- VaR / CVaR Gauges -->
      <div class="rg-row">
        <div class="rg-gauge" v-for="g in gauges" :key="g.label">
          <div class="rg-ring" :style="{ '--pct': g.pct, '--color': g.color }">
            <svg viewBox="0 0 36 36">
              <path
                class="rg-ring-bg"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                class="rg-ring-fill"
                :stroke="g.color"
                :stroke-dasharray="`${g.pct}, 100`"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
            </svg>
            <div class="rg-ring-value">{{ g.display }}</div>
          </div>
          <div class="rg-label">{{ g.label }}</div>
        </div>
      </div>

      <!-- Stress Tests -->
      <div class="rg-stress" v-if="stressTests.length">
        <div class="rg-stress-title">STRESS TESTS</div>
        <div
          v-for="st in stressTests"
          :key="st.scenario"
          class="rg-stress-row"
        >
          <div class="rg-stress-name">{{ shortScenario(st.scenario) }}</div>
          <div class="rg-stress-bar-wrap">
            <div
              class="rg-stress-bar"
              :style="{ width: stressBarWidth(st.max_drawdown), background: stressColor(st.max_drawdown) }"
            ></div>
          </div>
          <div :class="['rg-stress-val', stressSeverity(st.max_drawdown)]">
            {{ (Number(st.max_drawdown) * 100).toFixed(1) }}%
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  risk: { type: Object, default: null },
  stressTests: { type: Array, default: () => [] },
})

const gauges = computed(() => {
  if (!props.risk) return []
  const r = props.risk
  return [
    { label: 'VaR 95%', display: pct(r.historical_var), pct: Math.min(Math.abs(r.historical_var) * 2000, 100), color: '#fb923c' },
    { label: 'CVaR 95%', display: pct(r.historical_cvar), pct: Math.min(Math.abs(r.historical_cvar) * 1500, 100), color: '#f87171' },
    { label: 'Param VaR', display: pct(r.parametric_var), pct: Math.min(Math.abs(r.parametric_var) * 2000, 100), color: '#a78bfa' },
  ]
})

function pct(v) {
  if (v == null) return '--'
  return (Number(v) * 100).toFixed(2) + '%'
}

function shortScenario(s) {
  return s
    .replace('2008 Financial Crisis', '2008 Crisis')
    .replace('2015 A-Share Crash', '2015 Crash')
    .replace('2020 COVID-19 Shock', '2020 COVID')
}

function stressBarWidth(dd) {
  return Math.min(Math.abs(Number(dd)) * 100, 100) + '%'
}

function stressColor(dd) {
  const v = Math.abs(Number(dd))
  if (v > 0.4) return 'var(--red)'
  if (v > 0.2) return 'var(--orange)'
  return 'var(--green)'
}

function stressSeverity(dd) {
  const v = Math.abs(Number(dd))
  if (v > 0.4) return 'severe'
  if (v > 0.2) return 'moderate'
  return 'mild'
}
</script>

<style scoped>
.risk-gauges {
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: 100%;
}

.rg-row {
  display: flex;
  justify-content: space-around;
  gap: 8px;
}

.rg-gauge {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.rg-ring {
  position: relative;
  width: 56px;
  height: 56px;
}

.rg-ring svg {
  width: 100%;
  height: 100%;
  transform: rotate(-90deg);
}

.rg-ring-bg {
  fill: none;
  stroke: var(--border);
  stroke-width: 3;
}

.rg-ring-fill {
  fill: none;
  stroke-width: 3;
  stroke-linecap: round;
  transition: stroke-dasharray 0.8s ease;
}

.rg-ring-value {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 9px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}

.rg-label {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.rg-stress {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 0;
}

.rg-stress-title {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  letter-spacing: 0.8px;
  text-transform: uppercase;
}

.rg-stress-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.rg-stress-name {
  font-size: 10px;
  color: var(--text-secondary);
  min-width: 70px;
  white-space: nowrap;
}

.rg-stress-bar-wrap {
  flex: 1;
  height: 6px;
  background: var(--bg-input);
  border-radius: 3px;
  overflow: hidden;
}

.rg-stress-bar {
  height: 100%;
  border-radius: 3px;
  transition: width 0.6s ease;
}

.rg-stress-val {
  font-size: 10px;
  font-weight: 700;
  font-family: var(--font-mono);
  min-width: 42px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.rg-stress-val.severe { color: var(--red); }
.rg-stress-val.moderate { color: var(--orange); }
.rg-stress-val.mild { color: var(--green); }

.rg-empty {
  color: var(--text-dim);
  font-size: 11px;
  text-align: center;
  padding: 20px;
}
</style>
