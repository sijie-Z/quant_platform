<template>
  <div class="drawdown-periods">
    <div v-if="!periods || !periods.length" class="dp-empty">No drawdown periods</div>
    <div v-else class="dp-scroll">
      <table class="dp-tbl">
        <thead>
          <tr>
            <th class="dp-rank">#</th>
            <th>Start</th>
            <th>Trough</th>
            <th>End</th>
            <th class="dp-depth">Depth</th>
            <th class="dp-bar">Severity</th>
            <th class="dp-days">Days</th>
            <th class="dp-status">Status</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(p, i) in periods" :key="i" class="dp-row">
            <td class="dp-rank">{{ i + 1 }}</td>
            <td class="dp-date">{{ p.start }}</td>
            <td class="dp-date dp-trough">{{ p.trough }}</td>
            <td class="dp-date">{{ p.recovered ? p.end : '--' }}</td>
            <td :class="['dp-depth', depthClass(p.depth)]">
              {{ (p.depth * 100).toFixed(1) }}%
            </td>
            <td class="dp-bar">
              <div class="dp-bar-wrap">
                <div
                  class="dp-bar-fill"
                  :style="{ width: severityPct(p.depth) + '%' }"
                  :class="depthClass(p.depth)"
                ></div>
              </div>
            </td>
            <td class="dp-days">{{ p.duration_days }}d</td>
            <td :class="['dp-status', p.recovered ? 'dp-recovered' : 'dp-ongoing']">
              {{ p.recovered ? 'RECOVERED' : 'ONGOING' }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
defineProps({
  periods: { type: Array, default: () => [] },
})

function depthClass(depth) {
  const d = Math.abs(depth)
  if (d > 0.2) return 'dp-severe'
  if (d > 0.1) return 'dp-moderate'
  return 'dp-mild'
}

function severityPct(depth) {
  return Math.min(Math.abs(depth) * 100 / 0.35 * 100, 100)
}
</script>

<style scoped>
.drawdown-periods {
  height: 100%;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.dp-scroll {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
}

.dp-scroll::-webkit-scrollbar { width: 4px; }
.dp-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.dp-tbl {
  width: 100%;
  border-collapse: collapse;
  font-size: 10px;
  font-variant-numeric: tabular-nums;
}

.dp-tbl th {
  padding: 4px 6px;
  font-size: 8px;
  font-weight: 700;
  color: var(--text-dim);
  letter-spacing: 0.6px;
  text-transform: uppercase;
  text-align: left;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
  white-space: nowrap;
  position: sticky;
  top: 0;
  z-index: 1;
}

.dp-tbl td {
  padding: 3px 6px;
  border-bottom: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  white-space: nowrap;
}

.dp-row:hover td { background: rgba(77, 166, 255, 0.04); }

.dp-rank { width: 24px; text-align: center; color: var(--text-dim); font-size: 9px; }
.dp-date { font-family: var(--font-mono); font-size: 9px; color: var(--text-muted); }
.dp-trough { color: var(--text-secondary); font-weight: 600; }

.dp-depth {
  text-align: right;
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 10px;
}

.dp-severe { color: var(--red); }
.dp-moderate { color: var(--orange); }
.dp-mild { color: var(--green); }

.dp-bar { width: 60px; }
.dp-bar-wrap {
  height: 4px;
  background: var(--bg-input);
  border-radius: 2px;
  overflow: hidden;
}

.dp-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.4s ease;
  min-width: 2px;
}

.dp-bar-fill.dp-severe { background: linear-gradient(90deg, var(--red-dim), var(--red)); }
.dp-bar-fill.dp-moderate { background: linear-gradient(90deg, var(--orange), var(--gold)); }
.dp-bar-fill.dp-mild { background: linear-gradient(90deg, var(--green-dim), var(--green)); }

.dp-days { text-align: right; font-family: var(--font-mono); font-size: 9px; }

.dp-status {
  text-align: center;
  font-size: 8px;
  font-weight: 700;
  letter-spacing: 0.5px;
}

.dp-recovered { color: var(--green); }
.dp-ongoing { color: var(--red); }

.dp-empty {
  color: var(--text-dim);
  font-size: 11px;
  text-align: center;
  padding: 20px;
}
</style>
