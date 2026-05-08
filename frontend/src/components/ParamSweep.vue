<template>
  <div>
    <div class="section-header">
      <div>
        <div class="section-title">Parameter Sweep</div>
        <div class="section-subtitle">Grid search across optimizer, frequency, and universe size</div>
      </div>
    </div>

    <!-- Config Card -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span class="card-title-dot"></span>
          Sweep Configuration
        </div>
        <span class="tag tag-purple" v-if="totalCombinations">{{ totalCombinations }} combinations</span>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label for="sweep-opt">Optimizers</label>
          <select id="sweep-opt" v-model="selectedOptimizers" multiple class="select-multiple">
            <option value="equal_weight">Equal Weight</option>
            <option value="mean_variance">Mean Variance</option>
            <option value="risk_parity">Risk Parity</option>
          </select>
        </div>
        <div class="form-group">
          <label for="sweep-freq">Frequencies</label>
          <select id="sweep-freq" v-model="selectedFrequencies" multiple class="select-multiple-short">
            <option value="monthly">Monthly</option>
            <option value="weekly">Weekly</option>
          </select>
        </div>
        <div class="form-group">
          <label for="sweep-n">Universe Size</label>
          <select id="sweep-n" v-model="selectedNStocks" multiple class="select-multiple-short">
            <option :value="100">100 stocks</option>
            <option :value="200">200 stocks</option>
            <option :value="300">300 stocks</option>
          </select>
        </div>
      </div>
      <div class="flex-between mt-2">
        <button class="btn btn-primary" :disabled="loading" @click="runSweep">
          <span v-if="loading">
            <span class="status-spinner" style="display:inline-block;"></span>
            Sweeping...
          </span>
          <span v-else>&#9654; Run Sweep</span>
        </button>
        <span v-if="loading" class="text-muted text-sm">This may take several minutes...</span>
      </div>
    </div>

    <div v-if="error" role="alert" class="alert alert-error">
      <span aria-hidden="true">&#10007;</span> {{ error }}
    </div>

    <!-- Best Params -->
    <Transition name="tab-content">
      <div v-if="bestParams" class="best-params">
        <span class="best-params-badge">Optimal</span>
        <div class="best-params-info">
          <span>Optimizer: <strong>{{ bestParams.optimizer }}</strong></span>
          <span>Frequency: <strong>{{ bestParams.frequency }}</strong></span>
          <span>N Stocks: <strong>{{ bestParams.n_stocks }}</strong></span>
          <span>Sharpe: <strong class="text-green">{{ bestParams.sharpe }}</strong></span>
        </div>
      </div>
    </Transition>

    <!-- Heatmap -->
    <div v-if="heatmapData.length" class="card">
      <div class="card-header">
        <div class="card-title">
          <span class="card-title-dot"></span>
          Sharpe Ratio Heatmap
        </div>
        <span class="text-xs text-dim">Darker green = higher Sharpe</span>
      </div>
      <div class="heatmap-container">
        <div class="heatmap-block" v-for="hm in heatmapData" :key="hm.optimizer">
          <div class="heatmap-title">{{ hm.optimizer.replace('_', ' ') }}</div>
          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th style="font-size:10px;">N \ Freq</th>
                  <th v-for="freq in hm.frequencies" :key="freq">{{ freq }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="n in hm.nStocksList" :key="n">
                  <th>{{ n }}</th>
                  <td
                    v-for="freq in hm.frequencies"
                    :key="freq"
                    :style="{ background: cellColor(hm.grid[freq]?.[n], hm.minS, hm.maxS), color: cellTextColor(hm.grid[freq]?.[n]) }"
                    class="heatmap-cell-value"
                  >
                    {{ hm.grid[freq]?.[n] ?? '-' }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- Results Table -->
    <div v-if="table.length" class="card">
      <div class="card-header">
        <div class="card-title">
          <span class="card-title-dot"></span>
          Sweep Results
        </div>
        <span class="text-xs text-dim">{{ table.length }} combinations tested</span>
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th v-for="col in columns" :key="col">{{ col }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, i) in table" :key="i">
              <td v-for="col in columns" :key="col"
                :class="getCellClass(col, row)"
              >{{ row[col] }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Empty -->
    <div v-if="!table.length && !loading && !error" class="empty-state">
      <div class="empty-icon">&#9881;</div>
      <h3>Grid Search</h3>
      <p>Select optimizers, frequencies, and universe sizes, then click "Run Sweep" to find optimal parameters.</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { sweepParameters } from '../api/index.js'

const emit = defineEmits(['toast'])

const selectedOptimizers = ref(['equal_weight', 'mean_variance', 'risk_parity'])
const selectedFrequencies = ref(['monthly', 'weekly'])
const selectedNStocks = ref([200, 300])
const loading = ref(false)
const error = ref(null)
const table = ref([])
const columns = ref([])
const bestParams = ref(null)

const totalCombinations = computed(() =>
  selectedOptimizers.value.length * selectedFrequencies.value.length * selectedNStocks.value.length
)

const bestSharpe = computed(() => bestParams.value?.sharpe)

function isBestRow(row) {
  return bestSharpe.value != null && parseFloat(row['Sharpe']) === bestSharpe.value
}

function getCellClass(col, row) {
  if (col === 'Sharpe') {
    if (isBestRow(row)) return 'positive'
    const v = parseFloat(row[col])
    if (isNaN(v)) return ''
    return v >= 1 ? 'positive' : v >= 0 ? 'neutral' : 'negative'
  }
  if (col === 'Ann. Ret %') {
    const v = parseFloat(row[col])
    if (isNaN(v)) return ''
    return v >= 0 ? 'positive' : 'negative'
  }
  if (col === 'Max DD %') {
    const v = parseFloat(row[col])
    if (isNaN(v)) return ''
    return v <= -30 ? 'negative' : 'neutral'
  }
  return ''
}

const heatmapData = computed(() => {
  if (!table.value.length) return []
  const optimizers = [...new Set(table.value.map(r => r['Optimizer']).filter(Boolean))]
  const frequencies = [...new Set(table.value.map(r => r['Frequency']).filter(Boolean))]
  const nStocksList = [...new Set(table.value.map(r => r['N Stocks']).filter(Boolean))].sort((a, b) => a - b)

  const sharpes = table.value.map(r => parseFloat(r['Sharpe'])).filter(v => !isNaN(v))
  const minS = sharpes.length ? Math.min(...sharpes) : -0.5
  const maxS = sharpes.length ? Math.max(...sharpes) : 2.0

  return optimizers.map(opt => {
    const grid = {}
    for (const freq of frequencies) {
      grid[freq] = {}
      for (const n of nStocksList) {
        const row = table.value.find(r => r['Optimizer'] === opt && r['Frequency'] === freq && r['N Stocks'] === n)
        grid[freq][n] = row ? parseFloat(row['Sharpe']) : null
      }
    }
    return { optimizer: opt, frequencies, nStocksList, grid, minS, maxS }
  })
})

function cellColor(v, minS, maxS) {
  if (v == null || isNaN(v)) return 'transparent'
  const range = (maxS - minS) || 1
  const t = Math.max(0, Math.min(1, (v - minS) / range))
  const r = Math.round(12 + (1 - t) * 8)
  const g = Math.round(25 + t * 110)
  const b = Math.round(40 + t * 10)
  return `rgb(${r},${g},${b})`
}

function cellTextColor(v) {
  if (v == null || isNaN(v)) return '#556882'
  return v >= 0.8 ? '#e8edf5' : '#8b9dc0'
}

async function runSweep() {
  error.value = null
  bestParams.value = null
  loading.value = true
  table.value = []
  columns.value = []
  try {
    const res = await sweepParameters({
      optimizers: selectedOptimizers.value,
      frequencies: selectedFrequencies.value,
      n_stocks_list: selectedNStocks.value,
    })
    table.value = res.table
    bestParams.value = res.best_params
    if (res.table.length) {
      columns.value = Object.keys(res.table[0])
    }
    emit('toast', { message: `Sweep complete: ${res.table.length} combinations`, type: 'success' })
  } catch (e) {
    error.value = e.response?.data?.detail || e.message
    emit('toast', { message: 'Sweep failed', type: 'error' })
  } finally {
    loading.value = false
  }
}
</script>
