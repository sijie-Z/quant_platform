<template>
  <div>
    <div class="section-header">
      <div>
        <div class="section-title">Strategy Comparison</div>
        <div class="section-subtitle">Compare portfolio optimizers side by side</div>
      </div>
    </div>

    <!-- Config -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span class="card-title-dot"></span>
          Comparison Setup
        </div>
        <span class="tag tag-accent" v-if="selectedOptimizers.length">{{ selectedOptimizers.length }} strategies</span>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label for="cmp-opt">Optimizers</label>
          <select id="cmp-opt" v-model="selectedOptimizers" multiple class="select-multiple">
            <option value="equal_weight">Equal Weight</option>
            <option value="mean_variance">Mean Variance</option>
            <option value="risk_parity">Risk Parity</option>
          </select>
          <div class="form-hint">Hold Ctrl/Cmd to select multiple</div>
        </div>
        <div class="form-group">
          <label for="cmp-n">Universe Size</label>
          <select id="cmp-n" v-model.number="nStocks">
            <option :value="100">100 stocks</option>
            <option :value="200">200 stocks</option>
            <option :value="300">300 stocks</option>
            <option :value="500">500 stocks</option>
          </select>
        </div>
      </div>
      <button class="btn btn-primary" :disabled="loading" @click="runCompare">
        <span v-if="loading">
          <span class="status-spinner" style="display:inline-block;"></span>
          Comparing...
        </span>
        <span v-else>&#9654; Compare Strategies</span>
      </button>
    </div>

    <div v-if="error" role="alert" class="alert alert-error">
      <span aria-hidden="true">&#10007;</span> {{ error }}
    </div>

    <!-- Results -->
    <div v-if="table.length" class="card">
      <div class="card-header">
        <div class="card-title">
          <span class="card-title-dot"></span>
          Comparison Results
        </div>
        <span class="text-xs text-dim">{{ table.length }} strategies compared</span>
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
                :class="getCellClass(col, row[col])"
              >
                <span v-if="col === 'Optimizer'" class="highlight-number">{{ row[col] }}</span>
                <span v-else>{{ row[col] }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Empty -->
    <div v-if="!table.length && !loading && !error" class="empty-state">
      <div class="empty-icon">&#9878;</div>
      <h3>Compare Strategies</h3>
      <p>Select two or more portfolio optimizers and click "Compare" to see performance differences.</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { compareStrategies } from '../api/index.js'

const emit = defineEmits(['toast'])

const selectedOptimizers = ref(['equal_weight', 'mean_variance', 'risk_parity'])
const nStocks = ref(300)
const loading = ref(false)
const error = ref(null)
const table = ref([])
const columns = ref([])

function getCellClass(col, val) {
  if (col === 'Optimizer') return ''
  const v = parseFloat(val)
  if (isNaN(v)) return ''
  if (col === 'Sharpe' || col === 'Sortino' || col === 'Calmar') {
    return v >= 1 ? 'positive' : v >= 0 ? 'neutral' : 'negative'
  }
  if (col.includes('Return') || col.includes('Ret')) return v >= 0 ? 'positive' : 'negative'
  if (col.includes('DD') || col.includes('Drawdown')) return v <= -30 ? 'negative' : 'neutral'
  if (col.includes('Win Rate')) return v >= 50 ? 'positive' : 'negative'
  return ''
}

async function runCompare() {
  error.value = null
  loading.value = true
  table.value = []
  columns.value = []
  try {
    const res = await compareStrategies({
      optimizers: selectedOptimizers.value,
      n_stocks: nStocks.value,
    })
    table.value = res.table
    if (res.table.length) {
      columns.value = Object.keys(res.table[0])
    }
    emit('toast', { message: `Compared ${res.table.length} strategies`, type: 'success' })
  } catch (e) {
    error.value = e.response?.data?.detail || e.message
    emit('toast', { message: 'Comparison failed', type: 'error' })
  } finally {
    loading.value = false
  }
}
</script>
