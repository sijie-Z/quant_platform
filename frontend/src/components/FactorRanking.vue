<template>
  <div>
    <div class="section-header">
      <div>
        <div class="section-title">Factor IC Rankings</div>
        <div class="section-subtitle">Information Coefficient analysis for alpha factor evaluation</div>
      </div>
      <div class="flex-row gap-sm">
        <input
          v-model="search"
          class="search-input"
          placeholder="Search factors..."
          aria-label="Filter factors by name"
        />
        <button class="btn btn-secondary btn-sm" @click="loadDemo" :disabled="loading">
          &#9889; Demo
        </button>
      </div>
    </div>

    <div v-if="loading" class="empty-state">
      <div class="status-spinner" style="width:24px;height:24px;margin:0 auto 12px;"></div>
      <p>Loading factor data...</p>
    </div>

    <div v-else-if="error" role="alert" class="alert alert-error">
      <span aria-hidden="true">&#10007;</span> {{ error }}
    </div>

    <div v-else-if="!result || !result.factors?.length" class="empty-state">
      <div class="empty-icon">&#9776;</div>
      <h3>No Factor Data</h3>
      <p>Run a backtest from the Dashboard tab first, or load demo data.</p>
      <p class="mt-3">
        <button class="btn btn-primary" @click="loadDemo">&#9889; Load Demo Factors</button>
      </p>
    </div>

    <template v-else>
      <!-- Summary Stats -->
      <div class="metrics-grid mb-4">
        <div class="metric-card">
          <div class="metric-value accent">{{ result.factors.length }}</div>
          <div class="metric-label">Total Factors</div>
        </div>
        <div class="metric-card">
          <div class="metric-value positive">{{ positiveFactors }}</div>
          <div class="metric-label">ICIR > 0.2</div>
        </div>
        <div class="metric-card">
          <div class="metric-value negative">{{ negativeFactors }}</div>
          <div class="metric-label">ICIR < -0.2</div>
        </div>
        <div class="metric-card">
          <div :class="['metric-value', bestICIR > 0.3 ? 'positive' : 'neutral']">{{ bestICIR.toFixed(2) }}</div>
          <div class="metric-label">Best |ICIR|</div>
        </div>
      </div>

      <!-- Factor Table -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <span class="card-title-dot"></span>
            Factor Details
          </div>
          <span class="text-xs text-dim">{{ filteredFactors.length }} factors &middot; Click columns to sort</span>
        </div>
        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th style="width:40px;">#</th>
                <th
                  v-for="col in sortableColumns"
                  :key="col.key"
                  :class="['sortable', sortKey === col.key ? 'sorted' : '']"
                  @click="toggleSort(col.key)"
                  @keydown.enter="toggleSort(col.key)"
                  @keydown.space.prevent="toggleSort(col.key)"
                  tabindex="0"
                >
                  {{ col.label }}
                  <span v-if="sortKey === col.key" :class="['sort-icon', sortDir === 'desc' ? 'desc' : '']">&#9650;</span>
                </th>
                <th>IC Distribution</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(f, i) in filteredFactors" :key="f.name">
                <td class="text-dim">{{ i + 1 }}</td>
                <td>
                  <span class="highlight-number">{{ f.name }}</span>
                  <span v-if="i === 0 && sortKey === 'icir'" class="tag tag-green ml-2" style="font-size:9px;">TOP</span>
                </td>
                <td :class="f.mean_ic > 0.02 ? 'positive' : f.mean_ic < -0.02 ? 'negative' : ''">
                  {{ Number(f.mean_ic).toFixed(4) }}
                </td>
                <td class="text-mono">{{ Number(f.std_ic).toFixed(4) }}</td>
                <td>
                  <div style="display:flex;align-items:center;gap:8px;">
                    <span
                      :class="['icir-bar', Math.abs(Number(f.icir)) >= 0.2 ? 'positive-bar' : 'negative-bar']"
                      :style="{ width: Math.min(Math.abs(Number(f.icir)) * 80, 120) + 'px' }"
                      :aria-label="'ICIR: ' + Number(f.icir).toFixed(2)"
                    ></span>
                    <span :class="Math.abs(Number(f.icir)) > 0.3 ? 'positive' : Math.abs(Number(f.icir)) > 0.15 ? 'neutral' : ''" class="text-mono">
                      {{ Number(f.icir).toFixed(2) }}
                    </span>
                  </div>
                </td>
                <td :class="Number(f.ic_positive_ratio) > 0.55 ? 'positive' : Number(f.ic_positive_ratio) < 0.45 ? 'negative' : ''">
                  <span class="text-mono">{{ (Number(f.ic_positive_ratio) * 100).toFixed(1) }}%</span>
                </td>
                <td>
                  <!-- Mini IC bar visualization -->
                  <div style="display:flex;align-items:center;gap:4px;height:18px;">
                    <div :style="{
                      width: Math.abs(f.mean_ic) * 2000 + 'px',
                      height: '6px',
                      borderRadius: '3px',
                      background: f.mean_ic > 0 ? 'var(--green)' : 'var(--red)',
                      opacity: 0.6,
                      maxWidth: '60px',
                    }"></div>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { getFactors, getDemo } from '../api/index.js'

const emit = defineEmits(['toast'])

const result = ref(null)
const loading = ref(false)
const error = ref(null)
const search = ref('')
const sortKey = ref('icir')
const sortDir = ref('desc')

const sortableColumns = [
  { key: 'name', label: 'Factor' },
  { key: 'mean_ic', label: 'Mean IC' },
  { key: 'std_ic', label: 'Std IC' },
  { key: 'icir', label: 'ICIR' },
  { key: 'ic_positive_ratio', label: 'IC>0 %' },
]

const positiveFactors = computed(() =>
  (result.value?.factors || []).filter(f => Math.abs(Number(f.icir)) > 0.2).length
)

const negativeFactors = computed(() =>
  (result.value?.factors || []).filter(f => Number(f.icir) < -0.2).length
)

const bestICIR = computed(() => {
  const factors = result.value?.factors || []
  if (!factors.length) return 0
  return Math.max(...factors.map(f => Math.abs(Number(f.icir))))
})

const filteredFactors = computed(() => {
  if (!result.value?.factors) return []
  let list = [...result.value.factors]
  if (search.value.trim()) {
    const q = search.value.toLowerCase()
    list = list.filter(f => f.name.toLowerCase().includes(q))
  }
  const key = sortKey.value
  const dir = sortDir.value === 'asc' ? 1 : -1
  list.sort((a, b) => {
    if (key === 'name') return (a.name || '').localeCompare(b.name || '') * dir
    const va = parseFloat(a[key]) || 0
    const vb = parseFloat(b[key]) || 0
    return (va - vb) * dir
  })
  return list
})

function toggleSort(key) {
  if (sortKey.value === key) {
    sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortKey.value = key
    sortDir.value = 'desc'
  }
}

async function loadDemo() {
  loading.value = true
  error.value = null
  try {
    const demo = await getDemo()
    result.value = { factors: demo.factors }
    emit('toast', { message: `Loaded ${demo.factors.length} demo factors`, type: 'success' })
  } catch (e) {
    error.value = e.response?.data?.detail || e.message
    emit('toast', { message: 'Failed to load demo', type: 'error' })
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  loading.value = true
  error.value = null
  try {
    result.value = await getFactors()
  } catch (e) {
    error.value = e.response?.data?.detail || e.message
    emit('toast', { message: 'Failed to load factor data', type: 'error' })
  } finally {
    loading.value = false
  }
})
</script>
