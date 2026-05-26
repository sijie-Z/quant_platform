<template>
  <div>
    <div class="section-header">
      <div>
        <div class="section-title">{{ locale === 'zh-CN' ? '参数网格搜索' : 'Parameter Sweep' }}</div>
        <div class="section-subtitle">{{ locale === 'zh-CN' ? '在优化器、频率和股票池规模间进行网格搜索' : 'Grid search across optimizer, frequency, and universe size' }}</div>
      </div>
    </div>

    <!-- Config Card -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span class="card-title-dot"></span>
          {{ locale === 'zh-CN' ? '扫描配置' : 'Sweep Configuration' }}
        </div>
        <span class="tag tag-purple" v-if="totalCombinations">{{ totalCombinations }} {{ locale === 'zh-CN' ? '种组合' : 'combinations' }}</span>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label for="sweep-opt">{{ locale === 'zh-CN' ? '优化器' : 'Optimizers' }}</label>
          <select id="sweep-opt" v-model="selectedOptimizers" multiple class="select-multiple">
            <option value="equal_weight">{{ locale === 'zh-CN' ? '等权' : 'Equal Weight' }}</option>
            <option value="mean_variance">{{ locale === 'zh-CN' ? '均值方差' : 'Mean Variance' }}</option>
            <option value="risk_parity">{{ locale === 'zh-CN' ? '风险平价' : 'Risk Parity' }}</option>
          </select>
        </div>
        <div class="form-group">
          <label for="sweep-freq">{{ locale === 'zh-CN' ? '频率' : 'Frequencies' }}</label>
          <select id="sweep-freq" v-model="selectedFrequencies" multiple class="select-multiple-short">
            <option value="monthly">{{ locale === 'zh-CN' ? '月度' : 'Monthly' }}</option>
            <option value="weekly">{{ locale === 'zh-CN' ? '周度' : 'Weekly' }}</option>
          </select>
        </div>
        <div class="form-group">
          <label for="sweep-n">{{ locale === 'zh-CN' ? '股票池规模' : 'Universe Size' }}</label>
          <select id="sweep-n" v-model="selectedNStocks" multiple class="select-multiple-short">
            <option :value="100">{{ locale === 'zh-CN' ? '100只股票' : '100 stocks' }}</option>
            <option :value="200">{{ locale === 'zh-CN' ? '200只股票' : '200 stocks' }}</option>
            <option :value="300">{{ locale === 'zh-CN' ? '300只股票' : '300 stocks' }}</option>
          </select>
        </div>
      </div>
      <div class="flex-between mt-2">
        <button class="btn btn-primary" :disabled="loading" @click="runSweep">
          <span v-if="loading">
            <span class="status-spinner" style="display:inline-block;"></span>
            {{ locale === 'zh-CN' ? '扫描中...' : 'Sweeping...' }}
          </span>
          <span v-else>&#9654; {{ locale === 'zh-CN' ? '开始扫描' : 'Run Sweep' }}</span>
        </button>
        <span v-if="loading" class="text-muted text-sm">{{ locale === 'zh-CN' ? '这可能需要几分钟...' : 'This may take several minutes...' }}</span>
      </div>
    </div>

    <div v-if="error" role="alert" class="alert alert-error">
      <span aria-hidden="true">&#10007;</span> {{ error }}
    </div>

    <!-- Best Params -->
    <Transition name="tab-content">
      <div v-if="bestParams" class="best-params">
        <span class="best-params-badge">{{ locale === 'zh-CN' ? '最优' : 'Optimal' }}</span>
        <div class="best-params-info">
          <span>{{ locale === 'zh-CN' ? '优化器：' : 'Optimizer: ' }}<strong>{{ bestParams.optimizer }}</strong></span>
          <span>{{ locale === 'zh-CN' ? '频率：' : 'Frequency: ' }}<strong>{{ bestParams.frequency }}</strong></span>
          <span>{{ locale === 'zh-CN' ? '股票数：' : 'N Stocks: ' }}<strong>{{ bestParams.n_stocks }}</strong></span>
          <span>{{ locale === 'zh-CN' ? '夏普比：' : 'Sharpe: ' }}<strong class="text-green">{{ bestParams.sharpe }}</strong></span>
        </div>
      </div>
    </Transition>

    <!-- Heatmap -->
    <div v-if="heatmapData.length" class="card">
      <div class="card-header">
        <div class="card-title">
          <span class="card-title-dot"></span>
          {{ locale === 'zh-CN' ? '夏普比率热力图' : 'Sharpe Ratio Heatmap' }}
        </div>
        <span class="text-xs text-dim">{{ locale === 'zh-CN' ? '颜色越深 = 夏普比越高' : 'Darker green = higher Sharpe' }}</span>
      </div>
      <div class="heatmap-container">
        <div class="heatmap-block" v-for="hm in heatmapData" :key="hm.optimizer">
          <div class="heatmap-title">{{ hm.optimizer.replace('_', ' ') }}</div>
          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th style="font-size:10px;">N \ {{ locale === 'zh-CN' ? '频率' : 'Freq' }}</th>
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
          {{ locale === 'zh-CN' ? '扫描结果' : 'Sweep Results' }}
        </div>
        <span class="text-xs text-dim">{{ table.length }} {{ locale === 'zh-CN' ? '种组合已测试' : 'combinations tested' }}</span>
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
      <h3>{{ locale === 'zh-CN' ? '网格搜索' : 'Grid Search' }}</h3>
      <p>{{ locale === 'zh-CN' ? '选择优化器、频率和股票池规模，然后点击"开始扫描"以查找最优参数。' : 'Select optimizers, frequencies, and universe sizes, then click "Run Sweep" to find optimal parameters.' }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { sweepParameters } from '../api/index.js'
import { useI18n } from '../i18n/index.js'

const emit = defineEmits(['toast'])
const { $t, locale } = useI18n()

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
    emit('toast', { message: (locale.value === 'zh-CN' ? '扫描完成：' : 'Sweep complete: ') + res.table.length + (locale.value === 'zh-CN' ? '种组合' : ' combinations'), type: 'success' })
  } catch (e) {
    error.value = e.response?.data?.detail || e.message
    emit('toast', { message: locale.value === 'zh-CN' ? '扫描失败' : 'Sweep failed', type: 'error' })
  } finally {
    loading.value = false
  }
}
</script>
