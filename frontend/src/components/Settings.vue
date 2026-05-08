<template>
  <div>
    <div class="section-header">
      <div>
        <div class="section-title">Settings</div>
        <div class="section-subtitle">Configure platform parameters and preferences</div>
      </div>
    </div>

    <!-- API Configuration -->
    <div class="settings-section">
      <div class="settings-section-title">
        <span aria-hidden="true">&#9889;</span> API Configuration
      </div>
      <div class="settings-row">
        <div>
          <div class="settings-label">Backend URL</div>
          <div class="text-xs text-dim mt-1">API server endpoint for pipeline execution</div>
        </div>
        <div style="min-width:220px;">
          <input
            v-model="apiBase"
            class="form-group"
            style="margin:0;padding:7px 12px;background:var(--bg-input);border:1px solid var(--border);border-radius:var(--radius);color:var(--text-primary);font-size:13px;font-family:var(--font-mono);width:100%;"
            placeholder="/api"
          />
        </div>
      </div>
      <div class="settings-row">
        <div>
          <div class="settings-label">Tushare Token</div>
          <div class="text-xs text-dim mt-1">Required for real A-share data (leave empty for synthetic)</div>
        </div>
        <div style="min-width:220px;">
          <input
            v-model="tushareToken"
            type="password"
            style="padding:7px 12px;background:var(--bg-input);border:1px solid var(--border);border-radius:var(--radius);color:var(--text-primary);font-size:13px;font-family:var(--font-mono);width:100%;"
            placeholder="Enter token..."
          />
        </div>
      </div>
    </div>

    <!-- Default Parameters -->
    <div class="settings-section">
      <div class="settings-section-title">
        <span aria-hidden="true">&#9881;</span> Default Parameters
      </div>
      <div class="form-row">
        <div class="form-group">
          <label for="set-stocks">Stocks</label>
          <select id="set-stocks" v-model.number="defaults.n_stocks">
            <option :value="100">100</option>
            <option :value="200">200</option>
            <option :value="300">300</option>
            <option :value="500">500</option>
          </select>
        </div>
        <div class="form-group">
          <label for="set-optimizer">Optimizer</label>
          <select id="set-optimizer" v-model="defaults.optimizer">
            <option value="equal_weight">Equal Weight</option>
            <option value="mean_variance">Mean Variance</option>
            <option value="risk_parity">Risk Parity</option>
          </select>
        </div>
        <div class="form-group">
          <label for="set-alpha">Alpha Method</label>
          <select id="set-alpha" v-model="defaults.alpha_method">
            <option value="equal_weight">Equal Weight</option>
            <option value="ic_weighted">IC Weighted</option>
            <option value="icir_weighted">ICIR Weighted</option>
          </select>
        </div>
        <div class="form-group">
          <label for="set-freq">Frequency</label>
          <select id="set-freq" v-model="defaults.rebalance_frequency">
            <option value="monthly">Monthly</option>
            <option value="weekly">Weekly</option>
          </select>
        </div>
      </div>
    </div>

    <!-- About -->
    <div class="settings-section">
      <div class="settings-section-title">
        <span aria-hidden="true">&#9432;</span> About
      </div>
      <div class="settings-row">
        <div class="settings-label">Platform</div>
        <div class="settings-value">A-Share Multi-Factor Quant Platform</div>
      </div>
      <div class="settings-row">
        <div class="settings-label">Version</div>
        <div class="settings-value">1.0.0</div>
      </div>
      <div class="settings-row">
        <div class="settings-label">Stack</div>
        <div class="settings-value">FastAPI + Vue 3 + ECharts</div>
      </div>
      <div class="settings-row">
        <div class="settings-label">Factors</div>
        <div class="settings-value">15 (10 technical + 5 fundamental)</div>
      </div>
      <div class="settings-row">
        <div class="settings-label">Optimizers</div>
        <div class="settings-value">Equal Weight / MVO / Risk Parity</div>
      </div>
      <div class="settings-row">
        <div class="settings-label">Tests</div>
        <div class="settings-value text-green">105/105 passing</div>
      </div>
    </div>

    <!-- Data Export -->
    <div class="settings-section">
      <div class="settings-section-title">
        <span aria-hidden="true">&#9744;</span> Data Export
      </div>
      <div class="text-sm text-muted mb-3">Export configuration and results for backup or sharing.</div>
      <div class="flex-row gap-sm">
        <button class="btn btn-secondary btn-sm" @click="exportConfig">
          &#9744; Export Config
        </button>
        <button class="btn btn-secondary btn-sm" @click="clearCache">
          &#9850; Clear Local Cache
        </button>
      </div>
    </div>

    <div class="flex-between mt-4">
      <div></div>
      <button class="btn btn-primary" @click="saveSettings">
        &#10003; Save Settings
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'

const emit = defineEmits(['toast'])

const apiBase = ref('/api')
const tushareToken = ref('')
const defaults = reactive({
  n_stocks: 300,
  optimizer: 'mean_variance',
  alpha_method: 'icir_weighted',
  rebalance_frequency: 'monthly',
})

onMounted(() => {
  const saved = localStorage.getItem('quant_settings')
  if (saved) {
    try {
      const s = JSON.parse(saved)
      if (s.apiBase) apiBase.value = s.apiBase
      if (s.tushareToken) tushareToken.value = s.tushareToken
      if (s.defaults) Object.assign(defaults, s.defaults)
    } catch {}
  }
})

function saveSettings() {
  localStorage.setItem('quant_settings', JSON.stringify({
    apiBase: apiBase.value,
    tushareToken: tushareToken.value,
    defaults: { ...defaults },
  }))
  emit('toast', { message: 'Settings saved', type: 'success' })
}

function exportConfig() {
  const data = {
    apiBase: apiBase.value,
    defaults: { ...defaults },
    exportedAt: new Date().toISOString(),
  }
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'quant-platform-config.json'
  a.click()
  URL.revokeObjectURL(url)
  emit('toast', { message: 'Config exported', type: 'success' })
}

function clearCache() {
  localStorage.removeItem('quant_settings')
  emit('toast', { message: 'Local cache cleared', type: 'info' })
}
</script>
