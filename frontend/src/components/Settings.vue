<template>
  <div>
    <div class="section-header">
      <div>
        <div class="section-title">{{ locale === 'zh-CN' ? '设置' : 'Settings' }}</div>
        <div class="section-subtitle">{{ locale === 'zh-CN' ? '配置平台参数和偏好设置' : 'Configure platform parameters and preferences' }}</div>
      </div>
    </div>

    <!-- API Configuration -->
    <div class="settings-section">
      <div class="settings-section-title">
        <span aria-hidden="true">&#9889;</span> {{ locale === 'zh-CN' ? 'API 配置' : 'API Configuration' }}
      </div>
      <div class="settings-row">
        <div>
          <div class="settings-label">{{ locale === 'zh-CN' ? '后端地址' : 'Backend URL' }}</div>
          <div class="text-xs text-dim mt-1">{{ locale === 'zh-CN' ? '用于流水线执行的API服务器地址' : 'API server endpoint for pipeline execution' }}</div>
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
          <div class="settings-label">{{ locale === 'zh-CN' ? 'Tushare Token' : 'Tushare Token' }}</div>
          <div class="text-xs text-dim mt-1">{{ locale === 'zh-CN' ? '获取实时A股数据所需Token（留空使用合成数据）' : 'Required for real A-share data (leave empty for synthetic)' }}</div>
        </div>
        <div style="min-width:220px;">
          <input
            v-model="tushareToken"
            type="password"
            style="padding:7px 12px;background:var(--bg-input);border:1px solid var(--border);border-radius:var(--radius);color:var(--text-primary);font-size:13px;font-family:var(--font-mono);width:100%;"
            :placeholder="locale === 'zh-CN' ? '请输入Token...' : 'Enter token...'"
          />
        </div>
      </div>
    </div>

    <!-- Default Parameters -->
    <div class="settings-section">
      <div class="settings-section-title">
        <span aria-hidden="true">&#9881;</span> {{ locale === 'zh-CN' ? '默认参数' : 'Default Parameters' }}
      </div>
      <div class="form-row">
        <div class="form-group">
          <label for="set-stocks">{{ locale === 'zh-CN' ? '股票数量' : 'Stocks' }}</label>
          <select id="set-stocks" v-model.number="defaults.n_stocks">
            <option :value="100">100</option>
            <option :value="200">200</option>
            <option :value="300">300</option>
            <option :value="500">500</option>
          </select>
        </div>
        <div class="form-group">
          <label for="set-optimizer">{{ locale === 'zh-CN' ? '优化器' : 'Optimizer' }}</label>
          <select id="set-optimizer" v-model="defaults.optimizer">
            <option value="equal_weight">{{ locale === 'zh-CN' ? '等权' : 'Equal Weight' }}</option>
            <option value="mean_variance">{{ locale === 'zh-CN' ? '均值方差' : 'Mean Variance' }}</option>
            <option value="risk_parity">{{ locale === 'zh-CN' ? '风险平价' : 'Risk Parity' }}</option>
          </select>
        </div>
        <div class="form-group">
          <label for="set-alpha">{{ locale === 'zh-CN' ? 'Alpha方法' : 'Alpha Method' }}</label>
          <select id="set-alpha" v-model="defaults.alpha_method">
            <option value="equal_weight">{{ locale === 'zh-CN' ? '等权' : 'Equal Weight' }}</option>
            <option value="ic_weighted">{{ locale === 'zh-CN' ? 'IC加权' : 'IC Weighted' }}</option>
            <option value="icir_weighted">{{ locale === 'zh-CN' ? 'ICIR加权' : 'ICIR Weighted' }}</option>
          </select>
        </div>
        <div class="form-group">
          <label for="set-freq">{{ locale === 'zh-CN' ? '调仓频率' : 'Frequency' }}</label>
          <select id="set-freq" v-model="defaults.rebalance_frequency">
            <option value="monthly">{{ locale === 'zh-CN' ? '月度' : 'Monthly' }}</option>
            <option value="weekly">{{ locale === 'zh-CN' ? '周度' : 'Weekly' }}</option>
          </select>
        </div>
      </div>
    </div>

    <!-- About -->
    <div class="settings-section">
      <div class="settings-section-title">
        <span aria-hidden="true">&#9432;</span> {{ locale === 'zh-CN' ? '关于' : 'About' }}
      </div>
      <div class="settings-row">
        <div class="settings-label">{{ locale === 'zh-CN' ? '平台' : 'Platform' }}</div>
        <div class="settings-value">A-Share Multi-Factor Quant Platform</div>
      </div>
      <div class="settings-row">
        <div class="settings-label">{{ locale === 'zh-CN' ? '版本' : 'Version' }}</div>
        <div class="settings-value">1.0.0</div>
      </div>
      <div class="settings-row">
        <div class="settings-label">{{ locale === 'zh-CN' ? '技术栈' : 'Stack' }}</div>
        <div class="settings-value">FastAPI + Vue 3 + ECharts</div>
      </div>
      <div class="settings-row">
        <div class="settings-label">{{ locale === 'zh-CN' ? '因子数量' : 'Factors' }}</div>
        <div class="settings-value">{{ locale === 'zh-CN' ? '15个（10个技术面 + 5个基本面）' : '15 (10 technical + 5 fundamental)' }}</div>
      </div>
      <div class="settings-row">
        <div class="settings-label">{{ locale === 'zh-CN' ? '优化器' : 'Optimizers' }}</div>
        <div class="settings-value">Equal Weight / MVO / Risk Parity</div>
      </div>
      <div class="settings-row">
        <div class="settings-label">{{ locale === 'zh-CN' ? '测试' : 'Tests' }}</div>
        <div class="settings-value text-green">{{ locale === 'zh-CN' ? '105/105 通过' : '105/105 passing' }}</div>
      </div>
    </div>

    <!-- Data Export -->
    <div class="settings-section">
      <div class="settings-section-title">
        <span aria-hidden="true">&#9744;</span> {{ locale === 'zh-CN' ? '数据导出' : 'Data Export' }}
      </div>
      <div class="text-sm text-muted mb-3">{{ locale === 'zh-CN' ? '导出配置和结果用于备份或共享。' : 'Export configuration and results for backup or sharing.' }}</div>
      <div class="flex-row gap-sm">
        <button class="btn btn-secondary btn-sm" @click="exportConfig">
          &#9744; {{ locale === 'zh-CN' ? '导出配置' : 'Export Config' }}
        </button>
        <button class="btn btn-secondary btn-sm" @click="clearCache">
          &#9850; {{ locale === 'zh-CN' ? '清除本地缓存' : 'Clear Local Cache' }}
        </button>
      </div>
    </div>

    <div class="flex-between mt-4">
      <div></div>
      <button class="btn btn-primary" @click="saveSettings">
        &#10003; {{ locale === 'zh-CN' ? '保存设置' : 'Save Settings' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useI18n } from '../i18n/index.js'

const { $t, locale } = useI18n()

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
  emit('toast', { message: locale.value === 'zh-CN' ? '设置已保存' : 'Settings saved', type: 'success' })
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
  emit('toast', { message: locale.value === 'zh-CN' ? '配置已导出' : 'Config exported', type: 'success' })
}

function clearCache() {
  localStorage.removeItem('quant_settings')
  emit('toast', { message: locale.value === 'zh-CN' ? '本地缓存已清除' : 'Local cache cleared', type: 'info' })
}
</script>
