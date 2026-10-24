<template>
  <div>
    <div class="section-header">
      <div>
        <div class="section-title">{{ locale === 'zh-CN' ? '设置' : 'Settings' }}</div>
        <div class="section-subtitle">{{ locale === 'zh-CN' ? '浏览器本地偏好（不会修改服务端配置）' : 'Browser-local preferences (does not change server config)' }}</div>
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
          <div class="text-xs text-dim mt-1">{{ locale === 'zh-CN' ? 'REST 请求与 WebSocket 都使用该地址；保存后立即生效，刷新页面后依然有效' : 'Used by REST requests and the status WebSocket; applies immediately and survives reloads' }}</div>
        </div>
        <div style="min-width:220px;">
          <input
            v-model="apiBase"
            class="form-group"
            style="margin:0;padding:7px 12px;background:var(--bg-input);border:1px solid var(--border);border-radius:var(--radius);color:var(--text-primary);font-size:13px;font-family:var(--font-mono);width:100%;"
            placeholder="/api"
          />
          <div class="text-xs mt-1">
            <span class="text-dim">{{ locale === 'zh-CN' ? '实际生效：' : 'Effective: ' }}</span>
            <code class="effective-url">{{ effectiveApiBase }}</code>
            <div v-if="!baseValid" class="text-red">{{ locale === 'zh-CN' ? '地址无效（需为 /path 或 http(s)://host），保存将被拒绝；当前回退为 /api' : 'Invalid URL (expected /path or http(s)://host); saving is blocked and /api is used' }}</div>
          </div>
        </div>
      </div>
      <div class="settings-row">
        <div>
          <div class="settings-label">{{ locale === 'zh-CN' ? 'API Token' : 'API Token' }}</div>
          <div class="text-xs text-dim mt-1">{{ locale === 'zh-CN' ? '服务端 QUANT_API_TOKEN；仅保存在本浏览器，不会打包进前端' : 'Server QUANT_API_TOKEN; stored only in this browser, never baked into the bundle' }}</div>
        </div>
        <div style="min-width:220px;">
          <input
            v-model="apiToken"
            type="password"
            style="padding:7px 12px;background:var(--bg-input);border:1px solid var(--border);border-radius:var(--radius);color:var(--text-primary);font-size:13px;font-family:var(--font-mono);width:100%;"
            :placeholder="locale === 'zh-CN' ? '请输入API Token...' : 'Enter API token...'"
          />
        </div>
      </div>
      <div class="settings-row">
        <div>
          <div class="settings-label">{{ locale === 'zh-CN' ? 'Tushare Token' : 'Tushare Token' }}</div>
          <div class="text-xs text-dim mt-1">
            {{ locale === 'zh-CN'
              ? '由服务端环境变量 TUSHARE_TOKEN 配置（服务端读取，前端不可设置）；未配置时使用合成数据'
              : 'Configured server-side via the TUSHARE_TOKEN environment variable (never settable from the browser); synthetic data is used when absent' }}
          </div>
        </div>
        <div style="min-width:220px;">
          <div class="settings-value text-dim">TUSHARE_TOKEN <span aria-hidden="true">&#128274;</span></div>
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
        <div class="settings-value">{{ locale === 'zh-CN' ? '27 个已注册（默认启用 21 个）' : '27 registered (21 enabled by default)' }}</div>
      </div>
      <div class="settings-row">
        <div class="settings-label">{{ locale === 'zh-CN' ? '优化器' : 'Optimizers' }}</div>
        <div class="settings-value">Equal Weight / MVO / Risk Parity</div>
      </div>
      <div class="settings-row">
        <div class="settings-label">{{ locale === 'zh-CN' ? '测试' : 'Tests' }}</div>
        <div class="settings-value text-green">{{ locale === 'zh-CN' ? '2,843 通过 / 3 跳过' : '2,843 passed / 3 skipped' }}</div>
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
        &#10003; {{ locale === 'zh-CN' ? '保存本地偏好' : 'Save Local Preferences' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { useI18n } from '../i18n/index.js'
import { getApiBase, normalizeApiBase, isValidApiBase } from '../api/index.js'

const { $t, locale } = useI18n()

const emit = defineEmits(['toast'])

const apiBase = ref(getApiBase())
const apiToken = ref(localStorage.getItem('quant_api_token') || '')
const defaults = reactive({
  n_stocks: 300,
  optimizer: 'mean_variance',
  alpha_method: 'icir_weighted',
  rebalance_frequency: 'monthly',
})

const effectiveApiBase = computed(() => normalizeApiBase(apiBase.value))
const baseValid = computed(() => isValidApiBase(apiBase.value))

onMounted(() => {
  const saved = localStorage.getItem('quant_settings')
  if (saved) {
    try {
      const s = JSON.parse(saved)
      if (s.apiBase) apiBase.value = s.apiBase
      if (s.apiToken) apiToken.value = s.apiToken
      if (s.defaults) Object.assign(defaults, s.defaults)
    } catch {}
  }
})

function saveSettings() {
  if (!isValidApiBase(apiBase.value)) {
    emit('toast', {
      message: locale.value === 'zh-CN'
        ? '后端地址无效：请填写 /api 或 http(s)://主机[:端口]'
        : 'Invalid backend URL: use /api or http(s)://host[:port]',
      type: 'error',
    })
    return
  }
  // Persist the normalized form so the effective URL is unambiguous. The
  // axios request interceptor and the WebSocket factory read this at request/
  // connect time, so both switch over without a rebuild.
  const normalized = normalizeApiBase(apiBase.value)
  apiBase.value = normalized
  // The API token is stored under its own key so the axios/WS clients can
  // read it at request time; putting it in the Vite env bakes it into the
  // public bundle.
  localStorage.setItem('quant_api_token', apiToken.value)
  localStorage.setItem('quant_settings', JSON.stringify({
    apiBase: normalized,
    // The token is intentionally NOT duplicated here; `quant_api_token` is the
    // single key the axios/WS clients read.
    defaults: { ...defaults },
  }))
  emit('toast', {
    message: locale.value === 'zh-CN'
      ? `已保存到本浏览器；后端地址 ${normalized} 已对之后的请求生效`
      : `Saved in this browser; backend URL ${normalized} now applies to subsequent requests`,
    type: 'success',
  })
}

function exportConfig() {
  const data = {
    apiBase: effectiveApiBase.value,
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
  apiBase.value = getApiBase()
  emit('toast', { message: locale.value === 'zh-CN' ? '本地缓存已清除，后端地址已恢复为 /api' : 'Local cache cleared; backend URL reset to /api', type: 'info' })
}
</script>

<style scoped>
.effective-url {
  background: var(--bg-input);
  padding: 1px 6px;
  border-radius: 3px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--accent);
  word-break: break-all;
}
</style>
