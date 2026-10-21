<template>
  <footer class="statusbar">
    <div class="statusbar-left">
      <span class="statusbar-dot" :class="connected ? 'alive' : 'dead'"></span>
      <span>{{ connected ? $t('statusBar.connected') : $t('statusBar.disconnected') }}</span>
      <span class="statusbar-sep">|</span>
      <span class="text-mono" :title="apiBaseTitle">{{ apiBaseLabel }}</span>
    </div>
    <div class="statusbar-center" :title="readinessTitle">
      <template v-if="readiness.status === 'degraded'">
        <span class="statusbar-dot degraded"></span>
        <span class="statusbar-degraded">DEGRADED</span>
        <span v-if="readiness.reason" class="statusbar-reason">{{ readiness.reason }}</span>
      </template>
      <template v-else-if="readiness.status === 'ready'">
        <span class="statusbar-dot ready"></span>
        <span>{{ locale === 'zh-CN' ? '就绪' : 'READY' }}</span>
      </template>
      <template v-else>
        <span class="statusbar-dot unknown"></span>
        <span class="text-dim">{{ locale === 'zh-CN' ? '就绪状态未知' : 'READINESS UNKNOWN' }}</span>
      </template>
    </div>
    <div class="statusbar-right">
      <span class="statusbar-item" @click="$emit('action', 'palette')" title="Command Palette (Ctrl+K)">
        <kbd>Ctrl+K</kbd>
      </span>
      <span class="statusbar-sep">|</span>
      <span class="statusbar-item">{{ locale === 'zh-CN' ? 'R=运行 D=示例 E=导出' : 'R=Run D=Demo E=Export' }}</span>
      <span class="statusbar-sep">|</span>
      <span class="statusbar-item">{{ currentTime }}</span>
    </div>
  </footer>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useI18n } from '../i18n/index.js'
import { getApiBase } from '../api/index.js'

const { $t, locale } = useI18n()

const props = defineProps({
  connected: { type: Boolean, default: false },
  readiness: {
    type: Object,
    default: () => ({ status: 'unknown', reason: '', checks: [] }),
  },
})
defineEmits(['action'])

const currentTime = ref('')
const apiBaseLabel = ref(getApiBase())
let timer = null

const apiBaseTitle = computed(() =>
  locale.value === 'zh-CN' ? `实际生效的后端地址：${apiBaseLabel.value}` : `Effective backend URL: ${apiBaseLabel.value}`
)

const readinessTitle = computed(() => {
  if (props.readiness.status === 'ready') {
    return locale.value === 'zh-CN' ? '就绪探针通过' : 'Readiness probe passing'
  }
  if (props.readiness.status === 'degraded') {
    return props.readiness.reason || (locale.value === 'zh-CN' ? '就绪探针失败（503）' : 'Readiness probe failing (503)')
  }
  return props.readiness.reason || (locale.value === 'zh-CN' ? '无法读取就绪状态' : 'Readiness state unavailable')
})

function updateTime() {
  currentTime.value = new Date().toLocaleTimeString(locale.value === 'zh-CN' ? 'zh-CN' : 'en-US', {
    hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
  })
  // Cheap re-read so a Settings change is visible without a reload.
  apiBaseLabel.value = getApiBase()
}

onMounted(() => {
  updateTime()
  timer = setInterval(updateTime, 1000)
})

onBeforeUnmount(() => { if (timer) clearInterval(timer) })
</script>

<style scoped>
.statusbar {
  height: 24px;
  background: var(--bg-primary);
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 12px;
  flex-shrink: 0;
  user-select: none;
  font-size: 10px;
  color: var(--text-dim);
  font-weight: 500;
  letter-spacing: 0.3px;
}

.statusbar-left,
.statusbar-center,
.statusbar-right {
  display: flex;
  align-items: center;
  gap: 8px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.statusbar-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
}

.statusbar-dot.alive {
  background: var(--green);
  box-shadow: 0 0 4px rgba(52,211,153,0.4);
}

.statusbar-dot.dead {
  background: var(--red);
  box-shadow: 0 0 4px rgba(248,113,113,0.3);
}

.statusbar-dot.ready { background: var(--green); }
.statusbar-dot.degraded { background: var(--red); box-shadow: 0 0 4px rgba(248,113,113,0.5); }
.statusbar-dot.unknown { background: var(--text-dim); }

.statusbar-degraded {
  color: var(--red);
  font-weight: 700;
  letter-spacing: 0.5px;
}

.statusbar-reason {
  color: var(--orange);
  max-width: 320px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.statusbar-sep {
  color: var(--border);
  font-size: 10px;
}

.statusbar-item {
  cursor: default;
}

.statusbar-item kbd {
  font-family: var(--font-mono);
  font-size: 9px;
  padding: 1px 5px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 3px;
  cursor: pointer;
  transition: all 0.15s;
}

.statusbar-item kbd:hover {
  border-color: var(--border-light);
  color: var(--text-secondary);
}

.text-mono { font-family: var(--font-mono); }
.text-dim { color: var(--text-dim); }
</style>
