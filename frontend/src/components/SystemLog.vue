<template>
  <div class="syslog" ref="logRef">
    <div
      v-for="(entry, i) in entries"
      :key="i"
      :class="['syslog-entry', entry.level]"
    >
      <span class="syslog-time">{{ entry.time }}</span>
      <span class="syslog-level">{{ entry.level.toUpperCase() }}</span>
      <span class="syslog-msg">{{ entry.message }}</span>
    </div>
    <div v-if="!entries.length" class="syslog-empty">
      <span class="syslog-cursor">_</span> {{ locale === 'zh-CN' ? '等待流水线活动...' : 'Awaiting pipeline activity...' }}
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import { useI18n } from '../i18n/index.js'

const { locale } = useI18n()

const props = defineProps({
  entries: { type: Array, default: () => [] },
})

const logRef = ref(null)

watch(() => props.entries.length, () => {
  nextTick(() => {
    if (logRef.value) {
      logRef.value.scrollTop = logRef.value.scrollHeight
    }
  })
})
</script>

<style scoped>
.syslog {
  background: #020610;
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  padding: 6px 8px;
  font-family: var(--font-mono);
  font-size: 10px;
  line-height: 1.6;
  overflow-y: auto;
  height: 100%;
  min-height: 0;
}

.syslog-entry {
  display: flex;
  gap: 8px;
  white-space: nowrap;
}

.syslog-time {
  color: var(--text-dim);
  flex-shrink: 0;
  font-variant-numeric: tabular-nums;
}

.syslog-level {
  font-weight: 600;
  min-width: 36px;
  flex-shrink: 0;
}

.syslog-entry.info .syslog-level { color: var(--accent); }
.syslog-entry.success .syslog-level { color: var(--green); }
.syslog-entry.warn .syslog-level { color: var(--orange); }
.syslog-entry.error .syslog-level { color: var(--red); }

.syslog-msg {
  color: var(--text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
}

.syslog-entry.error .syslog-msg { color: var(--red-bright); }
.syslog-entry.success .syslog-msg { color: var(--green-bright); }

.syslog-empty {
  color: var(--text-dim);
  display: flex;
  align-items: center;
  gap: 4px;
}

.syslog-cursor {
  animation: blink 1s step-end infinite;
  color: var(--accent);
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
</style>
