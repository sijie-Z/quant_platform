<template>
  <header class="terminal-header">
    <div class="th-left">
      <div class="th-logo">Q</div>
      <div class="th-brand">
        <span class="th-name">QUANT TERMINAL</span>
        <span class="th-sub">A-Share Multi-Factor Research Platform</span>
      </div>
      <div class="th-sep"></div>
      <div class="th-status" :class="connected ? 'online' : 'offline'">
        <span class="th-status-dot"></span>
        {{ connected ? 'CONNECTED' : 'OFFLINE' }}
      </div>
    </div>

    <div class="th-center">
      <button
        v-for="view in views"
        :key="view.id"
        :class="['th-view-btn', { active: modelValue === view.id }]"
        @click="$emit('update:modelValue', view.id)"
      >{{ view.label }}</button>
    </div>

    <div class="th-right">
      <div class="th-clock">
        <span class="th-time">{{ time }}</span>
        <span class="th-date">{{ date }}</span>
      </div>
      <div class="th-sep"></div>
      <button class="th-action-btn" @click="$emit('run')" title="Run Pipeline">
        <span class="th-icon">&#9654;</span>
      </button>
      <button class="th-action-btn" @click="$emit('demo')" title="Load Demo">
        <span class="th-icon">&#9889;</span>
      </button>
      <button class="th-action-btn" @click="$emit('command')" title="Command Palette (Ctrl+K)">
        <span class="th-icon">&#8984;K</span>
      </button>
    </div>
  </header>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'

defineProps({
  connected: { type: Boolean, default: false },
  modelValue: { type: String, default: 'terminal' },
  views: { type: Array, default: () => [] },
})

defineEmits(['update:modelValue', 'run', 'demo', 'command'])

const time = ref('')
const date = ref('')
let timer = null

function updateClock() {
  const now = new Date()
  time.value = now.toLocaleTimeString('en-US', { hour12: false })
  date.value = now.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
}

onMounted(() => {
  updateClock()
  timer = setInterval(updateClock, 1000)
})

onBeforeUnmount(() => { if (timer) clearInterval(timer) })
</script>

<style scoped>
.terminal-header {
  height: 40px;
  background: var(--bg-primary);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 12px;
  flex-shrink: 0;
  user-select: none;
  z-index: 200;
}

.th-left, .th-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.th-logo {
  width: 26px;
  height: 26px;
  background: linear-gradient(135deg, var(--accent) 0%, #6366f1 100%);
  border-radius: 5px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 800;
  color: #fff;
  flex-shrink: 0;
}

.th-brand {
  display: flex;
  flex-direction: column;
  line-height: 1.1;
}

.th-name {
  font-size: 12px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: 1.5px;
}

.th-sub {
  font-size: 9px;
  color: var(--text-dim);
  letter-spacing: 0.3px;
}

.th-sep {
  width: 1px;
  height: 20px;
  background: var(--border);
  margin: 0 4px;
}

.th-status {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.8px;
}

.th-status.online { color: var(--green); }
.th-status.offline { color: var(--red); }

.th-status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}

.th-status.online .th-status-dot {
  background: var(--green);
  box-shadow: 0 0 8px rgba(52, 211, 153, 0.5);
  animation: pulse-green 2s ease-in-out infinite;
}

.th-status.offline .th-status-dot {
  background: var(--red);
  box-shadow: 0 0 8px rgba(248, 113, 113, 0.4);
}

@keyframes pulse-green {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.th-center {
  display: flex;
  gap: 2px;
  background: var(--bg-input);
  border-radius: 6px;
  padding: 2px;
  border: 1px solid var(--border-subtle);
}

.th-view-btn {
  padding: 5px 14px;
  border: none;
  background: transparent;
  color: var(--text-dim);
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  cursor: pointer;
  border-radius: 4px;
  transition: all 0.15s ease;
}

.th-view-btn:hover {
  color: var(--text-secondary);
  background: rgba(255,255,255,0.03);
}

.th-view-btn.active {
  background: var(--bg-card);
  color: var(--accent);
}

.th-clock {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  line-height: 1.1;
}

.th-time {
  font-size: 13px;
  font-weight: 600;
  font-family: var(--font-mono);
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}

.th-date {
  font-size: 9px;
  color: var(--text-dim);
  letter-spacing: 0.3px;
}

.th-action-btn {
  width: 28px;
  height: 28px;
  border: 1px solid var(--border);
  background: var(--bg-secondary);
  border-radius: 5px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.15s ease;
  color: var(--text-muted);
}

.th-action-btn:hover {
  border-color: var(--border-light);
  color: var(--text-primary);
  background: var(--bg-tertiary);
}

.th-icon {
  font-size: 11px;
  line-height: 1;
}
</style>
