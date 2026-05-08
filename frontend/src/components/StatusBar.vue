<template>
  <footer class="statusbar">
    <div class="statusbar-left">
      <span class="statusbar-dot" :class="connected ? 'alive' : 'dead'"></span>
      <span>{{ connected ? 'CONNECTED' : 'DISCONNECTED' }}</span>
      <span class="statusbar-sep">|</span>
      <span class="text-mono">/api</span>
    </div>
    <div class="statusbar-center">
      <span class="text-dim">READY</span>
    </div>
    <div class="statusbar-right">
      <span class="statusbar-item" @click="$emit('action', 'palette')" title="Command Palette (Ctrl+K)">
        <kbd>Ctrl+K</kbd>
      </span>
      <span class="statusbar-sep">|</span>
      <span class="statusbar-item">R=Run D=Demo E=Export</span>
      <span class="statusbar-sep">|</span>
      <span class="statusbar-item">{{ currentTime }}</span>
    </div>
  </footer>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'

defineProps({ connected: { type: Boolean, default: false } })
defineEmits(['action'])

const currentTime = ref('')
let timer = null

function updateTime() {
  currentTime.value = new Date().toLocaleTimeString('en-US', {
    hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
  })
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
