<template>
  <Teleport to="body">
    <Transition name="palette">
      <div v-if="open" class="palette-overlay" @mousedown.self="close">
        <div class="palette" role="dialog" aria-label="Command palette">
          <div class="palette-input-wrap">
            <span class="palette-icon" aria-hidden="true">&#8984;</span>
            <input
              ref="inputRef"
              v-model="query"
              class="palette-input"
              placeholder="Type a command or search..."
              @keydown.escape="close"
              @keydown.enter="execute"
              @keydown.up.prevent="moveUp"
              @keydown.down.prevent="moveDown"
              @keydown.tab.prevent="moveDown"
            />
            <kbd class="palette-kbd">ESC</kbd>
          </div>
          <div class="palette-divider"></div>
          <div class="palette-list" ref="listRef">
            <div
              v-for="(item, i) in filtered"
              :key="item.id"
              :class="['palette-item', { active: i === selected }]"
              @click="run(item)"
              @mouseenter="selected = i"
            >
              <span class="palette-item-icon" aria-hidden="true">{{ item.icon }}</span>
              <div class="palette-item-text">
                <div class="palette-item-label">{{ item.label }}</div>
                <div v-if="item.hint" class="palette-item-hint">{{ item.hint }}</div>
              </div>
              <kbd v-if="item.shortcut" class="palette-kbd-sm">{{ item.shortcut }}</kbd>
            </div>
            <div v-if="!filtered.length" class="palette-empty">No results for "{{ query }}"</div>
          </div>
          <div class="palette-footer">
            <span><kbd class="palette-kbd-sm">&#8593;&#8595;</kbd> navigate</span>
            <span><kbd class="palette-kbd-sm">&#9166;</kbd> select</span>
            <span><kbd class="palette-kbd-sm">esc</kbd> close</span>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted, onBeforeUnmount } from 'vue'

const props = defineProps({
  modelValue: { type: String, default: 'dashboard' },
})

const emit = defineEmits(['update:modelValue', 'action', 'toast'])

const open = ref(false)
const query = ref('')
const selected = ref(0)
const inputRef = ref(null)
const listRef = ref(null)

const commands = [
  { id: 'terminal',   icon: '▸', label: 'Terminal',        hint: 'Main backtest dashboard', shortcut: '1', tab: 'terminal' },
  { id: 'trading',    icon: '⚡', label: 'Live Trading',    hint: 'Paper/Live trading engine', shortcut: '2', tab: 'trading' },
  { id: 'live',       icon: '◉', label: 'Live Portfolio',  hint: 'Real-time portfolio tracker', shortcut: '3', tab: 'live' },
  { id: 'oms',        icon: '⊞', label: 'Order Blotter',   hint: 'OMS order management',    shortcut: '4', tab: 'oms' },
  { id: 'compare',    icon: '⇄', label: 'Compare',         hint: 'Strategy comparison',     shortcut: '5', tab: 'compare' },
  { id: 'sweep',      icon: '⌖', label: 'Sweep',           hint: 'Parameter grid search',   shortcut: '6', tab: 'sweep' },
  { id: 'factors',    icon: '≡', label: 'Factors',         hint: 'Factor IC rankings',      shortcut: '7', tab: 'factors' },
  { id: 'history',    icon: '⏱', label: 'History',         hint: 'Run history',             shortcut: '8', tab: 'history' },
  { id: 'settings',   icon: '⚙', label: 'Settings',        hint: 'Platform configuration',  shortcut: '9', tab: 'settings' },
  { id: 'run',        icon: '▶', label: 'Run Pipeline',    hint: 'Start a new backtest',    action: 'run' },
  { id: 'demo',       icon: '⚡', label: 'Load Demo Data',  hint: 'Load sample data',        action: 'demo' },
  { id: 'export',     icon: '☐', label: 'Export Results',   hint: 'Download as JSON',        action: 'export' },
  { id: 'refresh',    icon: '↻', label: 'Refresh',          hint: 'Reload current page',     action: 'refresh' },
  { id: 'health',     icon: '♥', label: 'Check Backend',    hint: 'Verify API connection',   action: 'health' },
]

const filtered = computed(() => {
  if (!query.value.trim()) return commands
  const q = query.value.toLowerCase()
  return commands.filter(c =>
    c.label.toLowerCase().includes(q) ||
    (c.hint || '').toLowerCase().includes(q) ||
    c.id.includes(q)
  )
})

watch(query, () => { selected.value = 0 })

watch(open, async (v) => {
  if (v) {
    query.value = ''
    selected.value = 0
    await nextTick()
    inputRef.value?.focus()
  }
})

function toggle() { open.value = !open.value }
function close() { open.value = false }

function moveUp() {
  selected.value = Math.max(0, selected.value - 1)
  scrollToSelected()
}

function moveDown() {
  selected.value = Math.min(filtered.value.length - 1, selected.value + 1)
  scrollToSelected()
}

function scrollToSelected() {
  nextTick(() => {
    const el = listRef.value?.children[selected.value]
    if (el) el.scrollIntoView({ block: 'nearest' })
  })
}

function execute() {
  if (filtered.value.length) run(filtered.value[selected.value])
}

function run(item) {
  close()
  if (item.tab) {
    emit('update:modelValue', item.tab)
  } else if (item.action) {
    emit('action', item.action)
  }
}

// Global keyboard shortcut
function onKeyDown(e) {
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault()
    toggle()
  }
  // Number shortcuts when palette is closed
  if (!open.value && !e.metaKey && !e.ctrlKey && !e.altKey) {
    const num = parseInt(e.key)
    if (num >= 1 && num <= 9 && document.activeElement?.tagName !== 'INPUT' && document.activeElement?.tagName !== 'SELECT') {
      const cmd = commands[num - 1]
      if (cmd?.tab) {
        emit('update:modelValue', cmd.tab)
      }
    }
  }
}

onMounted(() => document.addEventListener('keydown', onKeyDown))
onBeforeUnmount(() => document.removeEventListener('keydown', onKeyDown))

defineExpose({ toggle, open })
</script>

<style scoped>
.palette-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: rgba(6, 11, 20, 0.7);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding-top: 15vh;
}

.palette {
  width: 560px;
  max-width: 90vw;
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-xl);
  box-shadow: 0 24px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04);
  overflow: hidden;
  animation: palette-in 0.2s var(--ease-out);
}

@keyframes palette-in {
  from { opacity: 0; transform: translateY(-12px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

.palette-enter-active { transition: all 0.2s var(--ease-out); }
.palette-leave-active { transition: all 0.15s ease-in; }
.palette-enter-from, .palette-leave-to { opacity: 0; }

.palette-input-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 18px;
}

.palette-icon {
  font-size: 16px;
  color: var(--text-muted);
  flex-shrink: 0;
}

.palette-input {
  flex: 1;
  background: transparent;
  border: none;
  color: var(--text-primary);
  font-size: 15px;
  font-family: inherit;
  outline: none;
}

.palette-input::placeholder { color: var(--text-dim); }

.palette-kbd {
  font-family: var(--font-mono);
  font-size: 10px;
  padding: 2px 6px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-muted);
  flex-shrink: 0;
}

.palette-kbd-sm {
  font-family: var(--font-mono);
  font-size: 9px;
  padding: 1px 5px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 3px;
  color: var(--text-dim);
}

.palette-divider {
  height: 1px;
  background: var(--border);
}

.palette-list {
  max-height: 340px;
  overflow-y: auto;
  padding: 6px;
}

.palette-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: var(--radius);
  cursor: pointer;
  transition: background 0.1s;
}

.palette-item:hover,
.palette-item.active {
  background: var(--bg-elevated);
}

.palette-item-icon {
  font-size: 16px;
  width: 24px;
  text-align: center;
  opacity: 0.7;
  flex-shrink: 0;
}

.palette-item.active .palette-item-icon { opacity: 1; }

.palette-item-text { flex: 1; min-width: 0; }

.palette-item-label {
  font-size: 13.5px;
  font-weight: 500;
  color: var(--text-primary);
}

.palette-item-hint {
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 1px;
}

.palette-item.active .palette-item-label { color: var(--accent-bright); }

.palette-empty {
  padding: 24px;
  text-align: center;
  color: var(--text-dim);
  font-size: 13px;
}

.palette-footer {
  display: flex;
  gap: 16px;
  padding: 10px 18px;
  border-top: 1px solid var(--border-subtle);
  font-size: 11px;
  color: var(--text-dim);
}

.palette-footer kbd {
  font-family: var(--font-mono);
  font-size: 9px;
  padding: 1px 4px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 3px;
  margin-right: 3px;
}
</style>
