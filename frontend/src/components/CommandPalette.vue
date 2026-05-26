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
              :placeholder="locale === 'zh-CN' ? '输入命令或搜索...' : 'Type a command or search...'"
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
            <div v-if="!filtered.length" class="palette-empty">{{ locale === 'zh-CN' ? '无匹配结果' : 'No results' }} "{{ query }}"</div>
          </div>
          <div class="palette-footer">
            <span><kbd class="palette-kbd-sm">&#8593;&#8595;</kbd> {{ locale === 'zh-CN' ? '导航' : 'navigate' }}</span>
            <span><kbd class="palette-kbd-sm">&#9166;</kbd> {{ locale === 'zh-CN' ? '选择' : 'select' }}</span>
            <span><kbd class="palette-kbd-sm">esc</kbd> {{ locale === 'zh-CN' ? '关闭' : 'close' }}</span>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted, onBeforeUnmount } from 'vue'
import { useI18n } from '../i18n/index.js'

const { $t, locale } = useI18n()

const props = defineProps({
  modelValue: { type: String, default: 'dashboard' },
})

const emit = defineEmits(['update:modelValue', 'action', 'toast'])

const open = ref(false)
const query = ref('')
const selected = ref(0)
const inputRef = ref(null)
const listRef = ref(null)

const commands = computed(() => [
  { id: 'terminal',   icon: '▸', label: $t('nav.terminal'),        hint: locale.value === 'zh-CN' ? '主回测仪表盘' : 'Main backtest dashboard', shortcut: '1', tab: 'terminal' },
  { id: 'trading',    icon: '⚡', label: $t('nav.trading'),    hint: locale.value === 'zh-CN' ? '模拟/实盘交易引擎' : 'Paper/Live trading engine', shortcut: '2', tab: 'trading' },
  { id: 'live',       icon: '◉', label: $t('nav.live'),  hint: locale.value === 'zh-CN' ? '实时组合追踪' : 'Real-time portfolio tracker', shortcut: '3', tab: 'live' },
  { id: 'oms',        icon: '⊞', label: $t('nav.oms'),   hint: locale.value === 'zh-CN' ? '订单管理' : 'OMS order management',    shortcut: '4', tab: 'oms' },
  { id: 'compare',    icon: '⇄', label: $t('nav.compare'),         hint: locale.value === 'zh-CN' ? '策略对比' : 'Strategy comparison',     shortcut: '5', tab: 'compare' },
  { id: 'sweep',      icon: '⌖', label: $t('nav.sweep'),           hint: locale.value === 'zh-CN' ? '参数网格搜索' : 'Parameter grid search',   shortcut: '6', tab: 'sweep' },
  { id: 'factors',    icon: '≡', label: $t('nav.factors'),         hint: locale.value === 'zh-CN' ? '因子排名' : 'Factor IC rankings',      shortcut: '7', tab: 'factors' },
  { id: 'history',    icon: '⏱', label: $t('nav.history'),         hint: locale.value === 'zh-CN' ? '运行历史' : 'Run history',             shortcut: '8', tab: 'history' },
  { id: 'settings',   icon: '⚙', label: $t('nav.settings'),        hint: locale.value === 'zh-CN' ? '平台设置' : 'Platform configuration',  shortcut: '9', tab: 'settings' },
  { id: 'run',        icon: '▶', label: $t('terminal.runPipeline'),    hint: locale.value === 'zh-CN' ? '开始新回测' : 'Start a new backtest',    action: 'run' },
  { id: 'demo',       icon: '⚡', label: $t('terminal.demoBtn'),  hint: locale.value === 'zh-CN' ? '加载示例数据' : 'Load sample data',        action: 'demo' },
  { id: 'export',     icon: '☐', label: $t('terminal.exportBtn'),   hint: locale.value === 'zh-CN' ? '导出为 JSON' : 'Download as JSON',        action: 'export' },
  { id: 'refresh',    icon: '↻', label: locale.value === 'zh-CN' ? '刷新' : 'Refresh',          hint: locale.value === 'zh-CN' ? '刷新当前页' : 'Reload current page',     action: 'refresh' },
  { id: 'health',     icon: '♥', label: locale.value === 'zh-CN' ? '检查后端' : 'Check Backend',    hint: locale.value === 'zh-CN' ? '验证API连接' : 'Verify API connection',   action: 'health' },
])

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
