<template>
  <div :class="['panel', { 'panel-full': full, 'panel-maximized': maximized }]">
    <div class="panel-header">
      <div class="panel-title">
        <span class="panel-dot" :style="dotStyle"></span>
        <slot name="title">{{ title }}</slot>
      </div>
      <div class="panel-actions">
        <slot name="actions"></slot>
        <button
          class="panel-max-btn"
          @click="maximized = !maximized"
          :title="maximized ? 'Restore' : 'Maximize'"
        >{{ maximized ? '&#9633;' : '&#9634;' }}</button>
      </div>
    </div>
    <div class="panel-body">
      <slot></slot>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  title: { type: String, default: '' },
  dotColor: { type: String, default: '' },
  full: { type: Boolean, default: false },
})

const maximized = ref(false)

const dotStyle = computed(() => {
  if (!props.dotColor) return {}
  return {
    background: props.dotColor,
    boxShadow: `0 0 6px ${props.dotColor}80`,
  }
})
</script>

<style scoped>
.panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-width: 0;
  min-height: 0;
  transition: all 0.2s ease;
}

.panel-full {
  grid-column: 1 / -1;
}

.panel-maximized {
  position: fixed !important;
  inset: 44px 0 24px 0 !important;
  z-index: 300 !important;
  border-radius: 0 !important;
  border: none !important;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 10px;
  border-bottom: 1px solid var(--border-subtle);
  background: rgba(255,255,255,0.015);
  flex-shrink: 0;
  min-height: 30px;
}

.panel-title {
  font-size: 10px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  display: flex;
  align-items: center;
  gap: 7px;
  white-space: nowrap;
}

.panel-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--accent);
  flex-shrink: 0;
}

.panel-actions {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
}

.panel-max-btn {
  background: none;
  border: 1px solid transparent;
  color: var(--text-dim);
  font-size: 12px;
  cursor: pointer;
  padding: 1px 4px;
  border-radius: 3px;
  line-height: 1;
  transition: all 0.15s ease;
}

.panel-max-btn:hover {
  color: var(--text-secondary);
  border-color: var(--border);
  background: rgba(255,255,255,0.03);
}

.panel-body {
  flex: 1;
  overflow: auto;
  padding: 8px 10px;
  min-height: 0;
}
</style>
