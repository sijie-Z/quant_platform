<template>
  <div class="toast-container" aria-live="assertive" aria-relevant="additions removals">
    <TransitionGroup name="toast-list">
      <div
        v-for="toast in toasts"
        :key="toast.id"
        :class="['toast', `toast-${toast.type}`, { leaving: toast.leaving }]"
        role="alert"
      >
        <span class="toast-icon" aria-hidden="true">{{ icons[toast.type] || icons.info }}</span>
        <span class="toast-message">{{ toast.message }}</span>
      </div>
    </TransitionGroup>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const icons = { success: '✓', error: '✕', info: 'ℹ', warning: '⚠' }
const toasts = ref([])
let nextId = 0

function showToast(message, type = 'info', duration = 3500) {
  const id = nextId++
  toasts.value.push({ id, message, type, leaving: false })
  setTimeout(() => {
    const t = toasts.value.find(x => x.id === id)
    if (t) t.leaving = true
  }, duration - 300)
  setTimeout(() => {
    toasts.value = toasts.value.filter(x => x.id !== id)
  }, duration)
}

function toast(message) { showToast(message, 'info') }
function toastSuccess(message) { showToast(message, 'success') }
function toastError(message) { showToast(message, 'error', 5000) }

defineExpose({ toast, toastSuccess, toastError })
</script>

<style scoped>
.toast-icon {
  font-size: 14px;
  font-weight: 700;
  opacity: 0.9;
  flex-shrink: 0;
}

.toast-message {
  line-height: 1.4;
}

.toast-list-enter-active {
  transition: all 0.35s var(--ease-out);
}

.toast-list-leave-active {
  transition: all 0.25s var(--ease-out);
}

.toast-list-enter-from {
  opacity: 0;
  transform: translateX(40px) scale(0.96);
}

.toast-list-leave-to {
  opacity: 0;
  transform: translateX(40px) scale(0.96);
}

.toast-list-move {
  transition: transform 0.3s var(--ease-out);
}
</style>
