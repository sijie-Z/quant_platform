<template>
  <div class="app-shell">
    <!-- Terminal Header -->
    <TerminalHeader
      :connected="backendAlive"
      v-model="currentView"
      :views="views"
      @run="onRun"
      @demo="onDemo"
      @command="togglePalette"
    />

    <!-- Main Content -->
    <div class="app-main">
      <!-- Terminal View (default) -->
      <TerminalDashboard
        v-if="currentView === 'terminal'"
        ref="terminalRef"
        @toast="onToast"
      />

      <!-- Live Trading View -->
      <div v-else-if="currentView === 'trading'" class="app-scroll">
        <LiveTrading @toast="onToast" />
      </div>

      <!-- Live Portfolio View -->
      <div v-else-if="currentView === 'live'" class="app-scroll">
        <LivePortfolio @toast="onToast" />
      </div>

      <!-- OMS View -->
      <div v-else-if="currentView === 'oms'" class="app-scroll">
        <OrderBlotter @toast="onToast" />
      </div>

      <!-- Compare View -->
      <div v-else-if="currentView === 'compare'" class="app-scroll">
        <StrategyCompare @toast="onToast" />
      </div>

      <!-- Sweep View -->
      <div v-else-if="currentView === 'sweep'" class="app-scroll">
        <ParamSweep @toast="onToast" />
      </div>

      <!-- Factors View -->
      <div v-else-if="currentView === 'factors'" class="app-scroll">
        <FactorRanking @toast="onToast" />
      </div>

      <!-- History View -->
      <div v-else-if="currentView === 'history'" class="app-scroll">
        <RunHistory @toast="onToast" />
      </div>

      <!-- Settings View -->
      <div v-else-if="currentView === 'settings'" class="app-scroll">
        <SettingsPage @toast="onToast" />
      </div>
    </div>

    <!-- Status Bar -->
    <StatusBar :connected="backendAlive" @action="onStatusBarAction" />

    <!-- Command Palette -->
    <CommandPalette v-model="currentView" @action="onPaletteAction" @toast="onToast" />

    <!-- Toast -->
    <Toast ref="toastRef" />
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import TerminalHeader from './components/TerminalHeader.vue'
import TerminalDashboard from './components/TerminalDashboard.vue'
import LivePortfolio from './components/LivePortfolio.vue'
import LiveTrading from './components/LiveTrading.vue'
import OrderBlotter from './components/OrderBlotter.vue'
import StrategyCompare from './components/StrategyCompare.vue'
import ParamSweep from './components/ParamSweep.vue'
import FactorRanking from './components/FactorRanking.vue'
import RunHistory from './components/RunHistory.vue'
import SettingsPage from './components/Settings.vue'
import StatusBar from './components/StatusBar.vue'
import Toast from './components/Toast.vue'
import CommandPalette from './components/CommandPalette.vue'
import { healthCheck } from './api/index.js'

const views = [
  { id: 'terminal', label: 'Terminal' },
  { id: 'trading', label: 'Trading' },
  { id: 'live', label: 'Live' },
  { id: 'oms', label: 'OMS' },
  { id: 'compare', label: 'Compare' },
  { id: 'sweep', label: 'Sweep' },
  { id: 'factors', label: 'Factors' },
  { id: 'history', label: 'History' },
  { id: 'settings', label: 'Settings' },
]

const currentView = ref('terminal')
const backendAlive = ref(false)
const toastRef = ref(null)
const terminalRef = ref(null)
let healthTimer = null

function onToast({ message, type }) {
  if (!toastRef.value) return
  if (type === 'success') toastRef.value.toastSuccess(message)
  else if (type === 'error') toastRef.value.toastError(message)
  else toastRef.value.toast(message)
}

function onRun() {
  if (currentView.value !== 'terminal') {
    currentView.value = 'terminal'
  }
  setTimeout(() => {
    if (terminalRef.value) terminalRef.value.showRunPanel = true
  }, 150)
}

function onDemo() {
  if (currentView.value !== 'terminal') {
    currentView.value = 'terminal'
  }
  setTimeout(() => terminalRef.value?.loadDemo?.(), 150)
}

function togglePalette() {
  onRun()
}

function onPaletteAction(action) {
  if (action === 'run') onRun()
  else if (action === 'demo') onDemo()
  else if (action === 'export') {
    if (terminalRef.value) terminalRef.value.exportResults()
  }
}

function onStatusBarAction(action) {
  if (action === 'palette') onRun()
}

async function checkHealth() {
  try {
    await healthCheck()
    backendAlive.value = true
  } catch {
    backendAlive.value = false
  }
}

onMounted(() => {
  checkHealth()
  healthTimer = setInterval(checkHealth, 15000)
})

onBeforeUnmount(() => {
  if (healthTimer) clearInterval(healthTimer)
})
</script>

<style scoped>
.app-shell {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: var(--bg-base);
}

.app-main {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.app-scroll {
  height: calc(100vh - 64px);
  overflow-y: auto;
  padding: var(--space-6) var(--space-8);
  max-width: 1440px;
  margin: 0 auto;
  width: 100%;
}
</style>
