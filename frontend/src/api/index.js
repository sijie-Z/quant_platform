import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 5 min for long pipeline runs
})

export function getConfig() {
  return api.get('/config').then(r => r.data)
}

export function runPipeline(params) {
  return api.post('/run', params).then(r => r.data)
}

export function getRunStatus(runId) {
  return api.get(`/run/${runId}/status`).then(r => r.data)
}

export function getRunResult(runId) {
  return api.get(`/run/${runId}/result`).then(r => r.data)
}

export function compareStrategies(params) {
  return api.post('/compare', params).then(r => r.data)
}

export function sweepParameters(params) {
  return api.post('/sweep', params).then(r => r.data)
}

export function getFactors() {
  return api.get('/factors').then(r => r.data)
}

export function getDemo() {
  return api.get('/demo').then(r => r.data)
}

export function getRuns() {
  return api.get('/runs').then(r => r.data)
}

export function healthCheck() {
  return api.get('/health').then(r => r.data)
}

// OMS endpoints
export function createOrder(params) {
  return api.post('/oms/order', params).then(r => r.data)
}

export function fillOrder(params) {
  return api.post('/oms/fill', params).then(r => r.data)
}

export function getBlotter() {
  return api.get('/oms/blotter').then(r => r.data)
}

export function getPositions() {
  return api.get('/oms/positions').then(r => r.data)
}

export function getTCA() {
  return api.get('/oms/tca').then(r => r.data)
}

// Portfolio live endpoints
export function importPortfolio(data) {
  return api.post('/portfolio/import', { data }).then(r => r.data)
}

export function getPortfolioLive() {
  return api.get('/portfolio/live').then(r => r.data)
}

export function getPortfolioHoldings() {
  return api.get('/portfolio/holdings').then(r => r.data)
}

// Advanced analytics
export function runWalkForward(params) {
  return api.post('/walkforward', params).then(r => r.data)
}

export function runMonteCarlo(params) {
  return api.post('/montecarlo', params).then(r => r.data)
}

export function decomposeRisk(params) {
  return api.post('/risk/decompose', params).then(r => r.data)
}

export function getICDecay() {
  return api.get('/analysis/ic-decay').then(r => r.data)
}

export function getFactorCorrelation() {
  return api.get('/analysis/correlation').then(r => r.data)
}

// Regime detection
export function detectRegime(params) {
  return api.post('/regime/detect', params).then(r => r.data)
}

// Risk monitor
export function getRiskStatus() {
  return api.get('/risk/status').then(r => r.data)
}

export function toggleKillSwitch(params) {
  return api.post('/risk/kill-switch', params).then(r => r.data)
}

export function checkOrderRisk(order) {
  return api.post('/risk/check-order', order).then(r => r.data)
}

// Execution algorithms
export function smartRouteOrder(params) {
  return api.post('/execution/smart-route', params).then(r => r.data)
}

// HTML Report
export function generateReport(runId) {
  return api.post('/report/html', { run_id: runId }).then(r => r.data)
}

// Multi-Strategy
export function addStrategy(params) {
  return api.post('/strategy/add', params).then(r => r.data)
}

export function removeStrategy(strategyId) {
  return api.post('/strategy/remove', { strategy_id: strategyId }).then(r => r.data)
}

export function listStrategies() {
  return api.get('/strategy/list').then(r => r.data)
}

export function allocateCapital(weights) {
  return api.post('/strategy/allocate', { weights }).then(r => r.data)
}

export function getStrategyMetrics() {
  return api.get('/strategy/metrics').then(r => r.data)
}

export function getStrategyAlerts() {
  return api.get('/strategy/alerts').then(r => r.data)
}

export function updateStrategyPnl(strategyId, dailyReturn) {
  return api.post('/strategy/update-pnl', { strategy_id: strategyId, daily_return: dailyReturn }).then(r => r.data)
}

// Data Quality
export function runDataQuality(runId, params = {}) {
  return api.post('/data/quality', { run_id: runId, ...params }).then(r => r.data)
}

// Live Trading Engine
export function startTrading(params) {
  return api.post('/trading/start', params).then(r => r.data)
}

export function stopTrading() {
  return api.post('/trading/stop').then(r => r.data)
}

export function getTradingStatus() {
  return api.get('/trading/status').then(r => r.data)
}

export function getTradingPositions() {
  return api.get('/trading/positions').then(r => r.data)
}

export function getTradingAccount() {
  return api.get('/trading/account').then(r => r.data)
}

export function getTradingCycles() {
  return api.get('/trading/cycles').then(r => r.data)
}

export function placeManualOrder(order) {
  return api.post('/trading/order', order).then(r => r.data)
}

export function runTradingCycle() {
  return api.post('/trading/run-once').then(r => r.data)
}

// Real-time Market Data
export function getMarketSnapshot() {
  return api.get('/market/snapshot').then(r => r.data)
}

export function getMarketGainers() {
  return api.get('/market/gainers').then(r => r.data)
}

export function getMarketLosers() {
  return api.get('/market/losers').then(r => r.data)
}

export function getMarketSectors() {
  return api.get('/market/sectors').then(r => r.data)
}

// Market data
export function getBaostockHealth() {
  return api.get('/market/baostock/health').then(r => r.data)
}

export function getBaostockStock(code) {
  return api.get(`/market/baostock/stock/${code}`).then(r => r.data)
}

// Core Architecture Monitoring
export function getCoreEvents(topic = '', limit = 50) {
  return api.get('/core/events', { params: { topic, limit } }).then(r => r.data)
}

export function getCoreEventMetrics() {
  return api.get('/core/events/metrics').then(r => r.data)
}

export function getCoreStoreStats() {
  return api.get('/core/store/stats').then(r => r.data)
}

export function getCoreStoreOrders(status = '', code = '', limit = 100) {
  return api.get('/core/store/orders', { params: { status, code, limit } }).then(r => r.data)
}

export function getCoreStoreTrades(code = '', limit = 100) {
  return api.get('/core/store/trades', { params: { code, limit } }).then(r => r.data)
}

export function getCoreStorePnl(days = 30) {
  return api.get('/core/store/pnl', { params: { days } }).then(r => r.data)
}

export function getCoreStoreSignals(consumed = -1, limit = 100) {
  return api.get('/core/store/signals', { params: { consumed, limit } }).then(r => r.data)
}

export function getCoreStoreSessions(limit = 20) {
  return api.get('/core/store/sessions', { params: { limit } }).then(r => r.data)
}

export function getCoreState() {
  return api.get('/core/state').then(r => r.data)
}

export function getCoreAudit(action = '', limit = 50) {
  return api.get('/core/audit', { params: { action, limit } }).then(r => r.data)
}

export function getCoreRisk() {
  return api.get('/core/risk').then(r => r.data)
}

export function toggleCoreKillSwitch(activate = true, reason = '') {
  return api.post('/core/risk/kill-switch', { activate, reason }).then(r => r.data)
}

// Monitor Dashboard endpoints
export function getMonitorRiskOverview() {
  return api.get('/monitor/risk-overview').then(r => r.data)
}

export function getMonitorTCASummary() {
  return api.get('/monitor/tca-summary').then(r => r.data)
}

export function getMonitorFactorStatus() {
  return api.get('/monitor/factor-status').then(r => r.data)
}

export function getMonitorCapacityGauge() {
  return api.get('/monitor/capacity-gauge').then(r => r.data)
}

export function updateMonitorConfig(params) {
  return api.post('/monitor/config', params).then(r => r.data)
}

export function triggerMonitorKillSwitch(params) {
  return api.post('/monitor/kill-switch', params).then(r => r.data)
}

// WebSocket for real-time pipeline status
export function createStatusSocket(onMessage, onError) {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const ws = new WebSocket(`${protocol}//${location.host}/api/ws`)

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (onMessage) onMessage(data)
    } catch (_) {}
  }

  ws.onerror = (err) => {
    if (onError) onError(err)
  }

  ws.onclose = () => {
    // Auto-reconnect after 3s
    setTimeout(() => {
      try {
        const ws2 = createStatusSocket(onMessage, onError)
        return ws2
      } catch (_) {}
    }, 3000)
  }

  return ws
}
