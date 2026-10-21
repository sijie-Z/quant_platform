import axios from 'axios'

// ── Backend base URL ─────────────────────────────────────────────────────────
//
// Settings persists the operator's backend URL under `quant_settings.apiBase`.
// Historically nothing read it: axios was pinned to `/api` and the socket to
// `location.host`, so pointing the UI at another backend silently kept talking
// to the same origin. The base URL is now read at request time (so a change
// applies immediately, without a rebuild) and again on every WebSocket
// connect, and the saved value survives reloads via localStorage.

const DEFAULT_API_BASE = '/api'
const SETTINGS_KEY = 'quant_settings'

function readSavedSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch (_) {
    return {}
  }
}

// Accepts `http://host:8000`, `http://host:8000/api`, `/api`, `/proxy/api/`
// and returns a base URL whose path ends in `/api` and has no trailing slash.
// Anything unsafe or malformed (spaces, query strings, fragments, non-http(s)
// schemes, missing host, `http:/api`, `https:///api`) falls back to `/api`.
export function normalizeApiBase(value) {
  let base = String(value == null ? '' : value).trim()
  if (!base) return DEFAULT_API_BASE
  if (!isValidApiBase(base)) return DEFAULT_API_BASE
  base = base.replace(/\/+$/, '')
  if (/\/api$/i.test(base)) base = base.replace(/\/api$/i, '/api')
  else base = `${base}/api`
  if (/^https?:\/\//i.test(base)) {
    const url = new URL(base)
    return `${url.protocol}//${url.host}${url.pathname.replace(/\/+$/, '')}`
  }
  return base
}

export function isValidApiBase(value) {
  const raw = String(value == null ? '' : value).trim()
  if (!raw) return true
  if (/[\s?#]/.test(raw)) return false
  if (/^[a-z][a-z0-9+.-]*:/i.test(raw)) {
    // Require an explicit http(s) authority with a non-empty host. This
    // rejects `http:/api` (single slash) and `https:///api` (empty authority),
    // which WHATWG URL otherwise parses as host `api`.
    if (!/^https?:\/\//i.test(raw)) return false
    if (/^https?:\/\/[/\\]/.test(raw)) return false
    try {
      const url = new URL(raw)
      return !!url.hostname && (url.protocol === 'http:' || url.protocol === 'https:')
    } catch (_) {
      return false
    }
  }
  // Relative paths must start with `/`; a bare `api` would otherwise silently
  // become `/api/api`. Protocol-relative `//host` is rejected: it is neither a
  // path the server can route nor an absolute URL the socket factory handles.
  return raw.startsWith('/') && !raw.startsWith('//')
}

export function getApiBase() {
  const settings = readSavedSettings()
  return normalizeApiBase(settings.apiBase)
}

// Derive the status-stream URL from the same base. An absolute backend URL
// keeps its host (so a UI served from another origin can reach the API), and a
// relative base stays on the page's origin.
export function getWsUrl() {
  const base = getApiBase()
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(base)) {
    try {
      const url = new URL(base)
      const proto = url.protocol === 'https:' ? 'wss:' : 'ws:'
      const path = url.pathname.replace(/\/+$/, '')
      return `${proto}//${url.host}${path}/ws`
    } catch (_) {
      // fall through to the relative form
    }
  }
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${location.host}${base.replace(/\/+$/, '')}/ws`
}

const api = axios.create({
  baseURL: getApiBase(),
  timeout: 300000, // 5 min for long pipeline runs
})

// Optional bearer token, mirroring the backend's QUANT_API_TOKEN mode.
//
// Read at request time from localStorage, NOT from import.meta.env: a
// VITE_-prefixed value is inlined into the public bundle, so every visitor
// to the served UI would receive the long-lived token. Users set it in
// Settings (stored only in this browser).
export function getApiToken() {
  try {
    return localStorage.getItem('quant_api_token') || ''
  } catch (_) {
    return ''
  }
}

api.interceptors.request.use((config) => {
  // Re-read on every request so a URL saved in Settings takes effect
  // immediately (and after a reload, when the instance is rebuilt).
  config.baseURL = getApiBase()
  // Trim consistently with the WebSocket path: a token pasted with trailing
  // whitespace must not authenticate on the socket but fail on REST.
  const token = getApiToken().trim()
  if (token) {
    config.headers = config.headers || {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Turn the backend's two error shapes into one readable string. FastAPI
// `HTTPException` sends `detail` as a string, but `RequestValidationError`
// sends a list of `{loc, msg}` objects; components that interpolated `detail`
// printed `[object Object]`.
export function getApiErrorMessage(error, fallback = 'Request failed') {
  const data = error?.response?.data
  let detail = data?.detail
  if (Array.isArray(detail)) {
    detail = detail
      .map((item) => {
        if (item == null) return ''
        if (typeof item === 'string') return item
        const loc = Array.isArray(item.loc)
          ? item.loc.filter((part) => part !== 'body').join('.')
          : ''
        const msg = item.msg || item.message || JSON.stringify(item)
        return loc ? `${loc}: ${msg}` : msg
      })
      .filter(Boolean)
      .join('; ')
  }
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  if (typeof data?.message === 'string' && data.message.trim()) return data.message.trim()
  if (typeof data?.error === 'string' && data.error.trim()) return data.error.trim()
  return error?.message || fallback
}

export function importPortfolioHoldings(data) {
  return api.post('/portfolio/import', { data }).then(r => r.data)
}

export function getLivePortfolio() {
  return api.get('/portfolio/live').then(r => r.data)
}

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
  return api.get('/health', { timeout: 10000 }).then(r => r.data)
}

// Dead-man switch. Returns the 503 payload instead of throwing so callers can
// render the per-check reasons; transport failures still reject.
export function getReadiness() {
  return api
    .get('/readiness', { timeout: 10000, validateStatus: s => s === 200 || s === 503 })
    .then(r => r.data)
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
  // User-entered tickers go into the path; `#`, `?`, `/` or spaces would
  // corrupt the request instead of yielding a clean 404.
  return api.get(`/market/baostock/stock/${encodeURIComponent(code)}`).then(r => r.data)
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

// ── WebSocket for real-time pipeline status ─────────────────────────────────
//
// Returns a handle whose close() permanently stops reconnection.
//
// Token transport: browsers cannot set handshake headers, so the token travels
// as the `bearer.<token>` WebSocket subprotocol (never in the URL, which
// uvicorn's access log records). A WebSocket subprotocol must be an HTTP token
// (RFC 7230): base64 tokens containing `=` or `/` make `new WebSocket(...)`
// throw. In that case only, fall back to the backend's legacy `?token=` form,
// percent-encoded. The token is re-read on every connect, so changing it in
// Settings applies to the next reconnect.
//
// Lifecycle: exponential backoff with jitter, capped at 30 s, reset on a
// successful open. Policy-violation closes (1008, 4401) mean the token was
// rejected — retrying cannot help, so reconnection stops and the caller is
// told via onError({ type: 'auth' }).

const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 30000
const AUTH_CLOSE_CODES = new Set([1008, 4401])
// RFC 7230 token characters. `bearer.<token>` must match this to be a legal
// subprotocol.
const WS_TOKEN_RE = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/

function buildSocketTarget() {
  const url = getWsUrl()
  const token = getApiToken().trim()
  if (!token) return { url, protocols: [] }
  if (WS_TOKEN_RE.test(token)) {
    return { url, protocols: [`bearer.${token}`] }
  }
  return { url: `${url}?token=${encodeURIComponent(token)}`, protocols: [] }
}

export function createStatusSocket(onMessage, onError) {
  let socket = null
  let timer = null
  let disposed = false
  let attempt = 0

  const notify = (info) => {
    try {
      if (onError) onError(info)
    } catch (_) {}
  }

  const scheduleReconnect = () => {
    if (disposed) return
    const exp = Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * Math.pow(2, attempt))
    attempt = Math.min(attempt + 1, 10)
    const delay = Math.max(500, Math.round(exp * (0.7 + Math.random() * 0.6)))
    if (timer) clearTimeout(timer)
    timer = setTimeout(connect, delay)
  }

  function connect() {
    if (disposed) return
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
    const target = buildSocketTarget()
    let ws
    try {
      ws = target.protocols.length
        ? new WebSocket(target.url, target.protocols)
        : new WebSocket(target.url)
    } catch (_) {
      // Constructing the socket failed (e.g. an illegal subprotocol slip
      // through); never surface the URL, which may carry the legacy token.
      notify({ type: 'construct', message: 'WebSocket could not be created' })
      scheduleReconnect()
      return
    }
    socket = ws

    ws.onopen = () => {
      attempt = 0
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (onMessage) onMessage(data)
      } catch (_) {}
    }

    ws.onerror = () => {
      notify({ type: 'error', message: 'WebSocket connection error' })
    }

    ws.onclose = (event) => {
      if (disposed) return
      socket = null
      if (AUTH_CLOSE_CODES.has(event.code)) {
        disposed = true
        notify({
          type: 'auth',
          code: event.code,
          message: 'WebSocket authentication failed; reconnection stopped',
        })
        return
      }
      scheduleReconnect()
    }
  }

  connect()

  return {
    get socket() { return socket },
    // True once close() ran or an auth close stopped reconnection. Callers use
    // this to recreate the stream instead of trusting a stale handle.
    get disposed() { return disposed },
    close() {
      disposed = true
      if (timer) {
        clearTimeout(timer)
        timer = null
      }
      try { if (socket) socket.close() } catch (_) {}
    },
  }
}
