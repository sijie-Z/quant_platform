<template>
  <div class="oms-view">
    <!-- Header -->
    <div class="oms-header">
      <div class="oms-title">
        <span class="oms-dot"></span>
        ORDER MANAGEMENT SYSTEM
      </div>
      <div class="oms-actions">
        <button class="btn btn-sm btn-secondary" @click="showNewOrder = !showNewOrder">
          {{ showNewOrder ? 'Close' : 'New Order' }}
        </button>
        <button class="btn btn-sm btn-secondary" @click="refreshAll" :disabled="loading">
          {{ loading ? 'Loading...' : 'Refresh' }}
        </button>
      </div>
    </div>

    <!-- New Order Form -->
    <div v-if="showNewOrder" class="oms-form">
      <div class="oms-form-row">
        <div class="oms-field">
          <label>Ticker</label>
          <input v-model="newOrder.ticker" placeholder="600519" class="oms-input" />
        </div>
        <div class="oms-field">
          <label>Side</label>
          <select v-model="newOrder.side" class="oms-input">
            <option value="buy">Buy</option>
            <option value="sell">Sell</option>
          </select>
        </div>
        <div class="oms-field">
          <label>Quantity</label>
          <input v-model.number="newOrder.quantity" type="number" step="100" min="100" placeholder="1000" class="oms-input" />
        </div>
        <div class="oms-field">
          <label>Type</label>
          <select v-model="newOrder.order_type" class="oms-input">
            <option value="market">Market</option>
            <option value="limit">Limit</option>
          </select>
        </div>
        <div class="oms-field" v-if="newOrder.order_type === 'limit'">
          <label>Limit Price</label>
          <input v-model.number="newOrder.limit_price" type="number" step="0.01" class="oms-input" />
        </div>
        <div class="oms-field">
          <label>Strategy</label>
          <input v-model="newOrder.strategy" placeholder="alpha_v1" class="oms-input" />
        </div>
        <div class="oms-field oms-field-btn">
          <button class="btn btn-primary" @click="submitOrder" :disabled="submitting">
            {{ submitting ? 'Sending...' : 'Submit Order' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Summary Cards -->
    <div class="oms-summary" v-if="positions.length || tca.total_orders">
      <div class="oms-card" v-if="tca.total_orders">
        <div class="oms-card-label">Total Orders</div>
        <div class="oms-card-value">{{ tca.total_orders }}</div>
      </div>
      <div class="oms-card" v-if="tca.total_orders">
        <div class="oms-card-label">Total Volume</div>
        <div class="oms-card-value oms-accent">{{ formatNum(tca.total_volume) }}</div>
      </div>
      <div class="oms-card" v-if="tca.total_orders">
        <div class="oms-card-label">Cost (bps)</div>
        <div class="oms-card-value oms-warn">{{ tca.cost_bps?.toFixed(1) }}</div>
      </div>
      <div class="oms-card" v-if="tca.total_orders">
        <div class="oms-card-label">Commission</div>
        <div class="oms-card-value">{{ formatNum(tca.total_commission) }}</div>
      </div>
      <div class="oms-card" v-if="tca.total_orders">
        <div class="oms-card-label">Stamp Tax</div>
        <div class="oms-card-value">{{ formatNum(tca.total_tax) }}</div>
      </div>
      <div class="oms-card" v-if="tca.total_orders">
        <div class="oms-card-label">Slippage</div>
        <div class="oms-card-value">{{ formatNum(tca.total_slippage) }}</div>
      </div>
      <div class="oms-card" v-if="positions.length">
        <div class="oms-card-label">Positions</div>
        <div class="oms-card-value oms-green">{{ positions.length }}</div>
      </div>
    </div>

    <!-- Tabs -->
    <div class="oms-tabs">
      <button :class="['oms-tab', { active: tab === 'blotter' }]" @click="tab = 'blotter'">
        Order Blotter
        <span class="oms-badge" v-if="blotter.length">{{ blotter.length }}</span>
      </button>
      <button :class="['oms-tab', { active: tab === 'positions' }]" @click="tab = 'positions'">
        Positions
        <span class="oms-badge" v-if="positions.length">{{ positions.length }}</span>
      </button>
      <button :class="['oms-tab', { active: tab === 'tca' }]" @click="tab = 'tca'">
        TCA Analysis
      </button>
    </div>

    <!-- Order Blotter -->
    <div v-if="tab === 'blotter'" class="oms-table-wrap">
      <table class="oms-tbl" v-if="blotter.length">
        <thead>
          <tr>
            <th>Order ID</th>
            <th>Ticker</th>
            <th>Side</th>
            <th>Type</th>
            <th>Qty</th>
            <th>Avg Price</th>
            <th>Filled</th>
            <th>Commission</th>
            <th>Tax</th>
            <th>Slippage</th>
            <th>Status</th>
            <th>Strategy</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="o in blotter" :key="o.order_id" class="oms-row">
            <td class="oms-mono oms-dim">{{ o.order_id }}</td>
            <td class="oms-mono oms-accent">{{ o.ticker }}</td>
            <td :class="['oms-side', o.side === 'buy' ? 'oms-buy' : 'oms-sell']">
              {{ o.side?.toUpperCase() }}
            </td>
            <td class="oms-dim">{{ o.type }}</td>
            <td class="oms-mono">{{ formatNum(o.quantity) }}</td>
            <td class="oms-mono">{{ o.avg_fill_price?.toFixed(2) }}</td>
            <td class="oms-mono">{{ o.filled_quantity }}</td>
            <td class="oms-mono oms-dim">{{ o.commission?.toFixed(2) }}</td>
            <td class="oms-mono oms-dim">{{ o.tax?.toFixed(2) }}</td>
            <td class="oms-mono oms-dim">{{ o.slippage?.toFixed(2) }}</td>
            <td :class="['oms-status', `oms-status-${o.status}`]">{{ o.status }}</td>
            <td class="oms-dim">{{ o.strategy }}</td>
            <td class="oms-dim oms-time">{{ formatTime(o.filled_at || o.created_at) }}</td>
          </tr>
        </tbody>
      </table>
      <div v-else class="oms-empty">
        <div class="oms-empty-icon">&#9776;</div>
        <h3>No Orders</h3>
        <p>Create an order to start paper trading. Orders flow through PENDING → SUBMITTED → FILLED lifecycle.</p>
      </div>
    </div>

    <!-- Positions -->
    <div v-if="tab === 'positions'" class="oms-table-wrap">
      <table class="oms-tbl" v-if="positions.length">
        <thead>
          <tr>
            <th>#</th>
            <th>Ticker</th>
            <th>Quantity</th>
            <th>Avg Cost</th>
            <th>Current Price</th>
            <th>Market Value</th>
            <th>Unrealized P&L</th>
            <th>P&L %</th>
            <th>Weight</th>
            <th>Realized P&L</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(p, i) in positions" :key="p.ticker" class="oms-row">
            <td class="oms-rank">{{ i + 1 }}</td>
            <td class="oms-mono oms-accent">{{ p.ticker }}</td>
            <td class="oms-mono">{{ formatNum(p.quantity) }}</td>
            <td class="oms-mono">{{ p.avg_cost?.toFixed(2) }}</td>
            <td class="oms-mono">{{ p.current_price?.toFixed(2) }}</td>
            <td class="oms-mono oms-bold">{{ formatNum(p.market_value) }}</td>
            <td :class="['oms-mono', p.unrealized_pnl >= 0 ? 'oms-pos' : 'oms-neg']">
              {{ p.unrealized_pnl >= 0 ? '+' : '' }}{{ formatNum(p.unrealized_pnl) }}
            </td>
            <td :class="['oms-mono', p.unrealized_pnl_pct >= 0 ? 'oms-pos' : 'oms-neg']">
              {{ p.unrealized_pnl_pct >= 0 ? '+' : '' }}{{ p.unrealized_pnl_pct?.toFixed(2) }}%
            </td>
            <td class="oms-mono">{{ (p.weight * 100)?.toFixed(2) }}%</td>
            <td :class="['oms-mono', p.realized_pnl >= 0 ? 'oms-pos' : 'oms-neg']">
              {{ p.realized_pnl >= 0 ? '+' : '' }}{{ formatNum(p.realized_pnl) }}
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="oms-empty">
        <div class="oms-empty-icon">&#9734;</div>
        <h3>No Positions</h3>
        <p>Fill an order to start building positions. The OMS tracks avg cost, P&L, and weights.</p>
      </div>
    </div>

    <!-- TCA Analysis -->
    <div v-if="tab === 'tca'" class="oms-tca">
      <div v-if="tca.total_orders" class="oms-tca-grid">
        <div class="oms-tca-item">
          <div class="oms-tca-label">Buy Orders</div>
          <div class="oms-tca-value oms-pos">{{ tca.buy_orders }}</div>
        </div>
        <div class="oms-tca-item">
          <div class="oms-tca-label">Sell Orders</div>
          <div class="oms-tca-value oms-neg">{{ tca.sell_orders }}</div>
        </div>
        <div class="oms-tca-item">
          <div class="oms-tca-label">Avg Order Size</div>
          <div class="oms-tca-value">{{ formatNum(tca.avg_order_size) }}</div>
        </div>
        <div class="oms-tca-item">
          <div class="oms-tca-label">Total Cost (bps)</div>
          <div class="oms-tca-value oms-warn">{{ tca.cost_bps?.toFixed(1) }}</div>
        </div>
        <div class="oms-tca-item">
          <div class="oms-tca-label">Commission / Volume</div>
          <div class="oms-tca-value">{{ tca.total_volume ? ((tca.total_commission / tca.total_volume) * 10000).toFixed(2) : '0' }} bps</div>
        </div>
        <div class="oms-tca-item">
          <div class="oms-tca-label">Tax / Volume</div>
          <div class="oms-tca-value">{{ tca.total_volume ? ((tca.total_tax / tca.total_volume) * 10000).toFixed(2) : '0' }} bps</div>
        </div>
        <div class="oms-tca-item">
          <div class="oms-tca-label">Slippage / Volume</div>
          <div class="oms-tca-value">{{ tca.total_volume ? ((tca.total_slippage / tca.total_volume) * 10000).toFixed(2) : '0' }} bps</div>
        </div>
      </div>
      <div v-else class="oms-empty">
        <div class="oms-empty-icon">&#8776;</div>
        <h3>No Trade Data</h3>
        <p>TCA (Trade Cost Analysis) shows commission, tax, and slippage breakdown in basis points.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { createOrder, fillOrder, getBlotter, getPositions, getTCA } from '../api/index.js'

const emit = defineEmits(['toast'])

const tab = ref('blotter')
const blotter = ref([])
const positions = ref([])
const tca = ref({})
const loading = ref(false)
const submitting = ref(false)
const showNewOrder = ref(false)

const newOrder = ref({
  ticker: '',
  side: 'buy',
  quantity: 1000,
  order_type: 'market',
  limit_price: null,
  strategy: '',
})

function formatNum(v) {
  if (v == null) return '--'
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(v)
}

function formatTime(t) {
  if (!t) return '--'
  return t.replace('T', ' ').substring(0, 19)
}

async function submitOrder() {
  if (!newOrder.value.ticker) {
    emit('toast', { message: 'Ticker is required', type: 'error' })
    return
  }
  submitting.value = true
  try {
    const order = await createOrder({
      ticker: newOrder.value.ticker,
      side: newOrder.value.side,
      quantity: newOrder.value.quantity,
      order_type: newOrder.value.order_type,
      limit_price: newOrder.value.limit_price,
      strategy: newOrder.value.strategy,
    })

    // Auto-fill market orders (simulate immediate execution)
    if (newOrder.value.order_type === 'market') {
      await fillOrder({
        order_id: order.order_id,
        price: newOrder.value.limit_price || 100 + Math.random() * 50,
      })
      emit('toast', { message: `Order ${order.order_id} filled`, type: 'success' })
    } else {
      emit('toast', { message: `Order ${order.order_id} created (${order.status})`, type: 'success' })
    }

    await refreshAll()
    showNewOrder.value = false
  } catch (err) {
    emit('toast', { message: `Order failed: ${err.response?.data?.detail || err.message}`, type: 'error' })
  } finally {
    submitting.value = false
  }
}

async function refreshAll() {
  loading.value = true
  try {
    const [b, p, t] = await Promise.all([getBlotter(), getPositions(), getTCA()])
    blotter.value = b
    positions.value = p
    tca.value = t
  } catch {
    // endpoints may not have data yet
  } finally {
    loading.value = false
  }
}

onMounted(refreshAll)
</script>

<style scoped>
.oms-view {
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: 100%;
}

.oms-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}

.oms-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 8px;
  letter-spacing: 0.5px;
}

.oms-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 6px rgba(77,166,255,0.4);
}

.oms-actions {
  display: flex;
  gap: 8px;
}

.oms-form {
  background: var(--bg-secondary);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  padding: 12px;
  flex-shrink: 0;
}

.oms-form-row {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  flex-wrap: wrap;
}

.oms-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.oms-field label {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.oms-field-btn {
  justify-content: flex-end;
}

.oms-input {
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text-primary);
  font-size: 11px;
  font-family: var(--font-mono);
  padding: 5px 8px;
  min-width: 80px;
  outline: none;
}

.oms-input:focus {
  border-color: var(--accent);
}

.oms-summary {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  flex-shrink: 0;
}

.oms-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  padding: 8px 12px;
  text-align: center;
  min-width: 100px;
}

.oms-card-label {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 2px;
}

.oms-card-value {
  font-size: 14px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--text-primary);
}

.oms-accent { color: var(--accent); }
.oms-pos { color: var(--green); }
.oms-neg { color: var(--red); }
.oms-green { color: var(--green); }
.oms-warn { color: var(--orange); }

.oms-tabs {
  display: flex;
  gap: 2px;
  flex-shrink: 0;
}

.oms-tab {
  padding: 6px 14px;
  font-size: 10px;
  font-weight: 600;
  color: var(--text-dim);
  background: transparent;
  border: 1px solid transparent;
  border-bottom: none;
  border-radius: 4px 4px 0 0;
  cursor: pointer;
  transition: all 0.15s;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.oms-tab:hover { color: var(--text-secondary); }
.oms-tab.active {
  color: var(--accent);
  background: var(--bg-card);
  border-color: var(--border);
}

.oms-badge {
  font-size: 9px;
  padding: 1px 5px;
  background: var(--accent);
  color: var(--bg-base);
  border-radius: 8px;
  font-weight: 700;
}

.oms-table-wrap {
  flex: 1;
  overflow: auto;
  min-height: 0;
  border: 1px solid var(--border);
  border-radius: 6px;
}

.oms-tbl {
  width: 100%;
  border-collapse: collapse;
  font-size: 10.5px;
  font-variant-numeric: tabular-nums;
}

.oms-tbl th {
  padding: 6px 8px;
  font-size: 9px;
  font-weight: 700;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.6px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
  position: sticky;
  top: 0;
  z-index: 1;
}

.oms-tbl td {
  padding: 4px 8px;
  border-bottom: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  white-space: nowrap;
}

.oms-row:hover td { background: rgba(77,166,255,0.04); }

.oms-mono { font-family: var(--font-mono); }
.oms-dim { color: var(--text-dim); }
.oms-bold { font-weight: 600; }
.oms-rank { text-align: center; color: var(--text-dim); font-size: 9px; }
.oms-time { font-size: 9px; }

.oms-side {
  font-weight: 700;
  font-size: 10px;
  letter-spacing: 0.5px;
}

.oms-buy { color: var(--green); }
.oms-sell { color: var(--red); }

.oms-status {
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  padding: 2px 6px;
  border-radius: 3px;
}

.oms-status-filled { color: var(--green); background: rgba(52,211,153,0.1); }
.oms-status-pending { color: var(--orange); background: rgba(251,191,36,0.1); }
.oms-status-submitted { color: var(--accent); background: rgba(77,166,255,0.1); }
.oms-status-cancelled { color: var(--text-dim); background: rgba(100,100,100,0.1); }
.oms-status-rejected { color: var(--red); background: rgba(239,68,68,0.1); }

.oms-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: var(--text-muted);
}

.oms-empty-icon { font-size: 48px; opacity: 0.2; }
.oms-empty h3 { font-size: 14px; color: var(--text-secondary); }
.oms-empty p { font-size: 12px; max-width: 400px; text-align: center; line-height: 1.6; }

.oms-tca {
  flex: 1;
  min-height: 0;
}

.oms-tca-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 8px;
}

.oms-tca-item {
  background: var(--bg-secondary);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  padding: 12px 16px;
  text-align: center;
}

.oms-tca-label {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}

.oms-tca-value {
  font-size: 18px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--text-primary);
}
</style>
