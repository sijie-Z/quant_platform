<template>
  <div class="oms-view">
    <!-- Header -->
    <div class="oms-header">
      <div class="oms-title">
        <span class="oms-dot"></span>
        {{ locale === 'zh-CN' ? '订单管理系统' : 'ORDER MANAGEMENT SYSTEM' }}
      </div>
      <div class="oms-actions">
        <button class="btn btn-sm btn-secondary" @click="showNewOrder = !showNewOrder">
          {{ showNewOrder ? (locale === 'zh-CN' ? '关闭' : 'Close') : (locale === 'zh-CN' ? '新建订单' : 'New Order') }}
        </button>
        <button class="btn btn-sm btn-secondary" @click="refreshAll" :disabled="loading">
          {{ loading ? (locale === 'zh-CN' ? '加载中...' : 'Loading...') : (locale === 'zh-CN' ? '刷新' : 'Refresh') }}
        </button>
      </div>
    </div>

    <!-- New Order Form -->
    <div v-if="showNewOrder" class="oms-form">
      <div class="oms-form-row">
        <div class="oms-field">
          <label>{{ locale === 'zh-CN' ? '股票代码' : 'Ticker' }}</label>
          <input v-model="newOrder.ticker" placeholder="600519" class="oms-input" />
        </div>
        <div class="oms-field">
          <label>{{ locale === 'zh-CN' ? '方向' : 'Side' }}</label>
          <select v-model="newOrder.side" class="oms-input">
            <option value="buy">{{ locale === 'zh-CN' ? '买入' : 'Buy' }}</option>
            <option value="sell">{{ locale === 'zh-CN' ? '卖出' : 'Sell' }}</option>
          </select>
        </div>
        <div class="oms-field">
          <label>{{ locale === 'zh-CN' ? '数量' : 'Quantity' }}</label>
          <input v-model.number="newOrder.quantity" type="number" step="100" min="100" placeholder="1000" class="oms-input" />
        </div>
        <div class="oms-field">
          <label>{{ locale === 'zh-CN' ? '类型' : 'Type' }}</label>
          <select v-model="newOrder.order_type" class="oms-input">
            <option value="market">{{ locale === 'zh-CN' ? '市价单' : 'Market' }}</option>
            <option value="limit">{{ locale === 'zh-CN' ? '限价单' : 'Limit' }}</option>
          </select>
        </div>
        <div class="oms-field" v-if="newOrder.order_type === 'limit'">
          <label>{{ locale === 'zh-CN' ? '限价' : 'Limit Price' }}</label>
          <input v-model.number="newOrder.limit_price" type="number" step="0.01" class="oms-input" />
        </div>
        <div class="oms-field">
          <label>{{ locale === 'zh-CN' ? '策略' : 'Strategy' }}</label>
          <input v-model="newOrder.strategy" placeholder="alpha_v1" class="oms-input" />
        </div>
        <div class="oms-field oms-field-btn">
          <button class="btn btn-primary" @click="submitOrder" :disabled="submitting">
            {{ submitting ? (locale === 'zh-CN' ? '发送中...' : 'Sending...') : (locale === 'zh-CN' ? '提交订单' : 'Submit Order') }}
          </button>
        </div>
      </div>
    </div>

    <!-- Summary Cards -->
    <div class="oms-summary" v-if="positions.length || tca.total_orders">
      <div class="oms-card" v-if="tca.total_orders">
        <div class="oms-card-label">{{ locale === 'zh-CN' ? '总订单数' : 'Total Orders' }}</div>
        <div class="oms-card-value">{{ tca.total_orders }}</div>
      </div>
      <div class="oms-card" v-if="tca.total_orders">
        <div class="oms-card-label">{{ locale === 'zh-CN' ? '总成交量' : 'Total Volume' }}</div>
        <div class="oms-card-value oms-accent">{{ formatNum(tca.total_volume) }}</div>
      </div>
      <div class="oms-card" v-if="tca.total_orders">
        <div class="oms-card-label">{{ locale === 'zh-CN' ? '成本(bps)' : 'Cost (bps)' }}</div>
        <div class="oms-card-value oms-warn">{{ tca.cost_bps?.toFixed(1) }}</div>
      </div>
      <div class="oms-card" v-if="tca.total_orders">
        <div class="oms-card-label">{{ locale === 'zh-CN' ? '佣金' : 'Commission' }}</div>
        <div class="oms-card-value">{{ formatNum(tca.total_commission) }}</div>
      </div>
      <div class="oms-card" v-if="tca.total_orders">
        <div class="oms-card-label">{{ locale === 'zh-CN' ? '印花税' : 'Stamp Tax' }}</div>
        <div class="oms-card-value">{{ formatNum(tca.total_tax) }}</div>
      </div>
      <div class="oms-card" v-if="tca.total_orders">
        <div class="oms-card-label">{{ locale === 'zh-CN' ? '滑点' : 'Slippage' }}</div>
        <div class="oms-card-value">{{ formatNum(tca.total_slippage) }}</div>
      </div>
      <div class="oms-card" v-if="positions.length">
        <div class="oms-card-label">{{ locale === 'zh-CN' ? '持仓数' : 'Positions' }}</div>
        <div class="oms-card-value oms-green">{{ positions.length }}</div>
      </div>
    </div>

    <!-- Tabs -->
    <div class="oms-tabs">
      <button :class="['oms-tab', { active: tab === 'blotter' }]" @click="tab = 'blotter'">
        {{ locale === 'zh-CN' ? '订单簿' : 'Order Blotter' }}
        <span class="oms-badge" v-if="blotter.length">{{ blotter.length }}</span>
      </button>
      <button :class="['oms-tab', { active: tab === 'positions' }]" @click="tab = 'positions'">
        {{ locale === 'zh-CN' ? '持仓' : 'Positions' }}
        <span class="oms-badge" v-if="positions.length">{{ positions.length }}</span>
      </button>
      <button :class="['oms-tab', { active: tab === 'tca' }]" @click="tab = 'tca'">
        {{ locale === 'zh-CN' ? '交易成本分析' : 'TCA Analysis' }}
      </button>
    </div>

    <!-- Order Blotter -->
    <div v-if="tab === 'blotter'" class="oms-table-wrap">
      <table class="oms-tbl" v-if="blotter.length">
        <thead>
          <tr>
            <th>{{ locale === 'zh-CN' ? '订单ID' : 'Order ID' }}</th>
            <th>{{ locale === 'zh-CN' ? '代码' : 'Ticker' }}</th>
            <th>{{ locale === 'zh-CN' ? '方向' : 'Side' }}</th>
            <th>{{ locale === 'zh-CN' ? '类型' : 'Type' }}</th>
            <th>{{ locale === 'zh-CN' ? '数量' : 'Qty' }}</th>
            <th>{{ locale === 'zh-CN' ? '均价' : 'Avg Price' }}</th>
            <th>{{ locale === 'zh-CN' ? '已成交' : 'Filled' }}</th>
            <th>{{ locale === 'zh-CN' ? '佣金' : 'Commission' }}</th>
            <th>{{ locale === 'zh-CN' ? '印花税' : 'Tax' }}</th>
            <th>{{ locale === 'zh-CN' ? '滑点' : 'Slippage' }}</th>
            <th>{{ locale === 'zh-CN' ? '状态' : 'Status' }}</th>
            <th>{{ locale === 'zh-CN' ? '策略' : 'Strategy' }}</th>
            <th>{{ locale === 'zh-CN' ? '时间' : 'Time' }}</th>
            <th>{{ locale === 'zh-CN' ? '操作' : 'Action' }}</th>
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
            <td class="oms-action-cell">
              <div v-if="canFill(o)" class="oms-fill-inline">
                <input
                  v-model="fillPriceDrafts[o.order_id]"
                  type="number"
                  step="0.01"
                  min="0"
                  class="oms-input oms-fill-input"
                  :placeholder="fillPricePlaceholder(o)"
                  :aria-label="locale === 'zh-CN' ? '成交价格' : 'Fill price'"
                />
                <button
                  class="oms-fill-btn"
                  @click="fillRow(o)"
                  :disabled="fillingId === o.order_id"
                >{{ fillingId === o.order_id ? '...' : (locale === 'zh-CN' ? '成交' : 'Fill') }}</button>
              </div>
              <span v-else class="oms-dim">—</span>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else-if="loading" class="oms-empty">
        <div class="status-spinner" style="width:20px;height:20px;"></div>
        <p>{{ locale === 'zh-CN' ? '加载订单簿...' : 'Loading order blotter...' }}</p>
      </div>
      <div v-else class="oms-empty">
        <div class="oms-empty-icon">&#9776;</div>
        <h3>{{ locale === 'zh-CN' ? '暂无订单' : 'No Orders' }}</h3>
        <p>{{ locale === 'zh-CN' ? '创建订单以开始模拟交易。订单生命周期为 PENDING → SUBMITTED → FILLED。' : 'Create an order to start paper trading. Orders flow through PENDING → SUBMITTED → FILLED lifecycle.' }}</p>
      </div>
    </div>

    <!-- Positions -->
    <div v-if="tab === 'positions'" class="oms-table-wrap">
      <table class="oms-tbl" v-if="positions.length">
        <thead>
          <tr>
            <th>#</th>
            <th>{{ locale === 'zh-CN' ? '代码' : 'Ticker' }}</th>
            <th>{{ locale === 'zh-CN' ? '持仓数量' : 'Quantity' }}</th>
            <th>{{ locale === 'zh-CN' ? '均价' : 'Avg Cost' }}</th>
            <th>{{ locale === 'zh-CN' ? '现价' : 'Current Price' }}</th>
            <th>{{ locale === 'zh-CN' ? '市值' : 'Market Value' }}</th>
            <th>{{ locale === 'zh-CN' ? '浮动盈亏' : 'Unrealized P&L' }}</th>
            <th>{{ locale === 'zh-CN' ? '盈亏%' : 'P&L %' }}</th>
            <th>{{ locale === 'zh-CN' ? '权重' : 'Weight' }}</th>
            <th>{{ locale === 'zh-CN' ? '价格基准' : 'Basis' }}</th>
            <th>{{ locale === 'zh-CN' ? '已实现盈亏' : 'Realized P&L' }}</th>
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
            <td class="oms-basis">{{ (p.price_basis || positionsPriceBasis || '—').toString().toUpperCase() }}</td>
            <td :class="['oms-mono', p.realized_pnl >= 0 ? 'oms-pos' : 'oms-neg']">
              {{ p.realized_pnl >= 0 ? '+' : '' }}{{ formatNum(p.realized_pnl) }}
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else-if="loading" class="oms-empty">
        <div class="status-spinner" style="width:20px;height:20px;"></div>
        <p>{{ locale === 'zh-CN' ? '加载持仓...' : 'Loading positions...' }}</p>
      </div>
      <div v-else class="oms-empty">
        <div class="oms-empty-icon">&#9734;</div>
        <h3>{{ locale === 'zh-CN' ? '暂无持仓' : 'No Positions' }}</h3>
        <p>{{ locale === 'zh-CN' ? '成交订单以建立持仓。OMS追踪均价、盈亏和权重。' : 'Fill an order to start building positions. The OMS tracks avg cost, P&L, and weights.' }}</p>
      </div>
    </div>

    <!-- TCA Analysis -->
    <div v-if="tab === 'tca'" class="oms-tca">
      <div v-if="tca.total_orders" class="oms-tca-grid">
        <div class="oms-tca-item">
          <div class="oms-tca-label">{{ locale === 'zh-CN' ? '买单数' : 'Buy Orders' }}</div>
          <div class="oms-tca-value oms-pos">{{ tca.buy_orders }}</div>
        </div>
        <div class="oms-tca-item">
          <div class="oms-tca-label">{{ locale === 'zh-CN' ? '卖单数' : 'Sell Orders' }}</div>
          <div class="oms-tca-value oms-neg">{{ tca.sell_orders }}</div>
        </div>
        <div class="oms-tca-item">
          <div class="oms-tca-label">{{ locale === 'zh-CN' ? '平均订单规模' : 'Avg Order Size' }}</div>
          <div class="oms-tca-value">{{ formatNum(tca.avg_order_size) }}</div>
        </div>
        <div class="oms-tca-item">
          <div class="oms-tca-label">{{ locale === 'zh-CN' ? '总成本(bps)' : 'Total Cost (bps)' }}</div>
          <div class="oms-tca-value oms-warn">{{ tca.cost_bps?.toFixed(1) }}</div>
        </div>
        <div class="oms-tca-item">
          <div class="oms-tca-label">{{ locale === 'zh-CN' ? '佣金/成交量' : 'Commission / Volume' }}</div>
          <div class="oms-tca-value">{{ tca.total_volume ? ((tca.total_commission / tca.total_volume) * 10000).toFixed(2) : '0' }} bps</div>
        </div>
        <div class="oms-tca-item">
          <div class="oms-tca-label">{{ locale === 'zh-CN' ? '印花税/成交量' : 'Tax / Volume' }}</div>
          <div class="oms-tca-value">{{ tca.total_volume ? ((tca.total_tax / tca.total_volume) * 10000).toFixed(2) : '0' }} bps</div>
        </div>
        <div class="oms-tca-item">
          <div class="oms-tca-label">{{ locale === 'zh-CN' ? '滑点/成交量' : 'Slippage / Volume' }}</div>
          <div class="oms-tca-value">{{ tca.total_volume ? ((tca.total_slippage / tca.total_volume) * 10000).toFixed(2) : '0' }} bps</div>
        </div>
      </div>
      <div v-else class="oms-empty">
        <div class="oms-empty-icon">&#8776;</div>
        <h3>{{ locale === 'zh-CN' ? '暂无交易数据' : 'No Trade Data' }}</h3>
        <p>{{ locale === 'zh-CN' ? 'TCA（交易成本分析）展示佣金、印花税和滑点明细。' : 'TCA (Trade Cost Analysis) shows commission, tax, and slippage breakdown in basis points.' }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, watch, onMounted } from 'vue'
import { useI18n } from '../i18n/index.js'
import {
  createOrder,
  fillOrder,
  getBaostockStock,
  getBlotter,
  getPositions,
  getTCA,
  getApiErrorMessage,
} from '../api/index.js'

import { formatBackendTime, shanghaiToday } from '../utils/time.js'

const { $t, locale } = useI18n()

const emit = defineEmits(['toast'])

const tab = ref('blotter')
const blotter = ref([])
const positions = ref([])
const positionsPriceBasis = ref('')
const tca = ref({})
const loading = ref(false)
const submitting = ref(false)
const showNewOrder = ref(false)
const fillingId = ref(null)
// order_id -> price draft, seeded from the limit price when we created the
// order (the blotter response does not echo limit_price) or from
// `limit_price` if a future backend starts returning it.
const fillPriceDrafts = reactive({})

const newOrder = ref({
  ticker: '',
  side: 'buy',
  quantity: 1000,
  order_type: 'market',
  limit_price: null,
  strategy: '',
})

// A limit price typed for a limit order must not survive a switch to market:
// it used to be sent with the market order and used directly as the fill
// price, bypassing the quote/staleness check.
watch(() => newOrder.value.order_type, (type) => {
  if (type !== 'limit') newOrder.value.limit_price = null
})

function formatNum(v) {
  if (v == null) return '--'
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(v)
}

function formatTime(t) {
  // Naive Shanghai timestamps, rendered in the operator's locale/timezone
  // (same convention as RunHistory / LiveTrading).
  return formatBackendTime(t, locale.value)
}

// A resting order (pending/submitted/partially filled) can be filled from the
// UI; complete orders cannot.
function canFill(o) {
  if (!o) return false
  if (!['pending', 'submitted', 'partial'].includes(o.status)) return false
  const remaining = o.remaining_quantity == null ? o.quantity : o.remaining_quantity
  return Number(remaining) > 0
}

function fillPricePlaceholder(o) {
  const limit = o?.limit_price ?? fillPriceDrafts[o?.order_id]
  if (limit != null && Number(limit) > 0) return String(limit)
  return locale.value === 'zh-CN' ? '成交价' : 'price'
}

async function fillRow(order) {
  const price = Number(fillPriceDrafts[order.order_id])
  if (!Number.isFinite(price) || price <= 0) {
    emit('toast', {
      message: locale.value === 'zh-CN' ? '请输入有效的成交价格' : 'Enter a valid fill price',
      type: 'error',
    })
    return
  }
  fillingId.value = order.order_id
  try {
    const result = await fillOrder({ order_id: order.order_id, price })
    const basis = result?.price_basis ? String(result.price_basis).toUpperCase() : ''
    const filledAt = Number(result?.avg_price ?? price)
    emit('toast', {
      message: locale.value === 'zh-CN'
        ? `订单 ${order.order_id} 已成交 @ ${filledAt.toFixed(2)}${basis ? `（${basis}）` : ''}`
        : `Order ${order.order_id} filled @ ${filledAt.toFixed(2)}${basis ? ` (${basis})` : ''}`,
      type: 'success',
    })
  } catch (err) {
    emit('toast', {
      message: (locale.value === 'zh-CN' ? '成交失败: ' : 'Fill failed: ') + getApiErrorMessage(err),
      type: 'error',
    })
  } finally {
    fillingId.value = null
    // The server book changed even when the fill failed (it may have filled
    // partially); never leave the blotter silently stale.
    await refreshAll()
  }
}

async function submitOrder() {
  if (!newOrder.value.ticker) {
    emit('toast', { message: locale.value === 'zh-CN' ? '请输入股票代码' : 'Ticker is required', type: 'error' })
    return
  }
  submitting.value = true
  let created = null
  // Snapshot the submitted order: the form stays editable while the POST is in
  // flight, and reading `newOrder` after the await could quote a different
  // ticker / type than the order that was actually created.
  const submitted = {
    ticker: newOrder.value.ticker,
    side: newOrder.value.side,
    quantity: newOrder.value.quantity,
    order_type: newOrder.value.order_type,
    limit_price: newOrder.value.order_type === 'limit' ? newOrder.value.limit_price : null,
    strategy: newOrder.value.strategy,
  }
  try {
    created = await createOrder(submitted)

    // Remember the limit price so the row's Fill action can offer it (the
    // blotter response does not echo limit_price).
    if (submitted.limit_price != null && Number(submitted.limit_price) > 0) {
      fillPriceDrafts[created.order_id] = String(submitted.limit_price)
    }

    // Auto-fill market orders (simulate immediate execution). The fill price
    // must come from a real quote: the old fallback was Math.random()*50+100,
    // so every stock filled at CNY 100-150 regardless of its price.
    if (submitted.order_type === 'market') {
      try {
        // Always quote: a market order must not reuse a leftover limit price.
        const quote = await getBaostockStock(submitted.ticker)
        const today = shanghaiToday()
        // The quote endpoint falls back to the last available bar (up to a
        // few days old). Filling a market order at a stale close is worse
        // than refusing: ask for an explicit limit price instead.
        if (quote.date && quote.date !== today) {
          throw new Error(
            (locale.value === 'zh-CN'
              ? `行情为 ${quote.date} 收盘，非当日；请填写限价后下单`
              : `Last bar is ${quote.date}, not today; enter a limit price`),
          )
        }
        const price = quote.close
        if (!price || price <= 0) {
          throw new Error('No market price available for ' + submitted.ticker)
        }
        await fillOrder({ order_id: created.order_id, price })
        emit('toast', { message: locale.value === 'zh-CN' ? `订单 ${created.order_id} 已成交` : `Order ${created.order_id} filled`, type: 'success' })
      } catch (fillErr) {
        // The order already exists server-side; report it and point at the
        // row's Fill action instead of claiming the whole submission failed.
        emit('toast', {
          message: locale.value === 'zh-CN'
            ? `订单 ${created.order_id} 已提交但未成交：${getApiErrorMessage(fillErr)}。可在订单簿中手动成交。`
            : `Order ${created.order_id} was submitted but not filled: ${getApiErrorMessage(fillErr)}. Fill it manually from the blotter.`,
          type: 'warning',
        })
      }
    } else {
      emit('toast', {
        message: locale.value === 'zh-CN'
          ? `限价单 ${created.order_id} 已创建（${created.status}），可在订单簿中成交`
          : `Limit order ${created.order_id} created (${created.status}); fill it from the blotter`,
        type: 'success',
      })
    }
    showNewOrder.value = false
  } catch (err) {
    emit('toast', {
      message: (locale.value === 'zh-CN' ? '订单提交失败: ' : 'Order failed: ') + getApiErrorMessage(err),
      type: 'error',
    })
  } finally {
    submitting.value = false
    // Refresh unconditionally: a failed fill still mutated the server book.
    await refreshAll()
  }
}

async function refreshAll() {
  loading.value = true
  try {
    const [b, p, t] = await Promise.allSettled([getBlotter(), getPositions(), getTCA()])
    if (b.status === 'fulfilled') blotter.value = b.value.orders || []
    if (p.status === 'fulfilled') {
      positions.value = p.value.positions || []
      positionsPriceBasis.value = p.value.price_basis || ''
    }
    if (t.status === 'fulfilled') tca.value = t.value
    // One failing endpoint used to reject Promise.all and leave all three
    // tables at their previous state.
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

.oms-action-cell { white-space: nowrap; }

.oms-fill-inline {
  display: flex;
  gap: 4px;
  align-items: center;
}

.oms-fill-input {
  width: 74px;
  min-width: 0;
  padding: 3px 6px;
  font-size: 10px;
}

.oms-fill-btn {
  padding: 3px 8px;
  border: 1px solid var(--green);
  background: transparent;
  color: var(--green);
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
}

.oms-fill-btn:hover { background: rgba(52,211,153,0.12); }
.oms-fill-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.oms-basis {
  font-family: var(--font-mono);
  font-size: 9px;
  text-align: center;
  color: var(--green);
}
</style>
