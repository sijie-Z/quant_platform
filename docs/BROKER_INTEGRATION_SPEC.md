# Broker Integration Spec — alpha-v1.0 实盘接入层

> **状态**：设计文档，未实现
> **前提**：需要券商账户 + API 权限
> **风险警告**：实盘交易可能亏损本金。本系统不构成投资建议。

---

## 1. 架构

```
live_engine.py
    │  signal, risk, state
    ▼
execution/adapter.py   ← broker adapter layer
    │ 统一的 order/report 接口
    ▼
broker SDK (IB / 富途 / 某国内券商)
    │
    ▼
交易所
```

## 2. BrokerAdapter 接口

```python
class BrokerAdapter(ABC):
    """券商接入抽象层."""

    @abstractmethod
    def connect(self, config: dict) -> bool:
        """连接券商 API."""

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """查询账户信息 (现金, 持仓, 购买力)."""

    @abstractmethod
    def place_order(self, order: Order) -> OrderId:
        """下单."""

    @abstractmethod
    def cancel_order(self, order_id: OrderId) -> bool:
        """撤单."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """查询当前持仓."""

    @abstractmethod
    def get_orders(self) -> list[OrderStatus]:
        """查询未成交订单."""

    @abstractmethod
    def disconnect(self):
        """断开连接."""
```

## 3. 数据模型

```python
@dataclass
class Order:
    asset: str
    side: Literal["buy", "sell"]
    quantity: int
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = None

@dataclass
class OrderStatus:
    order_id: str
    asset: str
    side: str
    quantity: int
    filled: int
    status: str  # pending, filled, cancelled, rejected
    avg_price: float

@dataclass
class AccountInfo:
    cash: float
    equity: float
    buying_power: float
    positions: list[Position]
```

## 4. 接入方式

| 券商 | SDK | 接入难度 | 备注 |
|------|-----|---------|------|
| Interactive Brokers | ib_insync | 中 | 标准 API，文档完善 |
| 富途牛牛 | FutuOpenD | 中 | 需要 OpenD 进程 |
| 国内券商 (QMT) | xtquant | 低 | 国金/华鑫等支持 miniQMT |
| 华泰 |  | 高 | 需要额外权限 |

## 5. 安全规则（不可违反）

```
1. adapter 不存储任何账号密码
2. 首次交易前必须人工确认
3. 每笔订单有金额上限保护
4. Kill switch 必须可随时手动触发
5. 默认为 paper mode，需显式切换为 live
```

## 6. 实现优先级

```
P0: BrokerAdapter 接口定义
P1: PaperBrokerAdapter (已有 SimulatedExchange, 可复用)
P2: 接入一家具体券商
P3: 订单状态同步 + 对账
P4: 断线重连 + 错误恢复
```
