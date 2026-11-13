"""Issue #74: one canonical order model, one explicitly named broker DTO.

`trading.broker.Order` and `execution.models.Order` both defined an order
plus three duplicate enums. The broker shape is now `BrokerOrder`, the OMS
model is the single domain model, and the enums are shared.
"""

from quant_platform.execution import models as execution_models
from quant_platform.trading import broker
from quant_platform.trading.broker import BrokerOrder, Order


def test_broker_order_alias_is_the_same_class():
    assert Order is BrokerOrder


def test_enums_are_shared_not_redefined():
    assert broker.OrderSide is execution_models.OrderSide
    assert broker.OrderType is execution_models.OrderType
    assert broker.OrderStatus is execution_models.OrderStatus


def test_broker_order_converts_to_the_domain_model():
    broker_order = BrokerOrder(
        code="600519",
        side=broker.OrderSide.BUY,
        order_type=broker.OrderType.LIMIT,
        quantity=200,
        price=1800.0,
        filled_quantity=100,
        filled_price=1799.5,
        status=broker.OrderStatus.PARTIAL,
        commission=5.0,
    )

    domain = broker_order.to_execution_order()

    assert domain.ticker == "600519"
    assert domain.limit_price == 1800.0
    assert domain.quantity == 200
    assert domain.filled_quantity == 100
    assert domain.avg_fill_price == 1799.5
    assert domain.status is execution_models.OrderStatus.PARTIAL
    assert domain.fills[0].commission == 5.0


def test_domain_order_converts_back_to_a_broker_dto():
    domain = execution_models.Order(
        ticker="000001.SZ",
        side=execution_models.OrderSide.SELL,
        order_type=execution_models.OrderType.MARKET,
        quantity=100,
    )
    domain.fills.append(
        execution_models.Fill(price=10.5, quantity=100, commission=5.0)
    )
    domain.status = execution_models.OrderStatus.FILLED

    broker_order = BrokerOrder.from_execution_order(domain)

    assert broker_order.code == "000001.SZ"
    assert broker_order.side is execution_models.OrderSide.SELL
    assert broker_order.filled_quantity == 100
    assert broker_order.filled_price == 10.5
    assert broker_order.status is execution_models.OrderStatus.FILLED
    assert broker_order.price == 0.0
