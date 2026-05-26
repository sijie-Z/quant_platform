"""Tests for multi-tenant context and isolation."""

from unittest.mock import MagicMock

import pytest

from quant_platform.core.context import TenantContext


class TestTenantContext:
    """Test TenantContext set/get and isolation."""

    def setup_method(self):
        TenantContext.clear()

    def teardown_method(self):
        TenantContext.clear()

    def test_default_context(self):
        """Default context should have tenant_id='default'."""
        ctx = TenantContext.get_current()
        assert ctx.tenant_id == "default"

    def test_set_and_get(self):
        """set_current should make context available via get_current."""
        ctx = TenantContext(tenant_id="fund_001", strategy_id="mom_v2")
        TenantContext.set_current(ctx)

        retrieved = TenantContext.get_current()
        assert retrieved.tenant_id == "fund_001"
        assert retrieved.strategy_id == "mom_v2"

    def test_clear(self):
        """clear() should reset to default."""
        ctx = TenantContext(tenant_id="fund_001")
        TenantContext.set_current(ctx)
        TenantContext.clear()

        retrieved = TenantContext.get_current()
        assert retrieved.tenant_id == "default"

    def test_account_mapping(self):
        """Context should carry account mapping."""
        ctx = TenantContext(
            tenant_id="fund_002",
            account_mapping={"broker_a": "ACC123", "broker_b": "ACC456"},
        )
        TenantContext.set_current(ctx)

        retrieved = TenantContext.get_current()
        assert retrieved.account_mapping["broker_a"] == "ACC123"

    def test_risk_limits(self):
        """Context should carry risk limits."""
        ctx = TenantContext(
            tenant_id="fund_003",
            risk_limits={"max_position": 0.05, "max_drawdown": 0.10},
        )
        TenantContext.set_current(ctx)

        retrieved = TenantContext.get_current()
        assert retrieved.risk_limits["max_position"] == 0.05

    def test_context_isolation(self):
        """Different set_current calls should overwrite previous context."""
        ctx1 = TenantContext(tenant_id="tenant_A")
        ctx2 = TenantContext(tenant_id="tenant_B")

        TenantContext.set_current(ctx1)
        assert TenantContext.get_current().tenant_id == "tenant_A"

        TenantContext.set_current(ctx2)
        assert TenantContext.get_current().tenant_id == "tenant_B"


class TestOMSTenantIntegration:
    """Test OMS orders carry tenant_id."""

    def setup_method(self):
        TenantContext.clear()

    def teardown_method(self):
        TenantContext.clear()

    def test_order_gets_tenant_id(self):
        """create_order should stamp tenant_id from context."""
        from quant_platform.execution.oms import OrderManager

        TenantContext.set_current(TenantContext(tenant_id="fund_alpha"))
        om = OrderManager(initial_cash=1_000_000)
        order = om.create_order("600519", "buy", 1000)

        assert order.tenant_id == "fund_alpha"

    def test_order_default_tenant(self):
        """Without explicit context, order should get 'default'."""
        from quant_platform.execution.oms import OrderManager

        om = OrderManager(initial_cash=1_000_000)
        order = om.create_order("600519", "buy", 1000)

        assert order.tenant_id == "default"

    def test_blotter_filters_by_tenant(self):
        """get_order_blotter should filter by tenant_id."""
        from quant_platform.execution.oms import OrderManager

        om = OrderManager(initial_cash=10_000_000)

        # Create orders for tenant A
        TenantContext.set_current(TenantContext(tenant_id="tenant_A"))
        order_a = om.create_order("600519", "buy", 1000)
        om.fill_order(order_a.order_id, 100.0, 1000)

        # Create orders for tenant B
        TenantContext.set_current(TenantContext(tenant_id="tenant_B"))
        order_b = om.create_order("000001", "buy", 2000)
        om.fill_order(order_b.order_id, 10.0, 2000)

        # Filter by tenant A
        blotter_a = om.get_order_blotter(tenant_id="tenant_A")
        assert all(o["tenant_id"] == "tenant_A" for o in blotter_a)

        # Filter by tenant B
        blotter_b = om.get_order_blotter(tenant_id="tenant_B")
        assert all(o["tenant_id"] == "tenant_B" for o in blotter_b)

        # No filter returns all
        blotter_all = om.get_order_blotter()
        assert len(blotter_all) == 2


class TestHealthCheckTenant:
    """Test HealthCheck uses tenant context."""

    def setup_method(self):
        TenantContext.clear()

    def teardown_method(self):
        TenantContext.clear()

    def test_healthcheck_runs_with_tenant(self):
        """HealthCheck should run without error under tenant context."""
        from quant_platform.risk.healthcheck import HealthCheck

        TenantContext.set_current(TenantContext(tenant_id="fund_x"))
        mock_broker = MagicMock()
        mock_broker.get_account.return_value = {"cash": 500_000}
        mock_broker.get_positions.return_value = []

        hc = HealthCheck(broker=mock_broker, min_cash=100_000)
        report = hc.run_all_sync()

        assert report.overall_passed

    def test_healthcheck_balance_per_tenant(self):
        """HealthCheck should check balance for the current tenant."""
        from quant_platform.risk.healthcheck import HealthCheck, SystemBlockError

        TenantContext.set_current(TenantContext(tenant_id="fund_poor"))
        mock_broker = MagicMock()
        mock_broker.get_account.return_value = {"cash": 100}
        mock_broker.get_positions.return_value = []

        hc = HealthCheck(broker=mock_broker, min_cash=100_000)

        with pytest.raises(SystemBlockError):
            hc.run_all_sync()
