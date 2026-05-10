"""Microservice skeletons for distributed deployment.

Each service is a standalone process that communicates via MessageBus.

Services:
- DataService: Market data ingestion and distribution
- RiskService: Real-time risk computation
- ExecutionService: Order routing and execution
- FactorService: Factor computation and signal generation

Usage:
    # Start a service
    python -m quant_platform.services.risk_service

    # Or programmatically
    from quant_platform.services.risk_service import RiskService
    service = RiskService(bus=create_message_bus("kafka"))
    await service.run()
"""
