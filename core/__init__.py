"""Core architecture — event-driven, persistent, observable.

This module provides the foundational infrastructure:
- EventBus: async pub/sub for decoupled component communication
- Store: SQLite persistence for all state
- StateMachine: portfolio lifecycle management
- AuditLog: compliance-grade decision tracking
- Scheduler: trading session orchestration
"""
