"""Shared modules consumed by both the research core and the live platform.

STATUS: COMPATIBILITY SHIM (2026-10) — these modules re-export `quant_core.*`;
no runtime module imports the shims. See docs/DEAD_CODE_STATUS.md / #77.

These modules must stay free of live-specific dependencies so they can be
packaged into quant-core (or a shared common package) during the split.
"""
