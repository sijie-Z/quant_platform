"""lab package — Knowledge Layer for quant_platform.

Per CONSTITUTION.md: experiments, factors-as-content, reports, and the
research-run machinery live here. This is where Knowledge compounds
(Principle 3). Alpha (research output) lives here as an artifact, NOT as a
system layer — Production (prod/) consumes Alpha but does not own it.

v0.1 contents:
    runs/    — First Honest Research Run and its successors
    registry/— the run-record store (fails ALSO recorded, per Principle 3)
    reports/ — auto-generated research reports (WARNINGs driven by Registry)
"""
