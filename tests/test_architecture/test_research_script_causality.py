"""Research scripts must not embed the forward-return window in the factor.

`run_factor.py` and `tools/common_runner.py` computed 12-month momentum as
`rolling(252).prod().shift(-21)`: `.shift(-21)` moves the rolling value 21
days *backward*, so `factor[t]` contained returns through `t+21` — the exact
window the IC was measured against. The registry factor uses `shift(+21)`
(skip the recent month) correctly.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_research_scripts_do_not_shift_factors_backwards():
    for name in ("run_factor.py", "tools/common_runner.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert ".shift(-21)" not in text, (
            f"{name} embeds 21 days of future returns in its momentum factor"
        )
