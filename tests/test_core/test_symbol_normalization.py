"""Canonical A-share symbol normalization (D19/D30/D41).

One helper at the boundary: suffix, prefix, dotted-prefix and bare forms all
collapse to ``CODE.EXCHANGE``; board price limits follow the six-digit code
(ChiNext/STAR 20%, Beijing 30%); explicit exchanges are never overridden.
"""

from __future__ import annotations

import pytest

from quant_platform.core.instrument import (
    AssetUniverse,
    Instrument,
    normalize_symbol,
    price_limit_threshold,
    symbol_code,
    symbol_exchange,
)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("600519", "600519.SH"),
        ("600519.SH", "600519.SH"),
        ("sh600519", "600519.SH"),
        ("sh.600519", "600519.SH"),
        ("SH.600519", "600519.SH"),
        ("sz300001", "300001.SZ"),
        ("300001.SZ", "300001.SZ"),
        ("sz.300001", "300001.SZ"),
        ("sh.688001", "688001.SH"),
        ("688001", "688001.SH"),
        ("000001", "000001.SZ"),
        ("000001.SZ", "000001.SZ"),
        # Explicit exchange beats the bare-code inference: 000001.SH is the
        # Shanghai index, 000001.SZ is Ping An Bank.  They must stay distinct.
        ("000001.SH", "000001.SH"),
        ("sh.000001", "000001.SH"),
        ("920001", "920001.BJ"),
        ("920001.BJ", "920001.BJ"),
        ("bj920001", "920001.BJ"),
        ("430047", "430047.BJ"),
        ("830799", "830799.BJ"),
        ("bj.830799", "830799.BJ"),
        # Funds/bonds: 5xxxxx and 11xxxx are Shanghai, 15/16/18 and 12 are
        # Shenzhen (the old inference sent every 5xxxxx fund to Shenzhen).
        ("510300", "510300.SH"),
        ("588000", "588000.SH"),
        ("500001", "500001.SH"),
        ("113050", "113050.SH"),
        ("110059", "110059.SH"),
        ("159915", "159915.SZ"),
        ("160105", "160105.SZ"),
        (" 600519.sh ", "600519.SH"),
    ],
)
def test_normalize_symbol_table(raw, canonical):
    assert normalize_symbol(raw) == canonical


def test_symbol_helpers():
    assert symbol_code("sz300001") == "300001"
    assert symbol_code("no-code") is None
    assert symbol_exchange("sh.688001") == "SH"
    assert symbol_exchange("920001") == "BJ"
    assert symbol_exchange("not-a-symbol") == ""


@pytest.mark.parametrize(
    ("symbol", "threshold"),
    [
        ("600519", 0.10),
        ("600519.SH", 0.10),
        ("000001.SZ", 0.10),
        ("300001.SZ", 0.20),
        ("sz300001", 0.20),
        ("301001", 0.20),
        ("sh.688001", 0.20),
        ("688001.SH", 0.20),
        ("920001", 0.30),
        ("920001.BJ", 0.30),
        ("430047", 0.30),
        ("830799", 0.30),
        ("bj.830799", 0.30),
    ],
)
def test_price_limit_threshold(symbol, threshold):
    assert price_limit_threshold(symbol) == pytest.approx(threshold)


def test_instrument_stock_uses_board_rules():
    assert Instrument.stock("300750").price_limit == pytest.approx(0.20)
    assert Instrument.stock("688001").price_limit == pytest.approx(0.20)
    assert Instrument.stock("920001").price_limit == pytest.approx(0.30)
    assert Instrument.stock("430047").price_limit == pytest.approx(0.30)
    assert Instrument.stock("600519").price_limit == pytest.approx(0.10)


def test_asset_universe_lookup_accepts_both_forms():
    universe = AssetUniverse()
    universe.add(Instrument.stock("600519"))
    universe.add(Instrument.stock("300750"))

    assert universe.get("600519") is not None
    assert universe.get("600519.SH") is not None
    assert universe.get("sh.600519") is not None
    assert universe.get("300750") is not None
    assert universe.get("300750.SZ") is not None
    assert universe.get("600000") is None
    # An explicit exchange must not resolve to a same-numbered instrument on
    # another exchange (000001.SH index vs 000001.SZ stock).
    assert universe.get("600519") is not None
    universe.add(Instrument.stock("000001", exchange="SZSE"))
    assert universe.get("000001") is not None       # bare -> inferred SZ
    assert universe.get("000001.SH") is None        # explicit SH: no match
    assert universe.get_or_default("600519.SH").symbol == "600519"


def test_qmt_symbol_mapping_handles_prefix_and_beijing():
    from quant_platform.trading.qmt_utils import (
        classify_exchange,
        from_qmt_code,
        to_qmt_code,
        to_qmt_exchange,
    )

    assert to_qmt_code("600519") == "600519.SH"
    assert to_qmt_code("sz300001") == "300001.SZ"
    assert to_qmt_code("sh.688001") == "688001.SH"
    assert to_qmt_code("920001") == "920001.BJ"
    assert to_qmt_code("bj.830799") == "830799.BJ"
    assert to_qmt_code("600519.SH") == "600519.SH"  # idempotent

    assert from_qmt_code("600519.SH") == "600519"
    assert from_qmt_code("920001.BJ") == "920001"

    assert classify_exchange("600519") == "SH"
    assert classify_exchange("sh.688001") == "SH"   # used to fall to SZ
    assert classify_exchange("sz300001") == "SZ"
    assert classify_exchange("830799") == "BJ"      # used to fall to SZ
    assert classify_exchange("920001") == "BJ"

    assert to_qmt_exchange("600519") == 1
    assert to_qmt_exchange("000001") == 2
    assert to_qmt_exchange("920001") == 0  # Beijing
