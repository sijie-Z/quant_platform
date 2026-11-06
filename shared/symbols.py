"""Canonical A-share symbol helpers shared by the research core and live layer.

A-share symbols arrive in several shapes across the platform:
``600519`` (bare), ``600519.SH`` (suffix), ``sh600519`` / ``sh.600519``
(prefix), ``sz300001`` (prefix, ChiNext) and Beijing codes (``4``/``8``/
``920``).  Position identity, price-limit rules and broker routing must agree
on one form, otherwise ``600519`` and ``600519.SH`` become two different
positions and ``sz300001`` falls through to the 10% limit rule.

``normalize_symbol`` returns the canonical ``CODE.EXCHANGE`` form.  An
explicitly stated exchange always wins (e.g. ``000001.SH`` is the Shanghai
index, not Ping An Bank ``000001.SZ``); only a bare code is inferred from its
prefix.

This module lives in ``shared/`` because both the research core
(``data/pipeline.py`` board limits) and the live layer
(``core/instrument.py``, broker routing) consume it, and the architecture
boundary forbids ``data -> core`` imports.
"""

from __future__ import annotations

import re
from typing import Any

EXCHANGES = ("SH", "SZ", "BJ")

_SUFFIX_RE = re.compile(r"^(\d{6})[.\-_]?(SH|SZ|BJ)$")
_PREFIX_RE = re.compile(r"^(SH|SZ|BJ)[.\-_]?(\d{6})$")
_BARE_RE = re.compile(r"^(\d{6})$")
_CODE_RE = re.compile(r"(\d{6})")


def symbol_code(symbol: Any) -> str | None:
    """Extract the six-digit A-share code from any accepted symbol format."""
    match = _CODE_RE.search(str(symbol))
    return match.group(0) if match else None


def infer_exchange(code: str) -> str:
    """Infer the exchange for a bare six-digit code.

    Beijing: ``4``/``8``/``920`` prefixes (30% limit).  Shanghai: ``5``
    (funds/ETFs), ``6`` (stocks) and ``9`` (B-shares); Shenzhen: ``0``/``2``/
    ``3``.  ``1``-prefixed instruments are split by their real ranges --
    ``12``/``15``/``16``/``18`` are Shenzhen bonds/funds, the rest
    (``110``/``111``/``113``...) are Shanghai bonds.  The ``920`` check
    deliberately precedes the ``9`` branch so Beijing does not get Shanghai's
    rules.  Unknown prefixes default to Shenzhen (the historical behaviour).
    """
    if code.startswith("920") or code.startswith(("4", "8")):
        return "BJ"
    if code.startswith(("5", "6", "9")):
        return "SH"
    if code.startswith(("0", "2", "3")):
        return "SZ"
    if code.startswith(("12", "15", "16", "18")):
        return "SZ"
    if code.startswith("1"):
        return "SH"
    return "SZ"


def normalize_symbol(symbol: Any) -> str:
    """Return the canonical ``CODE.EXCHANGE`` symbol (e.g. ``300001.SZ``).

    Handles suffix (``300001.SZ``), prefix (``sz300001``), dotted prefix
    (``sh.688001``), lower/upper case and bare codes (``600519``,
    ``920001``).  An explicitly declared exchange is preserved, so
    ``000001.SH`` (index) and ``000001.SZ`` (stock) stay distinct.  A value
    without a six-digit code is returned upper-cased and stripped so callers
    can still log it; such a symbol never matches a real instrument.
    """
    text = str(symbol).strip().upper()
    match = _SUFFIX_RE.fullmatch(text)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    match = _PREFIX_RE.fullmatch(text)
    if match:
        return f"{match.group(2)}.{match.group(1)}"
    match = _BARE_RE.fullmatch(text)
    if match:
        code = match.group(1)
        return f"{code}.{infer_exchange(code)}"
    return text


def symbol_exchange(symbol: Any) -> str:
    """Exchange of a symbol in canonical form (``SH``/``SZ``/``BJ``).

    Returns an empty string when no exchange can be determined.
    """
    canonical = normalize_symbol(symbol)
    suffix = canonical.rsplit(".", 1)[-1]
    return suffix if suffix in EXCHANGES else ""


def price_limit_threshold(symbol: Any) -> float:
    """Daily price-limit threshold for an A-share symbol.

    Board rules: ChiNext/STAR (``300``/``301``/``688``/``689``) 20%, Beijing
    (``4``/``8``/``920``) 30%, everything else 10%.  The six-digit code drives
    the rule, so prefix-style symbols (``sz300001``, ``sh.688001``) and
    Beijing codes get the correct limit instead of falling through a
    ``startswith`` check on the raw label.  Unknown formats default to 10%.
    """
    code = symbol_code(symbol)
    if code is None:
        return 0.10
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("4", "8", "920")):
        return 0.30
    return 0.10
