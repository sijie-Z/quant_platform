"""ValidatedProvider: forward only the primary source's corporate-action feed.

An action is a discrete corporate event, not a price observation, so the
wrapper must not fuse secondary feeds (different announcement dates,
amounts or symbol conventions could invent events).  The primary's feed and
``corporate_actions_available`` flag are forwarded unchanged; a
secondary-only feed is deliberately not used.
"""

from __future__ import annotations

import pandas as pd
import pytest
from quant_platform.data.providers.base import (
    DataProvider,
    empty_corporate_actions,
)
from quant_platform.data.providers.validated_provider import ValidatedProvider


def _action(ex_date="2024-01-05", asset="600519.SH", cash=0.5):
    return {
        "ex_date": ex_date, "asset": asset, "kind": "cash",
        "cash_per_share": cash, "split_ratio": 1.0,
    }


class _ActionProvider(DataProvider):
    corporate_actions_available = True

    def __init__(self, rows=None, raises: bool = False):
        self._rows = rows
        self._raises = raises
        self.calls = 0

    def get_corporate_actions(self, start_date, end_date):
        self.calls += 1
        if self._raises:
            raise RuntimeError("feed unavailable")
        if self._rows is None:
            return empty_corporate_actions()
        return pd.DataFrame(self._rows)

    def get_prices(self, start_date, end_date, fields=None):
        return pd.DataFrame()

    def get_financials(self, start_date, end_date):
        return pd.DataFrame()

    def get_benchmark(self, start_date, end_date):
        return pd.Series(dtype=float)

    def get_metadata(self):
        return pd.DataFrame()


class _NoFeedProvider(DataProvider):
    def get_prices(self, start_date, end_date, fields=None):
        return pd.DataFrame()

    def get_financials(self, start_date, end_date):
        return pd.DataFrame()

    def get_benchmark(self, start_date, end_date):
        return pd.Series(dtype=float)

    def get_metadata(self):
        return pd.DataFrame()


def test_primary_feed_is_forwarded_not_merged_with_secondary():
    primary = _ActionProvider([_action(cash=0.5)])
    secondary = _ActionProvider([
        _action(asset="000001.SZ", cash=9.9),
        _action(cash=7.7),
    ])
    provider = ValidatedProvider(
        {"primary": primary, "secondary": secondary}, primary="primary",
    )

    frame = provider.get_corporate_actions("2024-01-01", "2024-12-31")

    assert list(frame["cash_per_share"]) == [0.5]
    assert list(frame["asset"]) == ["600519.SH"]
    assert provider.corporate_actions_available is True
    assert secondary.calls == 0, (
        "secondary corporate-action feeds must never be queried or merged"
    )


def test_secondary_only_feed_is_not_used():
    primary = _NoFeedProvider()
    secondary = _ActionProvider([_action(cash=0.5)])
    provider = ValidatedProvider(
        {"primary": primary, "secondary": secondary}, primary="primary",
    )

    frame = provider.get_corporate_actions("2024-01-01", "2024-12-31")

    assert frame.empty
    assert list(frame.columns) == [
        "ex_date", "asset", "kind", "cash_per_share", "split_ratio",
    ]
    assert provider.corporate_actions_available is False
    assert secondary.calls == 0


def test_primary_feed_failure_fails_closed():
    provider = ValidatedProvider(
        {"primary": _ActionProvider(raises=True)}, primary="primary",
    )
    frame = provider.get_corporate_actions("2024-01-01", "2024-12-31")

    assert frame.empty
    assert provider.corporate_actions_available is False


def test_combined_row_is_expanded_by_the_forwarding_contract():
    provider = ValidatedProvider(
        {"primary": _ActionProvider([{
            "ex_date": "2024-01-05", "asset": "600519", "kind": "cash",
            "cash_per_share": 0.5, "split_ratio": 2.0,
        }])},
        primary="primary",
    )
    frame = provider.get_corporate_actions("2024-01-01", "2024-12-31")

    assert sorted(frame["kind"].tolist()) == ["cash", "split"]
    assert list(frame["asset"].unique()) == ["600519.SH"]
    assert provider.corporate_actions_available is True
