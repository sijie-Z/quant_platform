"""Regression tests for POST /api/portfolio/import (BUG-40).

The code-filter regex was written `r"\\d{6}"` -- a six-character pattern
(escaped backslash, then six literal `d`s). It matched nothing, and only the
string `\\dddddd` would have passed. Every import therefore returned
`{"status": "ok", "n_stocks": 0}` with the whole file dropped.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.api import routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _import(client, rows):
    return client.post("/api/portfolio/import", json={"data": rows})


class TestPortfolioImportKeepsValidCodes:
    def test_six_digit_codes_survive_the_filter(self, client):
        resp = _import(client, [
            {"code": "600519", "hold_vol": 100},
            {"code": "000001", "hold_vol": 200},
        ])
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["n_stocks"] == 2, "every row was dropped by the code filter"
        assert body["total_shares"] == 300

    def test_leading_zeros_are_preserved(self, client):
        """000001 is a real code; zfill must survive the regex."""
        body = _import(client, [{"code": "1", "hold_vol": 100}]).json()
        assert body["n_stocks"] == 1
        assert body["sample"][0]["code"] == "000001"

    def test_rows_with_junk_codes_are_dropped_not_kept(self, client):
        body = _import(client, [
            {"code": "600519", "hold_vol": 100},
            {"code": "XX", "hold_vol": 100},
            {"code": "1234567", "hold_vol": 100},
        ]).json()
        assert body["n_stocks"] == 1
        assert body["sample"][0]["code"] == "600519"


class TestImportDoesNotSilentlySucceedOnNothing:
    def test_all_rows_invalid_raises_rather_than_reporting_ok(self, client):
        """Returning 200 with n_stocks=0 for a file full of rows is the
        silent-success failure mode this bug produced."""
        resp = _import(client, [
            {"code": "XX", "hold_vol": 100},
            {"code": "YY", "hold_vol": 100},
        ])
        assert resp.status_code == 400
        assert "6-digit code" in resp.json()["detail"]

    def test_zero_holdings_are_filtered_before_the_code_check(self, client):
        """A file of rows with no shares is legitimately empty, not an error."""
        resp = _import(client, [{"code": "600519", "hold_vol": 0}])
        assert resp.status_code == 200
        assert resp.json()["n_stocks"] == 0
