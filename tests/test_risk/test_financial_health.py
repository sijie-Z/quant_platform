"""Tests for financial health checks (fraud, ST risk, owner earnings)."""

import pandas as pd
import pytest

from quant_platform.risk.financial_health import (
    FraudDetector,
    FraudReport,
    assess_st_risk,
    owner_earnings,
    estimate_maintenance_capex,
    moat_score,
)


class TestOwnerEarnings:
    def test_basic_calculation(self):
        result = owner_earnings(
            net_income=1000,
            depreciation=200,
            maintenance_capex=150,
        )
        assert result["owner_earnings"] == 1050  # 1000 + 200 - 150
        assert result["earnings_quality"] == "high"

    def test_negative_owner_earnings(self):
        result = owner_earnings(
            net_income=100,
            depreciation=50,
            maintenance_capex=300,
        )
        assert result["owner_earnings"] < 0
        assert result["earnings_quality"] == "low"


class TestEstimateMaintenanceCapex:
    def test_capex_less_than_depreciation(self):
        result = estimate_maintenance_capex(total_capex=100, depreciation=150)
        assert result == 100  # All capex is maintenance

    def test_capex_exceeds_depreciation(self):
        result = estimate_maintenance_capex(total_capex=200, depreciation=100)
        # 100 + (200 - 100) * 0.3 = 130
        assert result == pytest.approx(130.0)


class TestMoatScore:
    def test_wide_moat(self):
        result = moat_score(
            gross_margin_stability=0.01,
            roe_avg=0.25,
            debt_equity=0.1,
            pricing_power=True,
        )
        assert result["score"] >= 8
        assert "宽" in result["label"]

    def test_no_moat(self):
        result = moat_score(
            gross_margin_stability=0.15,
            roe_avg=0.08,
            debt_equity=1.5,
            pricing_power=False,
        )
        assert result["score"] < 4


class TestFraudDetector:
    def test_empty_df(self):
        detector = FraudDetector()
        report = detector.analyze(pd.DataFrame())
        assert isinstance(report, FraudReport)
        assert report.total_score == 0

    def test_healthy_company(self):
        df = pd.DataFrame({
            "gross_margin": [0.40, 0.41, 0.42],
            "cfo": [100, 110, 120],
            "net_income": [100, 110, 120],
            "revenue": [1000, 1100, 1200],
            "receivables": [200, 205, 210],
            "goodwill": [10, 10, 10],
            "net_assets": [500, 550, 600],
        })
        detector = FraudDetector()
        report = detector.analyze(df)
        assert report.risk_level == "低风险"

    def test_audit_fail_direct_exclude(self):
        df = pd.DataFrame({
            "audit_opinion": ["无法表示意见"],
            "gross_margin": [0.35],
        })
        detector = FraudDetector()
        report = detector.analyze(df)
        assert report.risk_level == "直接排除"

    def test_cfo_negative_flag(self):
        df = pd.DataFrame({
            "cfo": [-100, -110, -120],
            "net_income": [50, 55, 60],
            "gross_margin": [0.35, 0.36, 0.37],
            "revenue": [1000, 1100, 1200],
            "receivables": [200, 210, 220],
            "goodwill": [10, 10, 10],
            "net_assets": [500, 550, 600],
        })
        detector = FraudDetector()
        report = detector.analyze(df)
        assert report.total_score > 0
        assert "中风险" in report.risk_level or "高风险" in report.risk_level


class TestSTRisk:
    def test_empty_df(self):
        report = assess_st_risk(pd.DataFrame())
        assert report.total_score == 0

    def test_consecutive_losses(self):
        df = pd.DataFrame({
            "net_income": [-100, -200, -300],
            "revenue": [1e9, 1e9, 1e9],
            "net_assets": [500, 300, 100],
        })
        report = assess_st_risk(df)
        assert report.total_score >= 4

    def test_healthy(self):
        df = pd.DataFrame({
            "net_income": [100, 200, 300],
            "revenue": [1e9, 2e9, 3e9],
            "net_assets": [1000, 1100, 1200],
        })
        report = assess_st_risk(df)
        assert report.risk_level == "低风险"
