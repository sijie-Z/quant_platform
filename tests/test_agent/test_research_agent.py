"""Tests for LLM Research Agent."""

import numpy as np
import pandas as pd
import pytest

from quant_platform.agent.research_agent import (
    ExtractedSignal,
    FactorHypothesis,
    ResearchAgent,
)


@pytest.fixture
def agent():
    return ResearchAgent(mode="keyword")


class TestResearchAgent:
    def test_init_default(self):
        a = ResearchAgent()
        assert a.mode == "keyword"

    def test_init_llm_no_key_fallback(self):
        a = ResearchAgent(mode="llm", api_key=None)
        assert a.mode == "keyword"  # Falls back

    def test_extract_signals_bullish(self, agent):
        text = "茅台Q3净利润同比增长30%，营收超预期增长25%，获得机构上调目标价至2000元"
        signals = agent.extract_signals_from_text(text, stock="600519")
        assert len(signals) > 0
        # Should find bullish signals
        bullish = [s for s in signals if s.direction == "bullish"]
        assert len(bullish) > 0

    def test_extract_signals_bearish(self, agent):
        text = "某公司净利润同比下降40%，大股东减持500万股，收到监管问询函"
        signals = agent.extract_signals_from_text(text, stock="000001")
        assert len(signals) > 0
        bearish = [s for s in signals if s.direction == "bearish"]
        assert len(bearish) > 0

    def test_extract_signals_neutral(self, agent):
        text = "公司召开股东大会，审议年度报告"
        signals = agent.extract_signals_from_text(text, stock="000002")
        # Should be neutral or minimal signals
        for s in signals:
            assert s.signal_type in ("sentiment", "event")

    def test_extract_signals_structure(self, agent):
        text = "公司净利润同比增长50%，业绩大幅超预期"
        signals = agent.extract_signals_from_text(text, stock="TEST")
        if signals:
            s = signals[0]
            assert isinstance(s, ExtractedSignal)
            assert s.stock == "TEST"
            assert -1 <= s.strength <= 1
            assert 0 <= s.confidence <= 1

    def test_generate_hypotheses(self, agent):
        narrative = "新能源板块资金持续流入，光伏装机量超预期增长"
        hypotheses = agent.generate_hypotheses(
            narrative,
            available_factors=["momentum_1m", "turnover_20d", "volatility_60d"],
        )
        assert isinstance(hypotheses, list)
        for h in hypotheses:
            assert isinstance(h, FactorHypothesis)
            assert h.factor_name in ["momentum_1m", "turnover_20d", "volatility_60d"]
            assert 0 < h.expected_ic < 0.1

    def test_generate_hypotheses_no_factors(self, agent):
        hypotheses = agent.generate_hypotheses("业绩增长超预期")
        assert isinstance(hypotheses, list)

    def test_summarize_attribution(self, agent):
        contributions = {
            "momentum_1m": 0.003,
            "value": -0.001,
            "size": 0.002,
            "volatility": -0.0005,
        }
        returns = pd.Series(np.random.randn(60) * 0.01)
        summary = agent.summarize_attribution(contributions, returns, period="本月")
        assert isinstance(summary, str)
        assert "本月" in summary
        assert "bps" in summary

    def test_summarize_attribution_no_returns(self, agent):
        contributions = {"momentum": 0.005, "value": -0.002}
        summary = agent.summarize_attribution(contributions)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_describe_risk(self, agent):
        risk = {
            "total_risk": 0.15,
            "factor_risk": 0.12,
            "specific_risk": 0.08,
            "r_squared": 0.64,
            "factor_exposures": {"momentum": 0.3, "value": -0.1},
            "factor_contributions": {"momentum": 0.005, "value": -0.001},
        }
        report = agent.describe_risk(risk, regime="neutral")
        assert isinstance(report, str)
        assert "15.0%" in report
        assert "风险" in report

    def test_chunk_text(self):
        text = "A。B。C。D。E。" * 100
        chunks = ResearchAgent._chunk_text(text, max_chars=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 60  # Some tolerance

    def test_chunk_short_text(self):
        text = "Short text"
        chunks = ResearchAgent._chunk_text(text, max_chars=2000)
        assert chunks == [text]

    def test_keyword_sentiment_bullish(self):
        score = ResearchAgent._keyword_sentiment("公司业绩增长超预期，利好不断")
        assert score > 0

    def test_keyword_sentiment_bearish(self):
        score = ResearchAgent._keyword_sentiment("公司亏损下降，减持违约风险")
        assert score < 0

    def test_keyword_sentiment_neutral(self):
        score = ResearchAgent._keyword_sentiment("今天天气不错")
        assert score == 0

    def test_deduplicate_signals(self):
        signals = [
            ExtractedSignal(stock="A", signal_type="sentiment", direction="bullish",
                          strength=0.5, confidence=0.3, reasoning="", source_text=""),
            ExtractedSignal(stock="A", signal_type="sentiment", direction="bullish",
                          strength=0.8, confidence=0.7, reasoning="", source_text=""),
        ]
        deduped = ResearchAgent._deduplicate_signals(signals)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.7

    def test_infer_factor_implications(self):
        impl = ResearchAgent._infer_factor_implications("profit_growth", 30)
        assert "roe" in impl
        assert impl["roe"] > 0

    def test_extract_signals_long_text(self, agent):
        # Test chunking with long text
        text = "公司净利润同比增长20%。" * 200
        signals = agent.extract_signals_from_text(text, stock="LONG")
        assert isinstance(signals, list)
