"""Honesty and parsing tests for the LLM research agent.

Covers audit items R16 (silent no-key downgrade), R17 (dict-wrapped / fenced
JSON) and R18 (contract validation of LLM fields), plus bounded LLM input.
"""

import json
import logging

import pytest

from quant_platform.agent.research_agent import ResearchAgent

VALID_SIGNAL = {
    "signal_type": "event",
    "direction": "bullish",
    "strength": 0.4,
    "confidence": 0.8,
    "reasoning": "业绩增长",
    "factor_implications": {"roe": 0.3},
}


def _agent(tmp_path):
    """LLM-mode agent with a fake key (no network call happens at init)."""
    return ResearchAgent(mode="llm", api_key="sk-test", cache_dir=str(tmp_path))


# ----------------------------------------------------------------------
# R16: visible downgrade without a key
# ----------------------------------------------------------------------

class TestBackendVisibility:
    def test_llm_without_key_is_visibly_downgraded(self, tmp_path, monkeypatch, caplog):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        caplog.set_level(logging.WARNING)

        agent = ResearchAgent(mode="llm", cache_dir=str(tmp_path))

        assert agent.requested_mode == "llm"
        assert agent.effective_mode == "keyword"
        assert agent.mode == "keyword"
        assert agent.is_llm is False
        assert agent.backend == "keyword"
        assert agent.mode_downgraded is True
        assert "DEEPSEEK_API_KEY" in (agent.downgrade_reason or "")

        status = agent.status()
        assert status["requested_mode"] == "llm"
        assert status["effective_mode"] == "keyword"
        assert status["backend"] == "keyword"
        assert status["mode_downgraded"] is True

        assert any(
            "downgraded" in record.getMessage().lower()
            for record in caplog.records
        )

    def test_llm_with_key_reports_llm_backend(self, tmp_path):
        agent = _agent(tmp_path)

        assert agent.mode == "llm"
        assert agent.mode_downgraded is False
        assert agent.backend == "llm:deepseek-chat"
        assert agent.status()["effective_mode"] == "llm"
        assert agent.status()["mode_downgraded"] is False

    def test_invalid_mode_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="mode"):
            ResearchAgent(mode="magic", cache_dir=str(tmp_path))


# ----------------------------------------------------------------------
# R17: dict wrappers, fenced JSON, prose around JSON
# ----------------------------------------------------------------------

class TestLLMResponseParsing:
    def test_dict_wrapper_signals(self, tmp_path):
        agent = _agent(tmp_path)
        agent._call_llm = lambda prompt: json.dumps({"signals": [VALID_SIGNAL]})

        signals = agent.extract_signals_from_text("公司发布重大合同公告", stock="600519")

        assert len(signals) == 1
        assert signals[0].stock == "600519"
        assert signals[0].direction == "bullish"
        assert agent.last_fallback_reason is None

    def test_fenced_json_with_prose(self, tmp_path):
        agent = _agent(tmp_path)
        payload = (
            "Here is the result:\n```json\n"
            + json.dumps([VALID_SIGNAL], ensure_ascii=False)
            + "\n```\nDone."
        )
        agent._call_llm = lambda prompt: payload

        signals = agent.extract_signals_from_text("公司公告", stock="000001")

        assert len(signals) == 1
        assert signals[0].strength == pytest.approx(0.4)
        assert agent.last_fallback_reason is None

    def test_bare_single_object(self, tmp_path):
        agent = _agent(tmp_path)
        agent._call_llm = lambda prompt: json.dumps(VALID_SIGNAL)

        signals = agent.extract_signals_from_text("公司公告", stock="A")

        assert len(signals) == 1

    def test_dict_wrapper_single_object(self, tmp_path):
        agent = _agent(tmp_path)
        agent._call_llm = lambda prompt: json.dumps({"signals": VALID_SIGNAL})

        signals = agent.extract_signals_from_text("公司公告", stock="A")

        assert len(signals) == 1

    def test_dict_wrapper_hypotheses(self, tmp_path):
        item = {
            "factor_name": "momentum_1m",
            "description": "动量效应",
            "expected_ic": 0.04,
            "economic_logic": "资金流入推动趋势延续",
            "data_requirements": ["日频行情"],
            "risk_factors": ["主题持续性不确定"],
            "confidence": 0.6,
        }
        agent = _agent(tmp_path)
        agent._call_llm = lambda prompt: json.dumps({"hypotheses": [item]})

        hypotheses = agent.generate_hypotheses(
            "资金流入明显", available_factors=["momentum_1m"],
        )

        assert len(hypotheses) == 1
        assert hypotheses[0].factor_name == "momentum_1m"
        assert hypotheses[0].expected_ic == pytest.approx(0.04)


# ----------------------------------------------------------------------
# R18: contract validation of LLM fields
# ----------------------------------------------------------------------

class TestContractValidation:
    def test_invalid_signal_entries_are_dropped_or_repaired(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING)
        items = [
            # Unknown Literal for signal_type: cannot be repaired -> dropped.
            {"signal_type": "rumor", "direction": "up",
             "strength": 0.5, "confidence": 0.5},
            "not a dict",
            # direction/strength/confidence/reasoning/implications out of contract.
            {
                "signal_type": "sentiment",
                "direction": "up",
                "strength": 5,
                "confidence": 2,
                "reasoning": 1.5,
                "factor_implications": {"roe": 9, "bad": "x"},
            },
        ]
        agent = _agent(tmp_path)
        agent._call_llm = lambda prompt: json.dumps(items)

        signals = agent.extract_signals_from_text("文本", stock="A")

        assert len(signals) == 1
        signal = signals[0]
        assert signal.direction == "bullish"
        assert signal.strength == pytest.approx(1.0)
        assert signal.confidence == pytest.approx(1.0)
        assert signal.reasoning == "1.5"
        assert signal.factor_implications == {"roe": pytest.approx(1.0)}

        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "dropped" in messages
        assert "repaired" in messages

    def test_invalid_hypothesis_fields_are_repaired(self, tmp_path):
        item = {
            "factor_name": "momentum_1m",
            "description": 5,
            "expected_ic": 99,
            "economic_logic": None,
            "data_requirements": "日频行情",
            "risk_factors": None,
            "confidence": -1,
        }
        agent = _agent(tmp_path)
        agent._call_llm = lambda prompt: json.dumps([item])

        hypotheses = agent.generate_hypotheses(
            "动量", available_factors=["momentum_1m"],
        )

        assert len(hypotheses) == 1
        hypothesis = hypotheses[0]
        assert hypothesis.expected_ic == pytest.approx(1.0)
        assert hypothesis.confidence == pytest.approx(0.0)
        assert hypothesis.description == "5"
        assert hypothesis.economic_logic == ""
        assert hypothesis.data_requirements == ["日频行情"]
        assert hypothesis.risk_factors == []

    def test_non_finite_hypothesis_is_dropped(self, tmp_path):
        item = {
            "factor_name": "momentum_1m",
            "description": "bad",
            "expected_ic": float("nan"),
            "economic_logic": "",
            "data_requirements": [],
            "risk_factors": [],
            "confidence": 0.5,
        }
        agent = _agent(tmp_path)
        agent._call_llm = lambda prompt: json.dumps([item])

        hypotheses = agent.generate_hypotheses("动量")

        assert hypotheses == []

    def test_missing_required_signal_fields_are_dropped(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING)
        agent = _agent(tmp_path)
        agent._call_llm = lambda prompt: json.dumps([{}])

        signals = agent.extract_signals_from_text("文本", stock="A")

        assert signals == []
        assert any(
            "missing required field" in record.getMessage()
            for record in caplog.records
        )

    def test_missing_required_hypothesis_fields_are_dropped(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING)
        agent = _agent(tmp_path)
        agent._call_llm = lambda prompt: json.dumps([{"factor_name": "x"}])

        hypotheses = agent.generate_hypotheses("叙事")

        assert hypotheses == []
        assert any(
            "missing required field" in record.getMessage()
            for record in caplog.records
        )


# ----------------------------------------------------------------------
# Fallback visibility
# ----------------------------------------------------------------------

class TestFallbackVisibility:
    def test_invalid_json_falls_back_and_records_reason(self, tmp_path):
        agent = _agent(tmp_path)
        agent._call_llm = lambda prompt: "I cannot comply with that."

        signals = agent.extract_signals_from_text("公司净利润同比增长30%", stock="A")

        assert signals, "keyword fallback should still extract signals"
        assert agent.last_fallback_reason is not None
        assert "llm_signal_extraction" in agent.last_fallback_reason
        assert agent.status()["last_fallback_reason"] == agent.last_fallback_reason

    def test_attribution_failure_is_recorded(self, tmp_path):
        agent = _agent(tmp_path)

        def boom(prompt):
            raise RuntimeError("api down")

        agent._call_llm = boom

        summary = agent.summarize_attribution({"momentum": 0.01})

        assert "momentum" in summary
        assert agent.last_fallback_reason is not None
        assert "llm_attribution_summary" in agent.last_fallback_reason

    def test_empty_llm_summary_falls_back(self, tmp_path):
        agent = _agent(tmp_path)
        agent._call_llm = lambda prompt: "   "

        summary = agent.summarize_attribution({"momentum": 0.01})

        assert "momentum" in summary
        assert agent.last_fallback_reason is not None
        assert "empty LLM attribution response" in agent.last_fallback_reason

    def test_partial_chunk_fallback_is_not_erased(self, tmp_path):
        agent = _agent(tmp_path)
        calls = {"count": 0}

        def flaky(prompt):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("first chunk failed")
            return json.dumps([VALID_SIGNAL])

        agent._call_llm = flaky
        # ~2,500 chars of sentence-delimited text -> two chunks.
        text = "公司公告。" * 500

        signals = agent.extract_signals_from_text(text, stock="A")

        assert calls["count"] == 2
        assert len(signals) == 1, "the successful chunk should still produce a signal"
        assert agent.last_fallback_reason is not None
        assert "llm_signal_extraction" in agent.last_fallback_reason
        assert agent.status()["last_fallback_reason"] == agent.last_fallback_reason

    def test_call_llm_refuses_without_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        agent = ResearchAgent(mode="keyword", cache_dir=str(tmp_path))

        with pytest.raises(RuntimeError, match="no DEEPSEEK_API_KEY"):
            agent._call_llm("prompt")


# ----------------------------------------------------------------------
# Bounded LLM input
# ----------------------------------------------------------------------

class TestBoundedPrompts:
    def test_gigantic_narrative_is_truncated(self, tmp_path):
        agent = _agent(tmp_path)
        captured = {}

        def fake(prompt):
            captured["prompt"] = prompt
            return "[]"

        agent._call_llm = fake

        agent.generate_hypotheses("X" * 100_000)

        assert len(captured["prompt"]) < 5000

    def test_stock_and_source_type_are_bounded(self, tmp_path):
        agent = _agent(tmp_path)
        captured = {}

        def fake(prompt):
            captured["prompt"] = prompt
            return "[]"

        agent._call_llm = fake

        agent.extract_signals_from_text(
            "text", stock="S" * 100_000, source_type="T" * 100_000,
        )

        assert len(captured["prompt"]) < 5000

    def test_factor_names_and_period_are_bounded(self, tmp_path):
        agent = _agent(tmp_path)
        captured = []

        def fake(prompt):
            captured.append(prompt)
            return "[]"

        agent._call_llm = fake
        agent.generate_hypotheses("叙事", available_factors=["F" * 100_000] * 3)
        assert len(captured[-1]) < 10_000

        captured.clear()
        agent.summarize_attribution(
            {"K" * 100_000: 0.01}, period="P" * 100_000,
        )
        assert len(captured[-1]) < 10_000

    def test_overlong_sentence_is_hard_split(self):
        text = "X" * 5000

        chunks = ResearchAgent._chunk_text(text, max_chars=1500)

        assert all(len(chunk) <= 1500 for chunk in chunks)
        assert "".join(chunks) == text

    def test_extreme_input_is_chunk_capped(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING)
        agent = _agent(tmp_path)
        calls = []

        def fake(prompt):
            calls.append(prompt)
            return "[]"

        agent._call_llm = fake

        agent.extract_signals_from_text("公告。" * 60_000)

        assert len(calls) <= agent.MAX_CHUNKS
        assert agent.last_input_truncated is True
        assert agent.status()["input_truncated"] is True

    def test_call_llm_rejects_oversized_prompt(self, tmp_path):
        agent = _agent(tmp_path)

        with pytest.raises(ValueError, match="too long"):
            agent._call_llm("X" * (agent.MAX_PROMPT_CHARS + 1))

    def test_huge_attribution_is_truncated(self, tmp_path):
        agent = _agent(tmp_path)
        captured = {}

        def fake(prompt):
            captured["prompt"] = prompt
            return "summary"

        agent._call_llm = fake
        contributions = {f"f{i}": 0.0001 * i for i in range(1000)}

        result = agent.summarize_attribution(contributions)

        assert result == "summary"
        assert len(captured["prompt"]) < 5000
