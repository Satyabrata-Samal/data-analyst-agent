"""Unit tests for the private parse helpers in each node.

These helpers convert raw LLM string output into typed objects. They are the
most likely source of runtime bugs (malformed JSON, markdown fences, missing
keys), so they get their own targeted tests.
"""

from __future__ import annotations

import pytest

from app.agent.nodes.clarifier import _parse_clarifier_response, _strip_markdown_fences
from app.agent.nodes.code_generator import _parse_code_generator_response
from app.agent.nodes.critic import _parse_critic_response
from app.agent.nodes.observer import _parse_observer_response
from app.agent.nodes.planner import _parse_planner_response
from app.agent.nodes.synthesizer import _parse_synthesizer_response


# --- _strip_markdown_fences (shared shape) ---


def test_strip_markdown_fences_plain():
    assert _strip_markdown_fences('{"a": 1}') == '{"a": 1}'


def test_strip_markdown_fences_json_block():
    raw = '```json\n{"a": 1}\n```'
    assert _strip_markdown_fences(raw) == '{"a": 1}'


def test_strip_markdown_fences_bare_block():
    raw = '```\n{"a": 1}\n```'
    assert _strip_markdown_fences(raw) == '{"a": 1}'


# --- clarifier ---


def test_parse_clarifier_response_ok():
    raw = '{"needs_clarification": true, "questions": ["q1", "q2"]}'
    needs, questions = _parse_clarifier_response(raw)
    assert needs is True
    assert questions == ["q1", "q2"]


def test_parse_clarifier_response_fenced():
    raw = '```json\n{"needs_clarification": false, "questions": []}\n```'
    needs, questions = _parse_clarifier_response(raw)
    assert needs is False
    assert questions == []


def test_parse_clarifier_response_malformed_returns_safe_default():
    needs, questions = _parse_clarifier_response("not json")
    assert needs is False
    assert questions == []


def test_parse_clarifier_response_wrong_questions_type():
    raw = '{"needs_clarification": true, "questions": "should be list"}'
    needs, questions = _parse_clarifier_response(raw)
    assert needs is False
    assert questions == []


# --- planner ---


def test_parse_planner_response_ok():
    raw = '{"reasoning": "because", "steps": ["s1", "s2", "s3"]}'
    reasoning, steps = _parse_planner_response(raw)
    assert reasoning == "because"
    assert steps == ["s1", "s2", "s3"]


def test_parse_planner_response_malformed_returns_raw_as_single_step():
    reasoning, steps = _parse_planner_response("not json")
    assert reasoning == "Parsing failed"
    assert steps == ["not json"]


def test_parse_planner_response_missing_steps():
    raw = '{"reasoning": "hi"}'
    reasoning, steps = _parse_planner_response(raw)
    # missing steps defaults to empty list; parser accepts it
    assert reasoning == "hi"
    assert steps == []


# --- code_generator ---


def test_parse_code_generator_response_ok():
    raw = (
        '{"code": "print(1)", "explanation": "prints one", '
        '"expected_output_description": "the number 1"}'
    )
    code, explanation, expected = _parse_code_generator_response(raw)
    assert code == "print(1)"
    assert explanation == "prints one"
    assert expected == "the number 1"


def test_parse_code_generator_response_malformed():
    code, explanation, expected = _parse_code_generator_response("not json")
    assert code == "not json"
    assert explanation == ""
    assert expected == ""


def test_parse_code_generator_response_fenced():
    raw = '```json\n{"code": "print(2)", "explanation": "", "expected_output_description": ""}\n```'
    code, _explanation, _expected = _parse_code_generator_response(raw)
    assert code == "print(2)"


# --- observer ---


def test_parse_observer_response_success():
    raw = '{"status": "success", "assessment": "ok", "fix_suggestion": null}'
    status, assessment, fix = _parse_observer_response(raw, execution_succeeded=True)
    assert status == "success"
    assert assessment == "ok"
    assert fix is None


def test_parse_observer_response_retry():
    raw = '{"status": "retry", "assessment": "bad column", "fix_suggestion": "use foo"}'
    status, assessment, fix = _parse_observer_response(raw, execution_succeeded=False)
    assert status == "retry"
    assert fix == "use foo"


def test_parse_observer_response_malformed_defaults_to_execution_state():
    status, _assessment, fix = _parse_observer_response("garbage", execution_succeeded=True)
    assert status == "success"
    assert fix is None

    status_bad, _assessment_bad, _fix = _parse_observer_response("garbage", execution_succeeded=False)
    assert status_bad == "retry"


# --- synthesizer ---


class _FakeState(dict):
    """Minimal AgentState substitute for parser tests."""


def test_parse_synthesizer_response_ok():
    raw = (
        '{"insights": [{"title": "t", "finding": "f", "confidence": "high", '
        '"chart_suggestion": null}], "data_coverage_note": "ok", '
        '"limitations": ["small n"]}'
    )
    report = _parse_synthesizer_response(raw, _FakeState())
    assert len(report.insights) == 1
    assert report.insights[0].confidence == "high"
    assert report.limitations == ["small n"]


def test_parse_synthesizer_response_malformed_returns_fallback():
    state = _FakeState(execution_result="raw stdout here")
    report = _parse_synthesizer_response("not json", state)
    assert len(report.insights) == 1
    assert report.insights[0].confidence == "low"
    assert "parsing" in report.data_coverage_note.lower()


def test_parse_synthesizer_response_bad_insight_shape_uses_fallback():
    raw = '{"insights": ["not-an-object"], "data_coverage_note": "", "limitations": []}'
    report = _parse_synthesizer_response(raw, _FakeState())
    # falls back to a single "low" confidence insight
    assert report.insights[0].confidence == "low"


# --- critic ---


def test_parse_critic_response_ok():
    raw = '{"score": 8.5, "issues": ["one"], "suggestions": ["fix"]}'
    critique = _parse_critic_response(raw)
    assert critique.score == 8.5
    assert critique.issues == ["one"]
    assert critique.suggestions == ["fix"]


def test_parse_critic_response_malformed_returns_default_7():
    critique = _parse_critic_response("not json")
    assert critique.score == 7.0
    assert critique.issues == []


def test_parse_critic_response_wrong_types():
    raw = '{"score": 5.0, "issues": "should be list", "suggestions": []}'
    critique = _parse_critic_response(raw)
    # non-list issues coerces to default critique
    assert critique.score == 7.0


def test_parse_critic_response_score_out_of_range_falls_back():
    raw = '{"score": 99.0, "issues": [], "suggestions": []}'
    critique = _parse_critic_response(raw)
    # Pydantic bound violation → default critique
    assert critique.score == 7.0
