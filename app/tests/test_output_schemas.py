"""Unit tests for Pydantic output schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.output_schemas import (
    AnalysisPlan,
    CritiqueResult,
    FinalResponse,
    GeneratedCode,
    Insight,
    InsightReport,
    ValidationResult,
)


def test_analysis_plan_ok():
    plan = AnalysisPlan(steps=["s1", "s2"], reasoning="because")
    assert plan.steps == ["s1", "s2"]
    assert plan.reasoning == "because"


def test_generated_code_requires_all_fields():
    with pytest.raises(ValidationError):
        GeneratedCode(code="print(1)")  # missing explanation + expected_output


def test_insight_confidence_literal():
    Insight(title="t", finding="f", confidence="high")
    Insight(title="t", finding="f", confidence="medium")
    Insight(title="t", finding="f", confidence="low")

    with pytest.raises(ValidationError):
        Insight(title="t", finding="f", confidence="very-high")


def test_insight_chart_suggestion_optional():
    i = Insight(title="t", finding="f", confidence="high")
    assert i.chart_suggestion is None


def test_insight_report_roundtrip():
    report = InsightReport(
        insights=[Insight(title="t", finding="f", confidence="high")],
        data_coverage_note="full",
        limitations=["small n"],
    )
    dumped = report.model_dump()
    revived = InsightReport.model_validate(dumped)
    assert revived.insights[0].title == "t"


def test_critique_result_score_bounds():
    CritiqueResult(score=0.0, issues=[], suggestions=[])
    CritiqueResult(score=10.0, issues=[], suggestions=[])

    with pytest.raises(ValidationError):
        CritiqueResult(score=-0.1, issues=[], suggestions=[])
    with pytest.raises(ValidationError):
        CritiqueResult(score=10.5, issues=[], suggestions=[])


def test_validation_result_pass():
    v = ValidationResult(
        passed=True,
        row_count=10,
        column_count=3,
        file_size_mb=0.1,
        was_sampled=False,
    )
    assert v.passed is True
    assert v.error_message is None


def test_validation_result_fail():
    v = ValidationResult(
        passed=False,
        error_message="bad",
        was_sampled=False,
    )
    assert v.error_message == "bad"
    assert v.row_count is None


def test_final_response_composes_report_and_validation():
    report = InsightReport(
        insights=[Insight(title="t", finding="f", confidence="high")],
        data_coverage_note="ok",
        limitations=[],
    )
    v = ValidationResult(passed=True, row_count=1, column_count=1, was_sampled=False)

    fr = FinalResponse(
        question="q",
        report=report,
        validation_result=v,
        execution_trace_summary=["a", "b"],
        total_retries=0,
        critique_iterations=1,
    )

    dumped = fr.model_dump()
    assert dumped["question"] == "q"
    assert dumped["total_retries"] == 0
    assert dumped["critique_iterations"] == 1
    assert dumped["report"]["insights"][0]["finding"] == "f"
