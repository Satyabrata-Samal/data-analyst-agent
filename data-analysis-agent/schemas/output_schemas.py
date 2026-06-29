"""Pydantic output schemas for structured LLM responses and final agent output.

These models validate and serialize structured outputs from individual graph nodes
(planner, code generator, synthesizer, critic, validator, responder). They are
separate from AgentState in state_schema.py, which tracks runtime graph state;
these schemas define the shape of parsed LLM JSON and the final user-facing response.
"""

from typing import Literal

from pydantic import BaseModel, Field


class AnalysisPlan(BaseModel):
    steps: list[str] = Field(
        description="Ordered list of analysis steps to execute against the dataset.",
    )
    reasoning: str = Field(
        description="Explanation of why these steps answer the user's question.",
    )


class GeneratedCode(BaseModel):
    code: str = Field(
        description="Complete, executable Python script for the analysis.",
    )
    explanation: str = Field(
        description="Plain-English summary of what the generated code does.",
    )
    expected_output_description: str = Field(
        description="Description of what the stdout output should look like when the code runs successfully.",
    )


class Insight(BaseModel):
    title: str = Field(
        description="Short headline for a single finding.",
    )
    finding: str = Field(
        description="The actual insight stated in plain English.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in this finding based on data quality and analysis rigor.",
    )
    chart_suggestion: str | None = Field(
        default=None,
        description='Optional visualization recommendation, e.g. "bar chart of region vs revenue".',
    )


class InsightReport(BaseModel):
    insights: list[Insight] = Field(
        description="Collection of structured findings from the analysis.",
    )
    data_coverage_note: str = Field(
        description='Note on data scope, e.g. "Analysis ran on 50K sampled rows out of 2M".',
    )
    limitations: list[str] = Field(
        description="Honest caveats and limitations of the analysis.",
    )


class CritiqueResult(BaseModel):
    score: float = Field(
        ge=0.0,
        le=10.0,
        description="Quality score from 0 to 10 for the current analysis output.",
    )
    issues: list[str] = Field(
        description="Specific problems identified in the analysis or execution.",
    )
    suggestions: list[str] = Field(
        description="Concrete improvements that should be made before approval.",
    )
    approved: bool = Field(
        description="True if the analysis is approved (score >= 7.0).",
    )


class ValidationResult(BaseModel):
    passed: bool = Field(
        description="Whether the input CSV and question passed validation.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message when validation fails; None when passed is True.",
    )
    row_count: int | None = Field(
        default=None,
        description="Number of rows in the dataset, if validation passed.",
    )
    column_count: int | None = Field(
        default=None,
        description="Number of columns in the dataset, if validation passed.",
    )
    file_size_mb: float | None = Field(
        default=None,
        description="File size of the CSV in megabytes, if validation passed.",
    )
    was_sampled: bool = Field(
        description="Whether the dataset was sampled due to size constraints.",
    )
    sample_size: int | None = Field(
        default=None,
        description="Number of rows used after sampling; None if the full dataset was used.",
    )


class FinalResponse(BaseModel):
    question: str = Field(
        description="The original user question that was analyzed.",
    )
    report: InsightReport = Field(
        description="Structured insight report with findings, coverage, and limitations.",
    )
    execution_trace_summary: list[str] = Field(
        description="High-level summary of steps taken during agent execution.",
    )
    total_retries: int = Field(
        description="Total number of code generation retries attempted.",
    )
    critique_iterations: int = Field(
        description="Number of critique loops performed before final response.",
    )
