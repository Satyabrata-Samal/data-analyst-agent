"""Responder node: produce the final structured response and user-facing summary."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.prompts.system_prompts import RESPONDER_PROMPT
from app.schemas.output_schemas import FinalResponse, InsightReport, ValidationResult
from app.schemas.state_schema import AgentState
from app.utils.llm import get_chat_model
from app.utils.logger import get_logger, log_error, log_node_entry, log_node_exit

logger = get_logger(__name__)


def _build_insight_report(state: AgentState) -> InsightReport:
    insight_report_dict = state.get("insight_report", {})
    try:
        return InsightReport.model_validate(insight_report_dict)
    except ValidationError:
        return InsightReport(
            insights=[],
            data_coverage_note="Unavailable",
            limitations=["Could not reconstruct insight report."],
        )


def _build_validation_result(state: AgentState) -> ValidationResult:
    try:
        return ValidationResult.model_validate(state.get("validation_result", {}))
    except ValidationError:
        return ValidationResult(passed=True, was_sampled=False)


def _build_user_message_content(state: AgentState, insight_report_dict: dict[str, Any]) -> str:
    return (
        f"User question: {state['user_question']}\n\n"
        f"Insight report:\n{json.dumps(insight_report_dict, indent=2)}\n\n"
        f"Critique score: {state.get('critique_score', 'N/A')}\n"
        f"Critique issues resolved: {state.get('critique_issues', [])}"
    )


def _print_analysis_result(natural_language_response: str) -> None:
    print("\n" + "=" * 60)
    print("ANALYSIS RESULT")
    print("=" * 60)
    print(natural_language_response)
    print("=" * 60 + "\n")


def run_responder(state: AgentState) -> dict[str, Any]:
    """Generate the final response and print a natural-language summary for the CLI."""
    entry_log = log_node_entry(
        logger,
        "responder",
        {"critique_score": state.get("critique_score", 0)},
    )

    insight_report_dict = state.get("insight_report", {})
    user_message_content = _build_user_message_content(state, insight_report_dict)

    try:
        insight_report = _build_insight_report(state)
        validation_result = _build_validation_result(state)

        model = get_chat_model()
        response = model.invoke(
            [
                SystemMessage(content=RESPONDER_PROMPT),
                HumanMessage(content=user_message_content),
            ],
        )
        natural_language_response = str(response.content)

        _print_analysis_result(natural_language_response)

        final_response = FinalResponse(
            question=state["user_question"],
            report=insight_report,
            execution_trace_summary=state.get("agent_log", []),
            total_retries=state.get("retry_count", 0),
            critique_iterations=state.get("critique_iteration", 0),
            validation_result=validation_result,
        )

        exit_log = log_node_exit(logger, "responder", {"complete": True})

        return {
            "final_response": final_response.model_dump(),
            "agent_log": [entry_log, exit_log],
        }
    except Exception as exc:
        log_error(logger, "responder", exc)
        exit_log = log_node_exit(logger, "responder", {"error": True})

        return {
            "final_response": {
                "error": str(exc),
                "question": state["user_question"],
            },
            "agent_log": [entry_log, exit_log],
        }
