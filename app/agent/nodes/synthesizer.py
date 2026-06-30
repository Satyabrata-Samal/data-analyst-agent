"""Synthesizer node: convert raw execution output into structured insights."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from app.config.settings import settings
from app.prompts.system_prompts import SYNTHESIZER_PROMPT
from app.schemas.output_schemas import Insight, InsightReport
from app.schemas.state_schema import AgentState
from app.utils.logger import get_logger, log_error, log_node_entry, log_node_exit

logger = get_logger(__name__)


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _build_user_message_content(state: AgentState) -> str:
    parts = [
        f"User question: {state['user_question']}",
        (
            "Raw analysis output:\n"
            f"{state.get('execution_result') or 'No output was produced.'}"
        ),
    ]

    critique_iteration = state.get("critique_iteration", 0)
    if critique_iteration > 0:
        critique_issues = state.get("critique_issues", [])
        parts.append(
            "Previous critique issues:\n" + "\n".join(critique_issues),
        )

    return "\n\n".join(parts)


def _build_fallback_report(state: AgentState) -> InsightReport:
    return InsightReport(
        insights=[
            Insight(
                title="Analysis Complete",
                finding=state.get("execution_result", "No output produced."),
                confidence="low",
                chart_suggestion=None,
            ),
        ],
        data_coverage_note="Raw output returned due to parsing error.",
        limitations=["Structured parsing failed — review raw output."],
    )


def _parse_synthesizer_response(raw_content: str, state: AgentState) -> InsightReport:
    try:
        parsed = json.loads(_strip_markdown_fences(raw_content))
        insights_raw = parsed.get("insights", [])
        if not isinstance(insights_raw, list):
            raise TypeError("insights must be a list")

        insights = [
            Insight(
                title=str(item.get("title", "")),
                finding=str(item.get("finding", "")),
                confidence=item.get("confidence", "low"),
                chart_suggestion=item.get("chart_suggestion"),
            )
            for item in insights_raw
            if isinstance(item, dict)
        ]
        if len(insights) != len(insights_raw):
            raise TypeError("each insight must be a dict")

        limitations = parsed.get("limitations", [])
        if not isinstance(limitations, list):
            raise TypeError("limitations must be a list")

        return InsightReport(
            insights=insights,
            data_coverage_note=str(parsed.get("data_coverage_note", "")),
            limitations=[str(limitation) for limitation in limitations],
        )
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
        return _build_fallback_report(state)


def _build_return(
    report: InsightReport,
    user_message_content: str,
    raw_response_content: str,
    entry_log: str,
    exit_log: str,
) -> dict[str, Any]:
    return {
        "insights": [insight.model_dump() for insight in report.insights],
        "insight_report": report.model_dump(),
        "messages": [
            HumanMessage(content=user_message_content),
            AIMessage(content=raw_response_content),
        ],
        "agent_log": [entry_log, exit_log],
    }


def run_synthesizer(state: AgentState) -> dict[str, Any]:
    """Synthesize structured insights from raw analysis execution output."""
    entry_log = log_node_entry(
        logger,
        "synthesizer",
        {"critique_iteration": state.get("critique_iteration", 0)},
    )

    user_message_content = _build_user_message_content(state)
    raw_response_content = ""

    try:
        model = ChatAnthropic(
            model=settings.model_name,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.max_tokens,
        )
        response = model.invoke(
            [SystemMessage(content=SYNTHESIZER_PROMPT)]
            + list(state.get("messages", []))
            + [HumanMessage(content=user_message_content)],
        )
        raw_response_content = str(response.content)
        report = _parse_synthesizer_response(raw_response_content, state)

        exit_log = log_node_exit(
            logger,
            "synthesizer",
            {"insights": len(report.insights)},
        )

        return _build_return(
            report,
            user_message_content,
            raw_response_content,
            entry_log,
            exit_log,
        )
    except Exception as exc:
        log_error(logger, "synthesizer", exc)
        exit_log = log_node_exit(logger, "synthesizer", {"error": True})
        report = _build_fallback_report(state)

        return {
            "insights": [insight.model_dump() for insight in report.insights],
            "insight_report": report.model_dump(),
            "messages": [],
            "agent_log": [entry_log, exit_log],
        }
