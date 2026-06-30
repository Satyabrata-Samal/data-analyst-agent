"""Planner node: produce a step-by-step analysis plan from profile and question."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config.settings import settings
from app.prompts.system_prompts import PLANNER_PROMPT
from app.schemas.output_schemas import AnalysisPlan
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
        f"Dataset profile:\n{json.dumps(state['df_profile'], indent=2)}",
    ]

    questions = state.get("clarifying_questions") or []
    answers = state.get("clarifying_answers") or []
    if questions and answers:
        qa_pairs = "\n".join(
            f"Q: {question}\nA: {answer}"
            for question, answer in zip(questions, answers)
        )
        parts.append(f"Clarification:\n{qa_pairs}")

    return "\n\n".join(parts)


def _parse_planner_response(raw_content: str) -> tuple[str, list[str]]:
    try:
        parsed = json.loads(_strip_markdown_fences(raw_content))
        reasoning = str(parsed.get("reasoning", ""))
        steps = parsed.get("steps", [])
        if not isinstance(steps, list):
            raise TypeError("steps must be a list")
        return reasoning, [str(step) for step in steps]
    except (json.JSONDecodeError, TypeError, ValueError):
        return "Parsing failed", [raw_content]


def run_planner(state: AgentState) -> dict[str, Any]:
    """Generate an executable analysis plan for the user's question."""
    entry_log = log_node_entry(
        logger,
        "planner",
        {"question": state["user_question"]},
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
            [SystemMessage(content=PLANNER_PROMPT)]
            + list(state.get("messages", []))
            + [HumanMessage(content=user_message_content)],
        )
        raw_response_content = str(response.content)

        reasoning, steps = _parse_planner_response(raw_response_content)
        plan = AnalysisPlan(steps=steps, reasoning=reasoning)

        exit_log = log_node_exit(
            logger,
            "planner",
            {"steps": len(plan.steps)},
        )

        return {
            "analysis_plan": plan.steps,
            "messages": [
                HumanMessage(content=user_message_content),
                AIMessage(content=raw_response_content),
            ],
            "agent_log": [entry_log, exit_log],
        }
    except Exception as exc:
        log_error(logger, "planner", exc)
        exit_log = log_node_exit(logger, "planner", {"error": True})

        return {
            "analysis_plan": [
                "Unable to generate plan due to error. Proceeding with general analysis.",
            ],
            "messages": [],
            "agent_log": [entry_log, exit_log],
        }
