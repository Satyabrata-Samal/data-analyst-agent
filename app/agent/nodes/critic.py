"""Critic node: evaluate insight quality and score the current analysis report."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from app.config.settings import settings
from app.prompts.few_shot_examples import CRITIC_EXAMPLES
from app.prompts.system_prompts import CRITIC_PROMPT
from app.schemas.output_schemas import CritiqueResult
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
    return (
        f"User question: {state['user_question']}\n\n"
        f"Dataset profile summary:\n{json.dumps(state['df_profile'], indent=2)}\n\n"
        f"Current insight report:\n{json.dumps(state.get('insight_report', {}), indent=2)}"
    )


def _few_shot_messages() -> list[HumanMessage | AIMessage]:
    return [
        HumanMessage(content=ex["content"])
        if ex["role"] == "user"
        else AIMessage(content=ex["content"])
        for ex in CRITIC_EXAMPLES
    ]


def _default_critique() -> CritiqueResult:
    return CritiqueResult(score=7.0, issues=[], suggestions=[])


def _parse_critic_response(raw_content: str) -> CritiqueResult:
    try:
        parsed = json.loads(_strip_markdown_fences(raw_content))
        issues = parsed.get("issues", [])
        suggestions = parsed.get("suggestions", [])
        if not isinstance(issues, list) or not isinstance(suggestions, list):
            raise TypeError("issues and suggestions must be lists")

        return CritiqueResult(
            score=float(parsed.get("score", 7.0)),
            issues=[str(issue) for issue in issues],
            suggestions=[str(suggestion) for suggestion in suggestions],
        )
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
        return _default_critique()


def run_critic(state: AgentState) -> dict[str, Any]:
    """Critique the current insight report and increment the critique iteration."""
    critique_iteration = state.get("critique_iteration", 0)

    entry_log = log_node_entry(
        logger,
        "critic",
        {"critique_iteration": critique_iteration},
    )

    user_message_content = _build_user_message_content(state)
    raw_response_content = ""
    new_iteration = critique_iteration + 1

    try:
        model = ChatAnthropic(
            model=settings.model_name,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.max_tokens,
        )
        messages = (
            [SystemMessage(content=CRITIC_PROMPT)]
            + _few_shot_messages()
            + list(state.get("messages", []))
            + [HumanMessage(content=user_message_content)]
        )
        response = model.invoke(messages)
        raw_response_content = str(response.content)
        critique = _parse_critic_response(raw_response_content)

        exit_log = log_node_exit(
            logger,
            "critic",
            {"score": critique.score, "critique_iteration": new_iteration},
        )

        return {
            "critique_score": critique.score,
            "critique_issues": critique.issues,
            "critique_iteration": new_iteration,
            "messages": [
                HumanMessage(content=user_message_content),
                AIMessage(content=raw_response_content),
            ],
            "agent_log": [entry_log, exit_log],
        }
    except Exception as exc:
        log_error(logger, "critic", exc)
        exit_log = log_node_exit(logger, "critic", {"error": True})
        critique = _default_critique()

        return {
            "critique_score": critique.score,
            "critique_issues": critique.issues,
            "critique_iteration": new_iteration,
            "messages": [],
            "agent_log": [entry_log, exit_log],
        }
