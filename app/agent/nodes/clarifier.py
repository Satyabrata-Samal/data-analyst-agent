"""Clarifier node: decide whether user clarification is needed before analysis."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config.settings import settings
from app.prompts.system_prompts import CLARIFIER_PROMPT
from app.schemas.state_schema import AgentState
from app.utils.logger import get_logger, log_error, log_node_entry, log_node_exit

logger = get_logger(__name__)


def _get_last_ai_message_content(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)
    return ""


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _parse_clarifier_response(raw_content: str) -> tuple[bool, list[str]]:
    try:
        parsed = json.loads(_strip_markdown_fences(raw_content))
        needs_clarification = bool(parsed.get("needs_clarification", False))
        questions = parsed.get("questions", [])
        if not isinstance(questions, list):
            return False, []
        return needs_clarification, [str(question) for question in questions]
    except (json.JSONDecodeError, TypeError, ValueError):
        return False, []


def run_clarifier(state: AgentState) -> dict[str, Any]:
    """Ask clarifying questions when the user request is ambiguous."""
    entry_log = log_node_entry(
        logger,
        "clarifier",
        {"question": state["user_question"]},
    )

    profile_summary = _get_last_ai_message_content(state.get("messages", []))
    new_human_msg = HumanMessage(
        content=(
            f"User question: {state['user_question']}\n\n"
            f"Dataset summary:\n{profile_summary}"
        ),
    )

    raw_response_content = ""
    needs_clarification = False
    questions: list[str] = []
    answers: list[str] = []

    try:
        model = ChatAnthropic(
            model=settings.model_name,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.max_tokens,
        )
        response = model.invoke(
            [SystemMessage(content=CLARIFIER_PROMPT)]
            + list(state.get("messages", []))
            + [new_human_msg],
        )
        raw_response_content = str(response.content)
        needs_clarification, questions = _parse_clarifier_response(raw_response_content)
    except Exception as exc:
        log_error(logger, "clarifier", exc)
        needs_clarification = False
        questions = []
        answers = []

    if needs_clarification and questions:
        for question in questions:
            answer = input(f"\nClarification needed:\n Q: {question}\n A: ")
            answers.append(answer)
    else:
        questions = []
        answers = []

    new_messages: list[BaseMessage] = [
        HumanMessage(content=f"User question: {state['user_question']}"),
        AIMessage(content=raw_response_content),
    ]

    if needs_clarification and questions and answers:
        new_messages.append(
            HumanMessage(
                content=f"Clarification answers: {json.dumps(dict(zip(questions, answers)))}",
            ),
        )

    exit_log = log_node_exit(
        logger,
        "clarifier",
        {
            "needs_clarification": needs_clarification,
            "questions": len(questions),
        },
    )

    return {
        "clarifying_questions": questions,
        "clarifying_answers": answers,
        "clarification_done": True,
        "messages": new_messages,
        "agent_log": [entry_log, exit_log],
    }
