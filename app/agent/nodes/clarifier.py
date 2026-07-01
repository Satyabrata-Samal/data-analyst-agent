"""Clarifier node: decide whether user clarification is needed before analysis."""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic
from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts.system_prompts import CLARIFIER_PROMPT
from app.schemas.state_schema import AgentState
from app.utils.llm import get_chat_model
from app.utils.logger import (
    get_logger,
    log_error,
    log_llm_call,
    log_node_entry,
    log_node_exit,
)

logger = get_logger(__name__)


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
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to parse clarifier response: %s | raw_content=%r",
            exc,
            raw_content[:500],
        )
        return False, []


def run_clarifier(state: AgentState) -> dict[str, Any]:
    """Ask clarifying questions when the user request is ambiguous."""
    entry_log = log_node_entry(
        logger,
        "clarifier",
        {"question": state["user_question"]},
    )

    profile_summary = state.get("profile_summary", "")
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
        model = get_chat_model()
        response = model.invoke(
            [SystemMessage(content=CLARIFIER_PROMPT), new_human_msg],
        )
        raw_response_content = str(response.content)
        log_llm_call(
            logger,
            "clarifier",
            CLARIFIER_PROMPT,
            new_human_msg.content,
            raw_response_content,
        )
        needs_clarification, questions = _parse_clarifier_response(raw_response_content)
    except anthropic.AuthenticationError:
        raise
    except Exception as exc:
        log_error(logger, "clarifier", exc)
        needs_clarification = False
        questions = []
        answers = []

    if needs_clarification and questions and not state.get("web_mode", False):
        # CLI mode: collect answers interactively. In web mode we never block on
        # stdin — the UI surfaces the questions and lets the user fold any extra
        # context into the question up front instead.
        for question in questions:
            answer = input(f"\nClarification needed:\n Q: {question}\n A: ")
            answers.append(answer)
    elif not (needs_clarification and questions):
        questions = []
        answers = []

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
        "agent_log": [entry_log, exit_log],
    }
