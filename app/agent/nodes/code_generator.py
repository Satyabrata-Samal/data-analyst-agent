"""Code generator node: produce executable Python analysis code from plan and profile."""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.prompts.few_shot_examples import CODE_GENERATOR_EXAMPLES
from app.prompts.system_prompts import CODE_GENERATOR_PROMPT
from app.schemas.output_schemas import GeneratedCode
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


def _build_user_message_content(state: AgentState) -> str:
    parts = [
        f"User question: {state['user_question']}",
        "Analysis plan:\n" + "\n".join(state.get("analysis_plan", [])),
        f"Dataset profile summary:\n{json.dumps(state['df_profile'], indent=2)}",
    ]

    retry_count = state.get("retry_count", 0)
    if retry_count > 0:
        parts.extend(
            [
                f"Previous code that failed:\n{state.get('generated_code', '')}",
                f"Execution error:\n{state.get('execution_error', '')}",
                f"Static analysis error:\n{state.get('static_analysis_error', '')}",
                "Please fix the issue and return corrected code.",
            ],
        )

    return "\n\n".join(parts)


def _few_shot_messages() -> list[HumanMessage | AIMessage]:
    return [
        HumanMessage(content=ex["content"])
        if ex["role"] == "user"
        else AIMessage(content=ex["content"])
        for ex in CODE_GENERATOR_EXAMPLES
    ]


def _parse_code_generator_response(
    raw_content: str,
) -> tuple[str, str, str]:
    try:
        parsed = json.loads(_strip_markdown_fences(raw_content))
        return (
            str(parsed.get("code", "")),
            str(parsed.get("explanation", "")),
            str(parsed.get("expected_output_description", "")),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to parse code_generator response: %s | raw_content=%r",
            exc,
            raw_content[:500],
        )
        return raw_content, "", ""


def run_code_generator(state: AgentState) -> dict[str, Any]:
    """Generate Python analysis code from the plan, profile, and optional retry context."""
    entry_log = log_node_entry(
        logger,
        "code_generator",
        {
            "attempt_number": state.get("retry_count", 0) + 1,
            "retry_count": state.get("retry_count", 0),
            "question": state["user_question"],
        },
    )

    user_message_content = _build_user_message_content(state)
    raw_response_content = ""

    try:
        model = get_chat_model()
        messages = (
            [SystemMessage(content=CODE_GENERATOR_PROMPT)]
            + _few_shot_messages()
            + [HumanMessage(content=user_message_content)]
        )
        response = model.invoke(messages)
        raw_response_content = str(response.content)
        log_llm_call(
            logger,
            "code_generator",
            CODE_GENERATOR_PROMPT,
            user_message_content,
            raw_response_content,
        )

        code, explanation, expected_output_description = _parse_code_generator_response(
            raw_response_content,
        )
        generated = GeneratedCode(
            code=code,
            explanation=explanation,
            expected_output_description=expected_output_description,
        )

        exit_log = log_node_exit(
            logger,
            "code_generator",
            {"code_length": len(generated.code)},
        )

        return {
            "generated_code": generated.code,
            "code_explanation": generated.explanation,
            "static_analysis_error": None,
            "agent_log": [entry_log, exit_log],
        }
    except anthropic.AuthenticationError:
        raise
    except Exception as exc:
        log_error(logger, "code_generator", exc)
        exit_log = log_node_exit(logger, "code_generator", {"error": True})

        return {
            "generated_code": "",
            "code_explanation": "Code generation failed.",
            "static_analysis_error": str(exc),
            "agent_log": [entry_log, exit_log],
        }
