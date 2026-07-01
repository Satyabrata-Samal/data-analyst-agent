"""Observer node: execute generated code and assess whether results warrant a retry."""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import settings
from app.prompts.system_prompts import OBSERVER_PROMPT
from app.schemas.state_schema import AgentState
from app.tools.code_executor import execute_code
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


def _build_user_message_content(
    state: AgentState,
    *,
    execution_result: str | None,
    execution_error: str | None,
    execution_truncated: bool,
) -> str:
    parts = [
        f"Generated code:\n{state.get('generated_code', '')}",
        "Analysis plan:\n" + "\n".join(state.get("analysis_plan", [])),
        f"Execution stdout:\n{execution_result or '(empty)'}",
    ]

    if execution_error:
        parts.append(f"Execution error:\n{execution_error}")

    if execution_truncated:
        parts.append("Note: stdout was truncated due to size limits.")

    return "\n\n".join(parts)


def _parse_observer_response(
    raw_content: str,
    *,
    execution_succeeded: bool,
) -> tuple[str, str, str | None]:
    try:
        parsed = json.loads(_strip_markdown_fences(raw_content))
        status = str(parsed.get("status", "success" if execution_succeeded else "retry"))
        assessment = str(parsed.get("assessment", ""))
        fix_suggestion = parsed.get("fix_suggestion")
        if fix_suggestion is not None:
            fix_suggestion = str(fix_suggestion)
        return status, assessment, fix_suggestion
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to parse observer response: %s | raw_content=%r",
            exc,
            raw_content[:500],
        )
        default_status = "success" if execution_succeeded else "retry"
        return default_status, raw_content, None


def run_observer(state: AgentState) -> dict[str, Any]:
    """Execute generated analysis code and assess the result for success or retry."""
    generated_code = state.get("generated_code", "")
    retry_count = state.get("retry_count", 0)

    entry_log = log_node_entry(
        logger,
        "observer",
        {
            "retry_count": retry_count,
            "code_length": len(generated_code),
        },
    )

    user_message_content = ""
    raw_response_content = ""

    try:
        result = execute_code(generated_code, state["csv_path"])

        execution_truncated = result.truncated
        if result.success:
            execution_result: str | None = result.stdout
            execution_error: str | None = None
            static_analysis_error: str | None = None
        else:
            execution_result = None
            execution_error = result.stderr
            static_analysis_error = (
                result.stderr if result.error_type == "static_analysis" else None
            )

        user_message_content = _build_user_message_content(
            state,
            execution_result=execution_result,
            execution_error=execution_error,
            execution_truncated=execution_truncated,
        )

        model = get_chat_model()
        response = model.invoke(
            [
                SystemMessage(content=OBSERVER_PROMPT),
                HumanMessage(content=user_message_content),
            ],
        )
        raw_response_content = str(response.content)
        log_llm_call(
            logger,
            "observer",
            OBSERVER_PROMPT,
            user_message_content,
            raw_response_content,
        )

        status, assessment, fix_suggestion = _parse_observer_response(
            raw_response_content,
            execution_succeeded=result.success,
        )

        new_retry_count = retry_count
        if status == "retry":
            observer_note = assessment
            if fix_suggestion:
                observer_note = f"{assessment}\nFix: {fix_suggestion}"

            if execution_error:
                execution_error = f"{execution_error}\n\nObserver: {observer_note}"
            else:
                execution_error = observer_note

            if retry_count < settings.max_retries:
                new_retry_count = retry_count + 1
        elif result.success:
            execution_error = None
            static_analysis_error = None

        exit_log = log_node_exit(
            logger,
            "observer",
            {
                "execution_success": result.success,
                "status": status,
                "retry_count": new_retry_count,
            },
        )

        return {
            "execution_result": execution_result,
            "execution_error": execution_error,
            "execution_truncated": execution_truncated,
            "static_analysis_error": static_analysis_error,
            "retry_count": new_retry_count,
            "agent_log": [entry_log, exit_log],
        }
    except anthropic.AuthenticationError:
        # Bad/expired credentials never succeed on retry — abort instead of
        # looping the graph back to code_generator until max_retries burns out.
        raise
    except Exception as exc:
        log_error(logger, "observer", exc)
        exit_log = log_node_exit(logger, "observer", {"error": True})
        new_retry_count = min(retry_count + 1, settings.max_retries)

        return {
            "execution_result": None,
            "execution_error": str(exc),
            "execution_truncated": False,
            "static_analysis_error": None,
            "retry_count": new_retry_count,
            "agent_log": [entry_log, exit_log],
        }
