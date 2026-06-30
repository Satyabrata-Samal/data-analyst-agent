"""Profiler node: build a structured dataset profile and LLM summary."""

from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config.settings import settings
from app.prompts.system_prompts import PROFILER_PROMPT
from app.schemas.state_schema import AgentState
from app.tools.csv_profiler import profile_dataframe
from app.utils.logger import get_logger, log_error, log_node_entry, log_node_exit

logger = get_logger(__name__)


def run_profiler(state: AgentState) -> dict[str, Any]:
    """Profile the validated DataFrame and produce a natural-language summary."""
    entry_log = log_node_entry(logger, "profiler", {"csv_path": state["csv_path"]})

    df = state["df"]
    df_profile = profile_dataframe(df)
    profile_json = json.dumps(df_profile, indent=2)

    profile_summary = "Profile summary unavailable due to error."

    try:
        model = ChatAnthropic(
            model=settings.model_name,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.max_tokens,
        )
        llm_messages = [
            SystemMessage(content=PROFILER_PROMPT),
            HumanMessage(content=f"Here is the dataset profile:\n\n{profile_json}"),
        ]
        response = model.invoke(llm_messages)
        profile_summary = str(response.content)
    except Exception as exc:
        log_error(logger, "profiler", exc)

    exit_log = log_node_exit(
        logger,
        "profiler",
        {"warnings": len(df_profile.get("warnings", []))},
    )

    return {
        "df_profile": df_profile,
        "messages": [
            SystemMessage(content=PROFILER_PROMPT),
            HumanMessage(content=f"Dataset profile:\n{profile_json}"),
            AIMessage(content=profile_summary),
        ],
        "agent_log": [entry_log, exit_log],
    }
