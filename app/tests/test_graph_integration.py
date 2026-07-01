"""Integration tests for the compiled LangGraph app.

Uses langchain_core's FakeListChatModel to return canned JSON responses in the
order nodes call the LLM. Verifies routing (happy path, code-retry loop,
critique loop), max-iteration caps, and state field population — all with
zero real API calls.

Node call order in the graph:
    profiler, clarifier, planner, code_generator, observer,
    (code_generator, observer) * on retry,
    synthesizer, critic,
    (synthesizer, critic) * on low score,
    responder.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import httpx
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agent.graph import app_graph


# --- helpers ---


def _profiler_reply() -> str:
    return "Dataset appears to be about: sales."


def _clarifier_reply(needs: bool = False, questions: list[str] | None = None) -> str:
    return json.dumps(
        {
            "needs_clarification": needs,
            "questions": questions or [],
        }
    )


def _planner_reply() -> str:
    return json.dumps(
        {
            "reasoning": "sum revenue per region",
            "steps": ["Step 1: groupby region and sum revenue", "Step 2: print"],
        }
    )


def _code_reply(code: str) -> str:
    return json.dumps(
        {
            "code": code,
            "explanation": "aggregates revenue by region",
            "expected_output_description": "printed table",
        }
    )


def _observer_reply(status: str, fix: str | None = None) -> str:
    return json.dumps(
        {
            "status": status,
            "assessment": "ok" if status == "success" else "bad",
            "fix_suggestion": fix,
        }
    )


def _synth_reply() -> str:
    return json.dumps(
        {
            "insights": [
                {
                    "title": "West leads",
                    "finding": "West has highest revenue",
                    "confidence": "high",
                    "chart_suggestion": None,
                }
            ],
            "data_coverage_note": "full dataset",
            "limitations": ["small sample"],
        }
    )


def _critic_reply(score: float) -> str:
    return json.dumps(
        {
            "score": score,
            "issues": [] if score >= 7 else ["too vague"],
            "suggestions": [] if score >= 7 else ["name the region"],
        }
    )


def _responder_reply() -> str:
    return "West is the top region by revenue with $31,000."


@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    path = tmp_path / "sample.csv"
    path.write_text(
        "region,revenue\nNorth,15000\nSouth,22000\nEast,9000\nWest,31000\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def sample_df(sample_csv: str) -> pd.DataFrame:
    return pd.read_csv(sample_csv)


def _initial_state(csv_path: str, df: pd.DataFrame, question: str = "top region?") -> dict:
    return {
        "csv_path": csv_path,
        "user_question": question,
        "df": df,
        "validation_result": {
            "passed": True,
            "row_count": len(df),
            "column_count": len(df.columns),
            "file_size_mb": 0.01,
            "was_sampled": False,
        },
        "messages": [],
        "agent_log": [],
        "retry_count": 0,
        "critique_iteration": 0,
    }


_NODE_MODULES = (
    "app.agent.nodes.profiler",
    "app.agent.nodes.clarifier",
    "app.agent.nodes.planner",
    "app.agent.nodes.code_generator",
    "app.agent.nodes.observer",
    "app.agent.nodes.synthesizer",
    "app.agent.nodes.critic",
    "app.agent.nodes.responder",
)


def _patch_llm(monkeypatch, responses: list[str]) -> FakeListChatModel:
    # Each node did `from app.utils.llm import get_chat_model`, so patching the
    # source module doesn't rebind the local name. Patch each node's own copy.
    fake = FakeListChatModel(responses=responses)
    for module_path in _NODE_MODULES:
        monkeypatch.setattr(f"{module_path}.get_chat_model", lambda fake=fake: fake)
    return fake


# --- happy path ---


def test_graph_happy_path(monkeypatch, sample_csv, sample_df):
    _patch_llm(
        monkeypatch,
        [
            _profiler_reply(),
            _clarifier_reply(needs=False),
            _planner_reply(),
            _code_reply("print(df.groupby('region')['revenue'].sum())"),
            _observer_reply("success"),
            _synth_reply(),
            _critic_reply(8.5),
            _responder_reply(),
        ],
    )

    final = app_graph.invoke(_initial_state(sample_csv, sample_df))

    assert final["final_response"]["question"] == "top region?"
    assert final["retry_count"] == 0
    assert final["critique_iteration"] == 1
    assert final["critique_score"] == 8.5
    assert final["final_response"]["report"]["insights"][0]["title"] == "West leads"


# --- code retry loop ---


def test_graph_code_retry_then_success(monkeypatch, sample_csv, sample_df):
    _patch_llm(
        monkeypatch,
        [
            _profiler_reply(),
            _clarifier_reply(),
            _planner_reply(),
            # attempt 1: code that raises at runtime
            _code_reply("raise ValueError('nope')"),
            _observer_reply("retry", fix="handle missing values"),
            # attempt 2: good code
            _code_reply("print(df.groupby('region')['revenue'].sum())"),
            _observer_reply("success"),
            _synth_reply(),
            _critic_reply(9.0),
            _responder_reply(),
        ],
    )

    final = app_graph.invoke(_initial_state(sample_csv, sample_df))

    assert final["retry_count"] == 1
    assert final["final_response"]["total_retries"] == 1


# --- critique loop ---


def test_graph_critique_loop(monkeypatch, sample_csv, sample_df):
    _patch_llm(
        monkeypatch,
        [
            _profiler_reply(),
            _clarifier_reply(),
            _planner_reply(),
            _code_reply("print(df.groupby('region')['revenue'].sum())"),
            _observer_reply("success"),
            # first draft — critic says low score
            _synth_reply(),
            _critic_reply(4.0),
            # second draft — critic approves
            _synth_reply(),
            _critic_reply(8.0),
            _responder_reply(),
        ],
    )

    final = app_graph.invoke(_initial_state(sample_csv, sample_df))

    assert final["critique_iteration"] == 2
    assert final["critique_score"] == 8.0
    assert final["final_response"]["critique_iterations"] == 2


# --- max iteration caps ---


def test_graph_max_critique_iterations_exits_even_when_score_stays_low(
    monkeypatch, sample_csv, sample_df
):
    # critic always returns 3.0 — graph should still terminate at max_critique_iterations
    _patch_llm(
        monkeypatch,
        [
            _profiler_reply(),
            _clarifier_reply(),
            _planner_reply(),
            _code_reply("print(df.groupby('region')['revenue'].sum())"),
            _observer_reply("success"),
            _synth_reply(),
            _critic_reply(3.0),
            _synth_reply(),
            _critic_reply(3.0),
            _responder_reply(),
        ],
    )

    final = app_graph.invoke(_initial_state(sample_csv, sample_df))

    assert final["critique_iteration"] == 2
    assert final["critique_score"] == 3.0
    assert "final_response" in final


# --- fatal LLM errors must not loop ---


def _auth_error() -> anthropic.AuthenticationError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request, json={"error": {"message": "invalid x-api-key"}})
    return anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)


class _RaisingModel:
    def invoke(self, *args, **kwargs):
        raise _auth_error()


def test_graph_raises_on_authentication_error_instead_of_looping(
    monkeypatch, sample_csv, sample_df
):
    # profiler/clarifier/planner succeed normally, then code_generator hits a
    # bad API key. This must propagate out of app_graph.invoke immediately
    # rather than bouncing between code_generator and observer forever.
    fake = FakeListChatModel(
        responses=[_profiler_reply(), _clarifier_reply(), _planner_reply()],
    )
    for module_path in ("app.agent.nodes.profiler", "app.agent.nodes.clarifier", "app.agent.nodes.planner"):
        monkeypatch.setattr(f"{module_path}.get_chat_model", lambda fake=fake: fake)
    monkeypatch.setattr(
        "app.agent.nodes.code_generator.get_chat_model", lambda: _RaisingModel()
    )

    with pytest.raises(anthropic.AuthenticationError):
        app_graph.invoke(_initial_state(sample_csv, sample_df))


def test_graph_increments_retry_count_on_generic_observer_exception(
    monkeypatch, sample_csv, sample_df
):
    # A non-fatal, unexpected exception from the observer's LLM call must still
    # advance retry_count so route_after_observer eventually stops retrying,
    # instead of looping with retry_count stuck at 0.
    _patch_llm(
        monkeypatch,
        [
            _profiler_reply(),
            _clarifier_reply(),
            _planner_reply(),
            _code_reply("print(df.groupby('region')['revenue'].sum())"),
            _synth_reply(),
            _critic_reply(8.0),
            _responder_reply(),
        ],
    )

    class _BrokenObserverModel:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("transient observer failure")

    monkeypatch.setattr(
        "app.agent.nodes.observer.get_chat_model", lambda: _BrokenObserverModel()
    )
    monkeypatch.setattr("app.agent.graph.settings.max_retries", 1)

    final = app_graph.invoke(_initial_state(sample_csv, sample_df))

    assert final["retry_count"] == 1


# --- routing predicates in isolation ---


def test_route_after_observer_on_error_and_success():
    from app.agent.graph import route_after_observer

    assert route_after_observer({"retry_count": 0, "execution_error": "boom"}) == "code_generator"
    assert route_after_observer({"retry_count": 0, "execution_error": None}) == "synthesizer"


def test_route_after_observer_respects_max_retries(monkeypatch):
    from app.agent import graph as graph_module

    monkeypatch.setattr(graph_module.settings, "max_retries", 3)
    state = {"retry_count": 3, "execution_error": "still failing"}
    # even with error, exhausted retries route to synthesizer
    assert graph_module.route_after_observer(state) == "synthesizer"


def test_route_after_critic_score_threshold(monkeypatch):
    from app.agent import graph as graph_module

    monkeypatch.setattr(graph_module.settings, "critique_approval_threshold", 7.0)
    monkeypatch.setattr(graph_module.settings, "max_critique_iterations", 2)

    # high score → responder
    assert (
        graph_module.route_after_critic({"critique_score": 8.0, "critique_iteration": 1})
        == "responder"
    )
    # low score but iterations remain → synthesizer
    assert (
        graph_module.route_after_critic({"critique_score": 5.0, "critique_iteration": 1})
        == "synthesizer"
    )
    # low score but iterations exhausted → responder
    assert (
        graph_module.route_after_critic({"critique_score": 5.0, "critique_iteration": 2})
        == "responder"
    )
