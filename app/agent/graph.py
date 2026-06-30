"""LangGraph StateGraph wiring for the data analysis agent.

Flow:
  profiler → clarifier → planner → code_generator → observer
       ↓ (retry on execution error, up to max_retries)
  code_generator ←──────────────────────────────────┘
       ↓ (success or retries exhausted)
  synthesizer → critic
       ↓ (score below threshold and iterations remaining)
  synthesizer ←────────────────────────────────────┘
       ↓ (approved or max critique iterations)
  responder → END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes.clarifier import run_clarifier
from app.agent.nodes.code_generator import run_code_generator
from app.agent.nodes.critic import run_critic
from app.agent.nodes.observer import run_observer
from app.agent.nodes.planner import run_planner
from app.agent.nodes.profiler import run_profiler
from app.agent.nodes.responder import run_responder
from app.agent.nodes.synthesizer import run_synthesizer
from app.config.settings import settings
from app.schemas.state_schema import AgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


def route_after_observer(state: AgentState) -> str:
    """Route to code retry or insight synthesis after code execution."""
    if state.get("retry_count", 0) >= settings.max_retries:
        return "synthesizer"
    if state.get("execution_error") is not None:
        return "code_generator"
    return "synthesizer"


def route_after_critic(state: AgentState) -> str:
    """Route to another synthesis pass or final response after critique."""
    score = state.get("critique_score", 0.0)
    iteration = state.get("critique_iteration", 0)
    if score >= settings.critique_approval_threshold:
        return "responder"
    if iteration >= settings.max_critique_iterations:
        return "responder"
    return "synthesizer"


def build_graph() -> CompiledStateGraph:
    """Construct and compile the agent StateGraph."""
    logger.info("Building data analysis agent graph")

    graph = StateGraph(AgentState)

    graph.add_node("profiler", run_profiler)
    graph.add_node("clarifier", run_clarifier)
    graph.add_node("planner", run_planner)
    graph.add_node("code_generator", run_code_generator)
    graph.add_node("observer", run_observer)
    graph.add_node("synthesizer", run_synthesizer)
    graph.add_node("critic", run_critic)
    graph.add_node("responder", run_responder)

    graph.set_entry_point("profiler")

    graph.add_edge("profiler", "clarifier")
    graph.add_edge("clarifier", "planner")
    graph.add_edge("planner", "code_generator")
    graph.add_edge("code_generator", "observer")
    graph.add_edge("synthesizer", "critic")

    graph.add_conditional_edges(
        "observer",
        route_after_observer,
        {
            "code_generator": "code_generator",
            "synthesizer": "synthesizer",
        },
    )
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "synthesizer": "synthesizer",
            "responder": "responder",
        },
    )

    graph.add_edge("responder", END)

    return graph.compile()


app_graph = build_graph()
