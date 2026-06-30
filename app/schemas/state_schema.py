"""Central state schema for the data analysis LangGraph agent.

AgentState is a TypedDict that flows through every node in the graph. Each node
reads the fields it needs, mutates or appends to relevant fields, and returns
the updated state to the next node.

Typical flow:
  1. Input (csv_path, user_question) enters via the CLI.
  2. Validator checks the question; on failure, validation_error is set.
  3. Profiler populates df_profile from the CSV.
  4. Clarifier may ask questions; clarifying_answers are collected until
     clarification_done is True.
  5. Planner writes analysis_plan; code_generator produces generated_code.
  6. Observer executes code and fills execution_result / execution_error.
  7. Synthesizer and critic populate insights and critique fields; the graph
     may loop back to code_generator if critique_iteration < 2.
  8. Responder sets final_response; agent_log accumulates one line per node
     for a full execution trace.
"""

from typing import Any, TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from operator import add

class InputState(TypedDict):
    csv_path: str
    user_question: str

class AgentState(TypedDict, total=False):

    # --- Profiling fields ---
    df_profile: dict[str, Any]  # shape, dtypes, nulls, sample rows, describe stats
    validation_error: str | None

    messages: Annotated[list[BaseMessage], add_messages]

    # --- Clarification fields ---
    clarifying_questions: list[str]
    clarifying_answers: list[str]
    clarification_done: bool

    # --- Planning fields ---
    analysis_plan: list[str]

    # --- Code generation fields ---
    generated_code: str
    code_explanation: str
    retry_count: int  # starts at 0, max 3
    static_analysis_error: str | None

    # --- Execution fields ---
    execution_result: str | None
    execution_error: str | None
    execution_truncated: bool

    # --- Insight fields ---
    # Each insight dict: {title, finding, confidence, chart_suggestion}
    insights: list[dict[str, Any]]
    critique_score: float  # 0-10
    critique_issues: list[str]
    critique_iteration: int  # starts at 0, max 2

    # --- Response fields ---
    final_response: dict[str, Any] | None
    agent_log: Annotated[list[str], add]
