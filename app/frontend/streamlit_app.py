"""Streamlit frontend for the data analysis agent.

Run from the project root so that the ``.env`` file and sample datasets resolve:

    streamlit run app/frontend/streamlit_app.py

The UI reuses the exact same pipeline as the CLI: it validates the uploaded CSV
with ``validate_and_load`` and then streams ``app_graph`` node by node, surfacing
the agent's reasoning (profile -> plan -> generated code -> execution -> insights
-> self-critique -> final answer) as it happens.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Streamlit sets sys.path[0] to this file's directory, so add the project root
# (three levels up: frontend -> app -> repo root) to import the ``app`` package.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
)

# Importing the graph pulls in settings, which raises if ANTHROPIC_API_KEY is
# missing. Fail gracefully with an actionable message instead of a stack trace.
try:
    from app.agent.graph import app_graph
    from app.config.settings import settings
    from app.validator import validate_and_load
except Exception as exc:  # noqa: BLE001 - surface any import/config failure to the user
    st.error(
        "Failed to start the agent. Make sure `ANTHROPIC_API_KEY` is set in your "
        "`.env` file and that you launched Streamlit from the project root.\n\n"
        f"Details: {exc}"
    )
    st.stop()


# --- Node presentation metadata -------------------------------------------------
# Maps each graph node to a human label + the loop stage it belongs to. Drives
# the live progress log while the graph streams.
NODE_META: dict[str, dict[str, str]] = {
    "profiler": {"label": "Profiling the dataset", "stage": "Reason"},
    "clarifier": {"label": "Checking the question for ambiguity", "stage": "Reason"},
    "planner": {"label": "Planning the analysis", "stage": "Plan"},
    "code_generator": {"label": "Writing pandas code", "stage": "Act"},
    "observer": {"label": "Executing code in the sandbox", "stage": "Observe"},
    "synthesizer": {"label": "Synthesizing insights", "stage": "Respond"},
    "critic": {"label": "Self-critiquing the answer", "stage": "Respond"},
    "responder": {"label": "Composing the final answer", "stage": "Respond"},
}

CONFIDENCE_BADGE = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}

SAMPLE_DIR = _PROJECT_ROOT / "data"


def list_sample_datasets() -> dict[str, Path]:
    """Return {display name: path} for CSVs shipped in the data/ folder."""
    if not SAMPLE_DIR.is_dir():
        return {}
    return {p.name: p for p in sorted(SAMPLE_DIR.glob("*.csv"))}


def resolve_csv_path(uploaded_file: Any, sample_choice: str, samples: dict[str, Path]) -> Path | None:
    """Persist the chosen CSV to a real path the sandbox subprocess can re-read."""
    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded_file.getvalue())
        tmp.flush()
        tmp.close()
        return Path(tmp.name)
    if sample_choice and sample_choice in samples:
        return samples[sample_choice]
    return None


def run_agent(csv_path: Path, question: str) -> dict[str, Any]:
    """Validate the CSV, stream the graph, and render live progress.

    Returns the merged final state so the caller can render results.
    """
    validation_result, df = validate_and_load(str(csv_path))
    if not validation_result.passed:
        st.error(f"❌ Validation failed: {validation_result.error_message}")
        return {}

    st.success(
        f"✅ CSV validated — {validation_result.row_count:,} rows × "
        f"{validation_result.column_count} columns ({validation_result.file_size_mb} MB)"
    )

    initial_state: dict[str, Any] = {
        "csv_path": str(csv_path),
        "user_question": question,
        "df": df,
        "validation_result": validation_result.model_dump(),
        "web_mode": True,
        "messages": [],
        "agent_log": [],
        "retry_count": 0,
        "critique_iteration": 0,
    }

    merged: dict[str, Any] = dict(initial_state)

    with st.status("🤖 Agent working…", expanded=True) as status:
        for chunk in app_graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in chunk.items():
                if isinstance(update, dict):
                    merged.update(update)
                meta = NODE_META.get(node_name, {"label": node_name, "stage": ""})
                detail = _node_progress_detail(node_name, merged)
                line = f"**{meta['stage']}** · {meta['label']}"
                if detail:
                    line += f" — {detail}"
                st.write(f"✓ {line}")
                status.update(label=f"🤖 {meta['label']}…")
        status.update(label="✅ Analysis complete", state="complete", expanded=False)

    return merged


def _node_progress_detail(node_name: str, state: dict[str, Any]) -> str:
    """One-line 'what just happened' note for the live progress log."""
    if node_name == "profiler":
        warnings = state.get("df_profile", {}).get("warnings", [])
        return f"{len(warnings)} data warning(s)" if warnings else "no data warnings"
    if node_name == "clarifier":
        n = len(state.get("clarifying_questions", []))
        return f"{n} clarifying question(s) noted" if n else "question is clear"
    if node_name == "planner":
        return f"{len(state.get('analysis_plan', []))} step(s)"
    if node_name == "observer":
        if state.get("execution_error"):
            return f"error — retry {state.get('retry_count', 0)}"
        return "executed successfully"
    if node_name == "synthesizer":
        return f"{len(state.get('insights', []))} insight(s)"
    if node_name == "critic":
        return f"score {state.get('critique_score', 0):.1f}/10"
    return ""


def render_results(state: dict[str, Any]) -> None:
    """Render the final answer, insights, and transparency panels."""
    final_response = state.get("final_response") or {}
    if "error" in final_response:
        st.error(f"The agent could not finish: {final_response['error']}")
        return

    # --- Headline metrics ---
    validation = state.get("validation_result", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows analyzed", f"{validation.get('row_count') or 0:,}")
    c2.metric("Columns", validation.get("column_count") or 0)
    c3.metric("Self-critique score", f"{state.get('critique_score', 0):.1f}/10")
    c4.metric(
        "Retries · critiques",
        f"{state.get('retry_count', 0)} · {state.get('critique_iteration', 0)}",
    )

    # --- Final natural-language answer ---
    answer = state.get("natural_language_response") or final_response.get(
        "natural_language_response", ""
    )
    if answer:
        st.subheader("Answer")
        st.markdown(answer)

    # --- Structured insights ---
    report = state.get("insight_report", {})
    insights = report.get("insights", [])
    if insights:
        st.subheader("Key insights")
        for insight in insights:
            badge = CONFIDENCE_BADGE.get(insight.get("confidence", "low"), "⚪️")
            with st.container(border=True):
                st.markdown(f"**{insight.get('title', 'Insight')}**  ·  {badge}")
                st.write(insight.get("finding", ""))
                if insight.get("chart_suggestion"):
                    st.caption(f"📈 Suggested chart: {insight['chart_suggestion']}")

    coverage = report.get("data_coverage_note")
    limitations = report.get("limitations", [])
    if coverage:
        st.info(f"**Data coverage:** {coverage}")
    if limitations:
        st.warning("**Limitations**\n\n" + "\n".join(f"- {lim}" for lim in limitations))

    # --- Transparency: how the agent got here ---
    st.subheader("How the agent got here")

    with st.expander("🧠 Dataset profile"):
        st.markdown(state.get("profile_summary", "_No summary available._"))
        st.json(state.get("df_profile", {}), expanded=False)

    clarifying = state.get("clarifying_questions", [])
    if clarifying:
        with st.expander("❓ Clarifying questions the agent flagged"):
            for q in clarifying:
                st.markdown(f"- {q}")
            st.caption(
                "In web mode the agent does not block on these; add any context to "
                "the 'extra context' box and re-run to address them."
            )

    with st.expander("🗺️ Analysis plan"):
        plan = state.get("analysis_plan", [])
        for i, step in enumerate(plan, 1):
            st.markdown(f"{i}. {step}")

    with st.expander("💻 Generated code (ran in the sandbox)"):
        st.code(state.get("generated_code", "# no code generated"), language="python")
        if state.get("code_explanation"):
            st.caption(state["code_explanation"])

    with st.expander("📤 Execution output"):
        if state.get("execution_error"):
            st.error(state["execution_error"])
        st.code(state.get("execution_result") or "(no stdout)", language="text")
        if state.get("execution_truncated"):
            st.caption("⚠️ Output was truncated due to size limits.")

    with st.expander("🔍 Self-critique"):
        st.metric("Final score", f"{state.get('critique_score', 0):.1f}/10")
        issues = state.get("critique_issues", [])
        if issues:
            st.markdown("**Issues raised & resolved:**")
            for issue in issues:
                st.markdown(f"- {issue}")
        else:
            st.caption("No outstanding issues.")

    with st.expander("📜 Full execution trace"):
        st.code("\n".join(state.get("agent_log", [])), language="text")


# --- Page layout ----------------------------------------------------------------
st.title("📊 Data Analysis Agent")
st.caption(
    "Upload a CSV, ask a question in plain English, and watch the agent reason, "
    "write and run its own pandas code, self-critique, and report insights."
)

with st.sidebar:
    st.header("Settings")
    st.write(f"**Model:** `{settings.model_name}`")
    st.write(f"**Max retries:** {settings.max_retries}")
    st.write(f"**Critique threshold:** {settings.critique_approval_threshold}")
    st.divider()
    st.caption(
        "The agent runs the same LangGraph pipeline as the CLI: "
        "Reason → Plan → Act → Observe → Respond, with a self-critique loop."
    )

samples = list_sample_datasets()

left, right = st.columns([1, 1])
with left:
    uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
with right:
    sample_choice = st.selectbox(
        "…or pick a sample dataset",
        options=["(none)"] + list(samples.keys()),
        index=0,
    )
    sample_choice = "" if sample_choice == "(none)" else sample_choice

# Preview whichever CSV is selected.
preview_path = resolve_csv_path(uploaded_file, sample_choice, samples) if (uploaded_file or sample_choice) else None
if preview_path is not None:
    try:
        st.dataframe(pd.read_csv(preview_path, nrows=50), use_container_width=True)
        st.caption("Showing up to the first 50 rows.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not preview file: {exc}")

question = st.text_input(
    "Your question",
    placeholder="e.g. What are the top performing regions by revenue?",
)
extra_context = st.text_area(
    "Extra context (optional)",
    placeholder="Definitions, which columns to use, assumptions to make…",
    height=80,
)

run_clicked = st.button("Analyze", type="primary", use_container_width=True)

if run_clicked:
    if preview_path is None:
        st.error("Please upload a CSV or choose a sample dataset first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        full_question = question.strip()
        if extra_context.strip():
            full_question += f"\n\nAdditional context: {extra_context.strip()}"
        final_state = run_agent(preview_path, full_question)
        if final_state:
            render_results(final_state)
