"""Lightweight qualitative and quantitative evaluation for the data analysis agent.

Runs fixed test scenarios, compares agent output against expected outcomes and
keywords, and prints a pass/fail report with keyword match scores.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.graph import app_graph
from app.evaluation.test_scenarios import TEST_SCENARIOS, TestScenario
from app.utils.logger import get_logger
from app.validator import validate_and_load

logger = get_logger(__name__)


@dataclass
class EvalResult:
    scenario_name: str
    question: str
    expected_outcome: str
    actual_response: str
    keyword_matches: list[str]
    keyword_misses: list[str]
    keyword_match_score: float
    passed: bool
    error: str | None


def _write_temp_csv(csv_data: str) -> str:
    """Write CSV content to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        encoding="utf-8",
    ) as temp_file:
        temp_file.write(csv_data)
        return temp_file.name


def _extract_actual_response(final_state: dict[str, Any]) -> str:
    """Return the last AI message content, or serialised final_response as fallback."""
    messages = final_state.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)

    final_response = final_state.get("final_response", {})
    return json.dumps(final_response)


def _score_keywords(
    actual_response: str,
    expected_keywords: list[str],
) -> tuple[list[str], list[str], float]:
    """Compute keyword matches, misses, and match score."""
    actual_lower = actual_response.lower()
    keyword_matches = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() in actual_lower
    ]
    keyword_misses = [
        keyword for keyword in expected_keywords if keyword not in keyword_matches
    ]
    if expected_keywords:
        score = len(keyword_matches) / len(expected_keywords)
    else:
        score = 1.0
    return keyword_matches, keyword_misses, score


def _build_eval_result(
    scenario: TestScenario,
    *,
    actual_response: str,
    keyword_matches: list[str],
    keyword_misses: list[str],
    keyword_match_score: float,
    error: str | None,
) -> EvalResult:
    return EvalResult(
        scenario_name=scenario.name,
        question=scenario.question,
        expected_outcome=scenario.expected_outcome,
        actual_response=actual_response,
        keyword_matches=keyword_matches,
        keyword_misses=keyword_misses,
        keyword_match_score=keyword_match_score,
        passed=keyword_match_score >= 0.5,
        error=error,
    )


def run_scenario(scenario: TestScenario) -> EvalResult:
    """Run a single test scenario through the agent graph and score keyword matches."""
    temp_path = _write_temp_csv(scenario.csv_data)

    try:
        validation_result, df = validate_and_load(temp_path)

        if not validation_result.passed:
            return _build_eval_result(
                scenario,
                actual_response="",
                keyword_matches=[],
                keyword_misses=list(scenario.expected_keywords),
                keyword_match_score=0.0,
                error=validation_result.error_message,
            )

        initial_state: dict[str, Any] = {
            "csv_path": temp_path,
            "user_question": scenario.question,
            "df": df,
            "validation_result": validation_result.model_dump(),
            "messages": [],
            "agent_log": [],
            "retry_count": 0,
            "critique_iteration": 0,
        }

        try:
            final_state = app_graph.invoke(initial_state)
            actual_response = _extract_actual_response(final_state)
            keyword_matches, keyword_misses, score = _score_keywords(
                actual_response,
                scenario.expected_keywords,
            )

            return _build_eval_result(
                scenario,
                actual_response=actual_response,
                keyword_matches=keyword_matches,
                keyword_misses=keyword_misses,
                keyword_match_score=score,
                error=None,
            )
        except Exception as exc:
            logger.error("Scenario %s failed: %s", scenario.name, exc)
            return _build_eval_result(
                scenario,
                actual_response="",
                keyword_matches=[],
                keyword_misses=list(scenario.expected_keywords),
                keyword_match_score=0.0,
                error=str(exc),
            )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_all_scenarios() -> list[EvalResult]:
    """Run every configured test scenario and return evaluation results."""
    results: list[EvalResult] = []

    for scenario in TEST_SCENARIOS:
        print(f"Running scenario: {scenario.name}...")
        results.append(run_scenario(scenario))

    return results


def print_evaluation_report(results: list[EvalResult]) -> None:
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    for result in results:
        print(f"\nScenario: {result.scenario_name}")
        print(f"Question: {result.question}")
        print(f"Expected: {result.expected_outcome}")
        print(
            f"Keyword matches: {result.keyword_matches} / "
            f"misses: {result.keyword_misses}"
        )
        print(f"Score: {result.keyword_match_score:.2f}")
        print(f"Status: {'PASS' if result.passed else 'FAIL'}")
        if result.error:
            print(f"Error: {result.error}")
        print("-" * 70)

    total = len(results)
    if total == 0:
        print("\nSUMMARY: 0/0 scenarios passed")
        print("Average score: 0.00")
        return

    passed = sum(1 for result in results if result.passed)
    average_score = sum(result.keyword_match_score for result in results) / total

    print(f"\nSUMMARY: {passed}/{total} scenarios passed")
    print(f"Average score: {average_score:.2f}")


if __name__ == "__main__":
    scenario_results = run_all_scenarios()
    print_evaluation_report(scenario_results)
