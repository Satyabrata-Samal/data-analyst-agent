"""CLI entry point for the data analysis agent.

Example usage (from project root):
    python -m app.main data.csv "What are the top performing regions?"
    python main.py data.csv "What are the top performing regions?"
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from app.agent.graph import app_graph
from app.config.settings import settings
from app.schemas.output_schemas import ValidationResult
from app.utils.logger import get_logger
from app.validator import validate_and_load

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CSV path and analysis question from the command line."""
    parser = argparse.ArgumentParser(
        description="Data Analysis Agent — ask questions about your CSV data.",
    )
    parser.add_argument("csv_path", type=str, help="Path to the CSV file to analyze")
    parser.add_argument("question", type=str, help="Analysis question about the data")
    return parser.parse_args()


def main() -> int:
    """Validate input, run the agent graph, and return a process exit code."""
    args = parse_args()

    validation_result, df = validate_and_load(args.csv_path)

    if not validation_result.passed:
        print(f"\n❌ Validation failed:\n{validation_result.error_message}\n")
        return 1

    print(
        f"\n✅ CSV validated: {validation_result.row_count} rows, "
        f"{validation_result.column_count} columns, "
        f"{validation_result.file_size_mb}MB\n"
    )

    initial_state: dict[str, Any] = {
        "csv_path": args.csv_path,
        "user_question": args.question,
        "df": df,
        "validation_result": validation_result.model_dump(),
        "messages": [],
        "agent_log": [],
        "retry_count": 0,
        "critique_iteration": 0,
    }

    print("\n🤖 Agent starting analysis...\n")

    # region agent log
    import json as _json
    import time as _time

    with open(
        "/Users/satya/Documents/ensemble_data_analysis_agent/.cursor/debug-c92400.log",
        "a",
        encoding="utf-8",
    ) as _dbg_f:
        _dbg_f.write(
            _json.dumps(
                {
                    "sessionId": "c92400",
                    "hypothesisId": "H2",
                    "location": "app/main.py:pre_invoke",
                    "message": "initial_state keys before graph invoke",
                    "data": {
                        "keys": sorted(initial_state.keys()),
                        "has_csv_path": "csv_path" in initial_state,
                        "has_df": "df" in initial_state,
                        "has_user_question": "user_question" in initial_state,
                    },
                    "timestamp": int(_time.time() * 1000),
                }
            )
            + "\n"
        )
    # endregion

    try:
        final_state = app_graph.invoke(initial_state)
    except Exception as exc:
        logger.error(f"Graph execution failed: {exc}")
        print(f"\n❌ Agent encountered an unrecoverable error: {exc}\n")
        return 1

    print(f"\n📋 Full execution trace saved to: {settings.log_file}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
