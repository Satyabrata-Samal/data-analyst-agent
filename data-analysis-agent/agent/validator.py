"""Input validation and CSV loading for the data analysis agent.

Runs a ordered sequence of file, size, parse, and data-quality checks before
returning a validated (and optionally sampled) DataFrame for downstream nodes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from config.settings import settings
from schemas.output_schemas import ValidationResult
from utils.logger import get_logger, log_error, log_tool_call

logger = get_logger(__name__)


def _fail(
    error_message: str,
    *,
    file_size_mb: float | None = None,
    error: Exception | None = None,
) -> tuple[ValidationResult, None]:
    if error is not None:
        log_error(logger, "validator", error)
    else:
        log_error(logger, "validator", ValueError(error_message))

    return (
        ValidationResult(
            passed=False,
            error_message=error_message,
            file_size_mb=file_size_mb,
            was_sampled=False,
        ),
        None,
    )


def _rename_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}
    new_columns: list[str] = []

    for col in df.columns:
        col_str = str(col)
        if col_str not in counts:
            counts[col_str] = 0
            new_columns.append(col_str)
        else:
            counts[col_str] += 1
            new_columns.append(f"{col_str}_{counts[col_str]}")

    df.columns = new_columns
    return df


def validate_and_load(csv_path: str) -> tuple[ValidationResult, pd.DataFrame | None]:
    """Validate a CSV file and load it, sampling if it exceeds the row threshold."""
    path = Path(csv_path)

    # 1. FILE EXISTS CHECK
    log_tool_call(
        logger,
        "validator",
        {"check": "file_exists", "path": str(path)},
        "running",
    )
    if not path.exists():
        log_tool_call(logger, "validator", {"check": "file_exists"}, "failed: not found")
        return _fail(f"File not found: {path}")

    log_tool_call(logger, "validator", {"check": "file_exists"}, "passed")

    # 2. EXTENSION CHECK
    ext = path.suffix.lower()
    log_tool_call(
        logger,
        "validator",
        {"check": "extension", "extension": ext},
        "running",
    )
    if ext != ".csv":
        log_tool_call(logger, "validator", {"check": "extension"}, f"failed: {ext}")
        return _fail(f"Only .csv files are supported. Got: {ext}")

    log_tool_call(logger, "validator", {"check": "extension"}, "passed")

    # 3. FILE SIZE CHECK
    size_mb = os.path.getsize(path) / (1024 * 1024)
    log_tool_call(
        logger,
        "validator",
        {"check": "file_size", "size_mb": round(size_mb, 2)},
        "running",
    )
    if size_mb > settings.max_file_size_mb:
        log_tool_call(logger, "validator", {"check": "file_size"}, "failed: too large")
        return _fail(
            f"File size {size_mb:.1f}MB exceeds limit of {settings.max_file_size_mb}MB. "
            "Please sample your data and re-upload.",
            file_size_mb=round(size_mb, 2),
        )

    log_tool_call(logger, "validator", {"check": "file_size"}, "passed")

    # 4. ENCODING + PARSE CHECK
    log_tool_call(logger, "validator", {"check": "parse_preview", "nrows": 5}, "running")
    try:
        pd.read_csv(path, nrows=5)
    except Exception as exc:
        log_tool_call(logger, "validator", {"check": "parse_preview"}, f"failed: {exc}")
        return _fail(
            f"Could not parse CSV: {exc}. "
            "Ensure the file is UTF-8 encoded with a consistent delimiter.",
            file_size_mb=round(size_mb, 2),
            error=exc,
        )

    log_tool_call(logger, "validator", {"check": "parse_preview"}, "passed")

    # 5. FULL LOAD CHECK
    log_tool_call(logger, "validator", {"check": "full_load"}, "running")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log_tool_call(logger, "validator", {"check": "full_load"}, f"failed: {exc}")
        return _fail(
            f"Could not parse CSV: {exc}. "
            "Ensure the file is UTF-8 encoded with a consistent delimiter.",
            file_size_mb=round(size_mb, 2),
            error=exc,
        )

    row_count = len(df)
    column_count = len(df.columns)
    log_tool_call(
        logger,
        "validator",
        {"check": "full_load", "row_count": row_count, "column_count": column_count},
        "passed",
    )

    # 6. EMPTY CHECK
    log_tool_call(logger, "validator", {"check": "empty"}, "running")
    if row_count == 0:
        log_tool_call(logger, "validator", {"check": "empty"}, "failed: no rows")
        return _fail(
            "CSV has no data rows.",
            file_size_mb=round(size_mb, 2),
        )

    if column_count == 0:
        log_tool_call(logger, "validator", {"check": "empty"}, "failed: no columns")
        return _fail(
            "CSV has no columns.",
            file_size_mb=round(size_mb, 2),
        )

    log_tool_call(logger, "validator", {"check": "empty"}, "passed")

    # 7. COLUMN COUNT CHECK
    log_tool_call(
        logger,
        "validator",
        {"check": "column_count", "column_count": column_count},
        "running",
    )
    if column_count > settings.max_column_count:
        log_tool_call(logger, "validator", {"check": "column_count"}, "failed: too many")
        return _fail(
            f"Too many columns ({column_count}). Max is {settings.max_column_count}. "
            "Please select relevant columns.",
            file_size_mb=round(size_mb, 2),
        )

    log_tool_call(logger, "validator", {"check": "column_count"}, "passed")

    # 8. ROW COUNT CHECK
    log_tool_call(
        logger,
        "validator",
        {"check": "row_count", "row_count": row_count},
        "running",
    )
    if row_count > settings.max_row_count:
        log_tool_call(logger, "validator", {"check": "row_count"}, "failed: too many rows")
        return _fail(
            f"Dataset has {row_count:,} rows which exceeds the {settings.max_row_count:,} "
            f"row limit. Please provide a sample: df.sample({settings.sample_size}) "
            "locally and re-upload.",
            file_size_mb=round(size_mb, 2),
        )

    log_tool_call(logger, "validator", {"check": "row_count"}, "passed")

    # 9. ALL-NULL CHECK
    log_tool_call(logger, "validator", {"check": "all_null"}, "running")
    if df.isna().all().all():
        log_tool_call(logger, "validator", {"check": "all_null"}, "failed: all null")
        return _fail(
            "Dataset appears entirely empty — all columns contain only null values.",
            file_size_mb=round(size_mb, 2),
        )

    log_tool_call(logger, "validator", {"check": "all_null"}, "passed")

    # 10. DUPLICATE COLUMN NAMES
    log_tool_call(logger, "validator", {"check": "duplicate_columns"}, "running")
    if df.columns.duplicated().any():
        duplicate_names = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
        logger.warning("Duplicate column names found, renaming: %s", duplicate_names)
        df = _rename_duplicate_columns(df)
        log_tool_call(
            logger,
            "validator",
            {"check": "duplicate_columns", "renamed": [str(c) for c in duplicate_names]},
            "renamed duplicates",
        )
    else:
        log_tool_call(logger, "validator", {"check": "duplicate_columns"}, "passed")

    # 11. SAMPLING
    log_tool_call(
        logger,
        "validator",
        {"check": "sampling", "row_count": row_count, "threshold": settings.sample_threshold},
        "running",
    )
    if row_count > settings.sample_threshold:
        df = df.sample(n=settings.sample_size, random_state=42)
        was_sampled = True
        sample_size = settings.sample_size
        log_tool_call(
            logger,
            "validator",
            {"check": "sampling", "sample_size": sample_size},
            "sampled",
        )
    else:
        was_sampled = False
        sample_size = None
        log_tool_call(logger, "validator", {"check": "sampling"}, "not needed")

    # 12. PASS
    result = ValidationResult(
        passed=True,
        error_message=None,
        row_count=row_count,
        column_count=column_count,
        file_size_mb=round(size_mb, 2),
        was_sampled=was_sampled,
        sample_size=sample_size,
    )
    log_tool_call(
        logger,
        "validator",
        {"check": "complete", "was_sampled": was_sampled},
        "passed",
    )

    return result, df
