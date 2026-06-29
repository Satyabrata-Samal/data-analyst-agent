"""CSV DataFrame profiling utilities.

Builds a structured profile dict (shape, column stats, summaries, warnings)
used by the profiler node to inform downstream planning and code generation.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings
from utils.logger import get_logger, log_error, log_tool_call

logger = get_logger(__name__)


def _round_float(value: Any, decimals: int = 4) -> float:
    result = float(value)
    if np.isnan(result):
        raise ValueError("NaN value cannot be rounded")
    return round(result, decimals)


def _get_sample_values(series: pd.Series) -> list[str]:
    return [str(value) for value in series.dropna().head(3).tolist()]


def _build_column_profiles(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round(float(series.isna().mean() * 100), 2)

        columns.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "null_count": null_count,
                "null_pct": null_pct,
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": _get_sample_values(series),
            }
        )

    return columns


def _build_numeric_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    numeric_summary: dict[str, dict[str, float]] = {}

    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        try:
            described = series.describe()
            numeric_summary[str(col)] = {
                "mean": _round_float(described["mean"]),
                "median": _round_float(series.median()),
                "std": _round_float(described["std"]),
                "min": _round_float(described["min"]),
                "max": _round_float(described["max"]),
                "q25": _round_float(described["25%"]),
                "q75": _round_float(described["75%"]),
            }
        except Exception:
            continue

    return numeric_summary


def _build_categorical_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    categorical_summary: dict[str, dict[str, Any]] = {}

    for col in df.columns:
        series = df[col]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            continue

        value_counts = series.astype(str).value_counts().head(5)
        categorical_summary[str(col)] = {
            "top_5_values": {str(value): int(count) for value, count in value_counts.items()},
            "unique_count": int(series.nunique(dropna=True)),
        }

    return categorical_summary


def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    datetime_columns: list[str] = []

    for col in df.columns:
        series = df[col]
        if series.empty:
            continue

        try:
            parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        except Exception:
            continue

        parse_rate = parsed.notna().mean()
        if parse_rate > 0.5:
            datetime_columns.append(str(col))

    return datetime_columns


def _build_missing_summary(df: pd.DataFrame) -> dict[str, int | float]:
    total_missing = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells else 0.0

    return {
        "total_missing": total_missing,
        "total_cells": total_cells,
        "missing_pct": missing_pct,
    }


def _build_sample_rows(df: pd.DataFrame) -> list[dict[str, str]]:
    sample_rows: list[dict[str, str]] = []

    for row in df.head(5).to_dict(orient="records"):
        sample_rows.append({str(key): str(value) for key, value in row.items()})

    return sample_rows


def _build_warnings(df: pd.DataFrame, columns: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    row_count = len(df)

    for column in columns:
        name = column["name"]
        null_pct = column["null_pct"]
        unique_count = column["unique_count"]
        dtype = column["dtype"]

        if null_pct > 50:
            warnings.append(f"Column '{name}' has {null_pct}% missing values")

        if unique_count == 1:
            warnings.append(f"Column '{name}' has only one unique value")

        if unique_count == row_count and dtype == "object":
            warnings.append(f"Column '{name}' may be an ID column (all values unique)")

    return warnings


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Profile a DataFrame and return a JSON-serialisable summary dict."""
    log_tool_call(
        logger,
        "csv_profiler",
        {
            "rows": len(df),
            "columns": len(df.columns),
            "sample_threshold": settings.sample_threshold,
        },
        "starting",
    )

    try:
        columns = _build_column_profiles(df)
        profile = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": columns,
            "numeric_summary": _build_numeric_summary(df),
            "categorical_summary": _build_categorical_summary(df),
            "datetime_columns": _detect_datetime_columns(df),
            "missing_summary": _build_missing_summary(df),
            "sample_rows": _build_sample_rows(df),
            "warnings": _build_warnings(df, columns),
        }

        log_tool_call(
            logger,
            "csv_profiler",
            {"shape": profile["shape"], "warning_count": len(profile["warnings"])},
            json.dumps({"keys": list(profile.keys())}),
        )

        return profile
    except Exception as exc:
        log_error(logger, "csv_profiler", exc)
        raise
