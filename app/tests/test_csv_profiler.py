# tests/test_csv_profiler.py

import numpy as np
import pandas as pd
import pytest

from app.tools.csv_profiler import (
    _to_python_int,
    _round_float,
    _get_sample_values,
    _build_column_profiles,
    _build_numeric_summary,
    _build_categorical_summary,
    _detect_datetime_columns,
    _build_missing_summary,
    _build_sample_rows,
    _build_warnings,
    profile_dataframe,
)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def test_to_python_int():
    assert _to_python_int(np.int64(5)) == 5
    assert isinstance(_to_python_int(np.int64(5)), int)


def test_round_float():
    assert _round_float(3.1415926) == 3.1416
    assert _round_float(2.123456, 2) == 2.12


def test_round_float_nan():
    with pytest.raises(ValueError):
        _round_float(np.nan)


def test_get_sample_values():
    s = pd.Series([1, np.nan, 3, 4])
    assert _get_sample_values(s) == ["1.0", "3.0", "4.0"]


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [20, 30, 40, np.nan],
            "salary": [100, 200, 300, 400],
            "city": pd.Series(["NY", "LA", "NY", "SF"], dtype=object),
            "id": pd.Series(["a", "b", "c", "d"], dtype=object),
            "constant": pd.Series(["x", "x", "x", "x"], dtype=object),
            "date": pd.Series(
                ["2024-01-01", "2024-01-02", "2024-01-03", None],
                dtype=object,
            ),
        }
    )


# -----------------------------------------------------------------------------
# Column profile
# -----------------------------------------------------------------------------

def test_build_column_profiles(sample_df):
    profiles = _build_column_profiles(sample_df)

    assert len(profiles) == 6

    age = next(c for c in profiles if c["name"] == "age")

    assert age["null_count"] == 1
    assert age["unique_count"] == 3
    assert age["null_pct"] == 25.0
    assert age["sample_values"] == ["20.0", "30.0", "40.0"]


# -----------------------------------------------------------------------------
# Numeric summary
# -----------------------------------------------------------------------------

def test_build_numeric_summary(sample_df):
    summary = _build_numeric_summary(sample_df)

    assert "age" in summary
    assert "salary" in summary

    assert summary["salary"]["mean"] == 250.0
    assert summary["salary"]["min"] == 100.0
    assert summary["salary"]["max"] == 400.0


# -----------------------------------------------------------------------------
# Categorical summary
# -----------------------------------------------------------------------------

def test_build_categorical_summary(sample_df):
    summary = _build_categorical_summary(sample_df)

    assert "city" in summary
    assert summary["city"]["unique_count"] == 3

    top = summary["city"]["top_5_values"]

    assert top["NY"] == 2
    assert top["LA"] == 1


# -----------------------------------------------------------------------------
# Datetime detection
# -----------------------------------------------------------------------------

def test_detect_datetime_columns(sample_df):
    cols = _detect_datetime_columns(sample_df)

    assert "date" in cols
    assert "city" not in cols


# -----------------------------------------------------------------------------
# Missing summary
# -----------------------------------------------------------------------------

def test_build_missing_summary(sample_df):
    summary = _build_missing_summary(sample_df)

    assert summary["total_missing"] == 2
    assert summary["total_cells"] == 24
    assert summary["missing_pct"] == round((2 / 24) * 100, 2)


# -----------------------------------------------------------------------------
# Sample rows
# -----------------------------------------------------------------------------

def test_build_sample_rows(sample_df):
    rows = _build_sample_rows(sample_df)

    assert len(rows) == 4
    assert rows[0]["age"] == "20.0"
    assert rows[0]["city"] == "NY"


# -----------------------------------------------------------------------------
# Warnings
# -----------------------------------------------------------------------------

def test_build_warnings(sample_df):
    cols = _build_column_profiles(sample_df)
    warnings = _build_warnings(sample_df, cols)

    assert any("constant" in w for w in warnings)
    assert any("id" in w for w in warnings)


def test_missing_warning():
    df = pd.DataFrame(
        {
            "a": [1, None, None, None]
        }
    )

    cols = _build_column_profiles(df)
    warnings = _build_warnings(df, cols)

    assert any("missing values" in w for w in warnings)


# -----------------------------------------------------------------------------
# profile_dataframe()
# -----------------------------------------------------------------------------

def test_profile_dataframe(sample_df):
    profile = profile_dataframe(sample_df)

    assert profile["shape"] == {
        "rows": 4,
        "columns": 6,
    }

    assert len(profile["columns"]) == 6

    assert "numeric_summary" in profile
    assert "categorical_summary" in profile
    assert "missing_summary" in profile
    assert "sample_rows" in profile
    assert "warnings" in profile

    assert "date" in profile["datetime_columns"]


def test_profile_dataframe_empty():
    df = pd.DataFrame()

    profile = profile_dataframe(df)

    assert profile["shape"] == {
        "rows": 0,
        "columns": 0,
    }

    assert profile["columns"] == []
    assert profile["warnings"] == []