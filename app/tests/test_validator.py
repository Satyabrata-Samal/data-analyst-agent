"""Unit tests for app.agent.validator.validate_and_load — covers all 11 gates."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.agent.validator import _rename_duplicate_columns, validate_and_load


@pytest.fixture
def write_csv(tmp_path: Path):
    """Return a helper that writes a CSV file and returns its path."""

    def _write(name: str, content: str) -> str:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    return _write


# --- happy path ---


def test_validate_happy_path(write_csv):
    path = write_csv("ok.csv", "a,b\n1,2\n3,4\n")
    result, df = validate_and_load(path)

    assert result.passed is True
    assert result.row_count == 2
    assert result.column_count == 2
    assert df is not None
    assert list(df.columns) == ["a", "b"]


# --- gate 1: file exists ---


def test_validate_missing_file(tmp_path):
    result, df = validate_and_load(str(tmp_path / "nope.csv"))

    assert result.passed is False
    assert "not found" in result.error_message.lower()
    assert df is None


# --- gate 2: extension ---


def test_validate_wrong_extension(write_csv):
    path = write_csv("data.tsv", "a\tb\n1\t2\n")
    result, df = validate_and_load(path)

    assert result.passed is False
    assert ".csv" in result.error_message
    assert df is None


# --- gate 3: file size ---


def test_validate_file_too_large(write_csv, monkeypatch):
    from app.agent import validator

    monkeypatch.setattr(validator.settings, "max_file_size_mb", 0.00001)
    path = write_csv("big.csv", "a,b\n" + "1,2\n" * 1000)

    result, df = validate_and_load(path)

    assert result.passed is False
    assert "exceeds limit" in result.error_message
    assert df is None


# --- gate 4/5: parse / full load ---


def test_validate_malformed_csv(write_csv):
    path = write_csv("bad.csv", '"unterminated,quote\nfoo,bar\n')
    result, df = validate_and_load(path)

    assert result.passed is False
    assert "parse" in result.error_message.lower() or "could not" in result.error_message.lower()


# --- gate 6: empty ---


def test_validate_empty_rows(write_csv):
    path = write_csv("empty.csv", "a,b\n")
    result, df = validate_and_load(path)

    assert result.passed is False
    assert "no data" in result.error_message.lower()


# --- gate 7: column count ---


def test_validate_too_many_columns(write_csv, monkeypatch):
    from app.agent import validator

    monkeypatch.setattr(validator.settings, "max_column_count", 2)
    path = write_csv("wide.csv", "a,b,c\n1,2,3\n")

    result, df = validate_and_load(path)

    assert result.passed is False
    assert "Too many columns" in result.error_message


# --- gate 8: row count ---


def test_validate_too_many_rows(write_csv, monkeypatch):
    from app.agent import validator

    monkeypatch.setattr(validator.settings, "max_row_count", 2)
    path = write_csv("tall.csv", "a\n" + "1\n" * 5)

    result, df = validate_and_load(path)

    assert result.passed is False
    assert "exceeds" in result.error_message


# --- gate 9: all null ---


def test_validate_all_null(write_csv):
    path = write_csv("nulls.csv", "a,b\n,\n,\n")
    result, df = validate_and_load(path)

    assert result.passed is False
    assert "empty" in result.error_message.lower() or "null" in result.error_message.lower()


# --- gate 10: duplicate columns (auto-renamed, not rejected) ---


def test_validate_duplicate_columns_are_handled(write_csv):
    # pandas.read_csv auto-suffixes duplicate headers to `a.1`, so the file
    # never reaches the validator with true duplicates. This test just
    # confirms validation still passes and both columns are preserved.
    path = write_csv("dup.csv", "a,a\n1,2\n3,4\n")
    result, df = validate_and_load(path)

    assert result.passed is True
    assert df is not None
    assert len(df.columns) == 2


# --- helper: _rename_duplicate_columns ---


def test_rename_duplicate_columns():
    import pandas as pd

    df = pd.DataFrame([[1, 2, 3]], columns=["x", "x", "x"])
    renamed = _rename_duplicate_columns(df)

    assert list(renamed.columns) == ["x", "x_1", "x_2"]
