# app/tests/test_code_executor.py

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from app.tools.code_executor import (
    ExecutionResult,
    _blocked_message,
    _build_script,
    _find_open_write_mode,
    _truncate_stdout,
    execute_code,
    static_analysis,
)


# -----------------------------------------------------------------------------
# _blocked_message
# -----------------------------------------------------------------------------

def test_blocked_message():
    assert _blocked_message("eval(") == (
        "Blocked: generated code contains 'eval(' which is not permitted."
    )


# -----------------------------------------------------------------------------
# _find_open_write_mode
# -----------------------------------------------------------------------------

def test_find_open_write_mode_write():
    code = "f = open('a.txt', 'w')"
    assert _find_open_write_mode(code) == "open("


def test_find_open_write_mode_append():
    code = 'f = open("a.txt", "a")'
    assert _find_open_write_mode(code) == "open("


def test_find_open_write_mode_read():
    code = "f = open('a.txt', 'r')"
    assert _find_open_write_mode(code) is None


# -----------------------------------------------------------------------------
# static_analysis
# -----------------------------------------------------------------------------

def test_static_analysis_safe():
    ok, reason = static_analysis("print(df.head())")

    assert ok is True
    assert reason is None


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess",
        "os.system('ls')",
        "eval('1+1')",
        "__import__('os')",
        "requests.get('http://x.com')",
    ],
)
def test_static_analysis_blocked_patterns(code):
    ok, reason = static_analysis(code)

    assert ok is False
    assert "Blocked" in reason


def test_static_analysis_open_write():
    ok, reason = static_analysis("open('x.txt', 'w')")

    assert ok is False
    assert "open(" in reason


def test_static_analysis_syntax_error():
    ok, reason = static_analysis("for")

    assert ok is False
    assert reason.startswith("SyntaxError")


# -----------------------------------------------------------------------------
# _build_script
# -----------------------------------------------------------------------------

def test_build_script():
    script = _build_script("print(df.head())", "/tmp/data.csv")

    assert "pd.read_csv" in script
    assert "/tmp/data.csv" in script
    assert "print(df.head())" in script


# -----------------------------------------------------------------------------
# _truncate_stdout
# -----------------------------------------------------------------------------

@patch("app.tools.code_executor.settings")
def test_truncate_stdout_no_truncate(mock_settings):
    mock_settings.max_stdout_bytes = 100

    out, truncated = _truncate_stdout("hello")

    assert out == "hello"
    assert truncated is False


@patch("app.tools.code_executor.settings")
def test_truncate_stdout(mock_settings):
    mock_settings.max_stdout_bytes = 5

    out, truncated = _truncate_stdout("abcdefghijklmnopqrstuvwxyz")

    assert truncated is True
    assert len(out.encode()) == 5


# -----------------------------------------------------------------------------
# execute_code()
# -----------------------------------------------------------------------------

@patch("app.tools.code_executor.subprocess.run")
def test_execute_code_success(mock_run):
    process = MagicMock()
    process.returncode = 0
    process.stdout = "Success!"
    process.stderr = ""

    mock_run.return_value = process

    result = execute_code(
        "print('Success!')",
        "/tmp/data.csv",
    )

    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.stdout == "Success!"
    assert result.stderr == ""
    assert result.error_type is None


def test_execute_code_static_analysis_failure():
    result = execute_code(
        "import subprocess",
        "/tmp/data.csv",
    )

    assert result.success is False
    assert result.error_type == "static_analysis"
    assert "Blocked" in result.stderr


@patch("app.tools.code_executor.subprocess.run")
def test_execute_code_runtime_error(mock_run):
    process = MagicMock()

    process.returncode = 1
    process.stdout = ""
    process.stderr = "ZeroDivisionError"

    mock_run.return_value = process

    result = execute_code(
        "1/0",
        "/tmp/data.csv",
    )

    assert result.success is False
    assert result.error_type == "runtime"
    assert "ZeroDivisionError" in result.stderr


@patch("app.tools.code_executor.subprocess.run")
def test_execute_code_timeout(mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(
        cmd="python",
        timeout=5,
    )

    result = execute_code(
        "while True: pass",
        "/tmp/data.csv",
    )

    assert result.success is False
    assert result.error_type == "timeout"


@patch("app.tools.code_executor.subprocess.run")
def test_execute_code_unexpected_exception(mock_run):
    mock_run.side_effect = RuntimeError("boom")

    result = execute_code(
        "print(1)",
        "/tmp/data.csv",
    )

    assert result.success is False
    assert result.error_type == "runtime"
    assert "boom" in result.stderr