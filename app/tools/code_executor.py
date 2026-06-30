"""Safe subprocess execution of generated analysis code.

Runs static analysis to block dangerous patterns, prepends a DataFrame loader,
executes code in an isolated subprocess with timeout and stdout limits, and
returns a structured :class:`ExecutionResult`.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass

from app.config.settings import settings
from app.utils.logger import get_logger, log_error, log_tool_call

logger = get_logger(__name__)

_BLOCKED_PATTERNS: tuple[str, ...] = (
    "os.system",
    "os.popen",
    "os.remove",
    "os.rmdir",
    "shutil",
    "subprocess",
    "requests",
    "urllib",
    "httpx",
    "socket",
    "eval(",
    "exec(",
    "__import__",
    "pickle",
)

_WRITE_MODES: tuple[str, ...] = ("'w'", "'a'", '"w"', '"a"')


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    truncated: bool
    error_type: str | None


def _blocked_message(pattern: str) -> str:
    return f"Blocked: generated code contains '{pattern}' which is not permitted."


def _find_open_write_mode(code: str) -> str | None:
    start = 0
    while True:
        index = code.find("open(", start)
        if index == -1:
            return None

        window = code[index : index + 100]
        for mode in _WRITE_MODES:
            if mode in window:
                return "open("

        start = index + 1


def static_analysis(code: str) -> tuple[bool, str | None]:
    """Scan generated code for dangerous patterns before execution."""
    for pattern in _BLOCKED_PATTERNS:
        if pattern in code:
            reason = _blocked_message(pattern)
            log_tool_call(
                logger,
                "code_executor",
                {"check": "static_analysis", "pattern": pattern},
                "blocked",
            )
            return False, reason

    if _find_open_write_mode(code) is not None:
        reason = _blocked_message("open(")
        log_tool_call(
            logger,
            "code_executor",
            {"check": "static_analysis", "pattern": "open("},
            "blocked",
        )
        return False, reason

    try:
        ast.parse(code)
    except SyntaxError as exc:
        reason = f"SyntaxError: {exc}"
        log_tool_call(
            logger,
            "code_executor",
            {"check": "static_analysis", "error": str(exc)},
            "syntax_error",
        )
        return False, reason

    log_tool_call(
        logger,
        "code_executor",
        {"check": "static_analysis"},
        "passed",
    )
    return True, None


def _build_script(code: str, df_csv_path: str) -> str:
    preamble = textwrap.dedent(
        f"""\
        import pandas as pd
        import numpy as np
        df = pd.read_csv({df_csv_path!r})
        """
    )
    return f"{preamble}\n{code}"


def _truncate_stdout(stdout: str) -> tuple[str, bool]:
    encoded = stdout.encode()
    if len(encoded) <= settings.max_stdout_bytes:
        return stdout, False

    truncated = encoded[: settings.max_stdout_bytes].decode(errors="replace")
    return truncated, True


def execute_code(code: str, df_csv_path: str) -> ExecutionResult:
    """Validate, run, and capture output from generated analysis code."""
    is_safe, blocked_reason = static_analysis(code)
    if not is_safe:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=blocked_reason or "Static analysis failed.",
            truncated=False,
            error_type="static_analysis",
        )

    tmp_file_path: str | None = None

    try:
        script = _build_script(code, df_csv_path)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as tmp_file:
            tmp_file.write(script)
            tmp_file_path = tmp_file.name

        log_tool_call(
            logger,
            "code_executor",
            {
                "check": "execute",
                "csv_path": df_csv_path,
                "timeout": settings.code_execution_timeout,
            },
            "starting",
        )

        result = subprocess.run(
            [sys.executable, tmp_file_path],
            timeout=settings.code_execution_timeout,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )

        if result.returncode == 0:
            stdout, truncated = _truncate_stdout(result.stdout)
            execution_result = ExecutionResult(
                success=True,
                stdout=stdout,
                stderr="",
                truncated=truncated,
                error_type=None,
            )
            log_tool_call(
                logger,
                "code_executor",
                {
                    "success": True,
                    "truncated": truncated,
                    "stdout_length": len(stdout),
                },
                "complete",
            )
            return execution_result

        execution_result = ExecutionResult(
            success=False,
            stdout="",
            stderr=result.stderr,
            truncated=False,
            error_type="runtime",
        )
        log_tool_call(
            logger,
            "code_executor",
            {"success": False, "returncode": result.returncode},
            "runtime_error",
        )
        return execution_result

    except subprocess.TimeoutExpired:
        log_tool_call(
            logger,
            "code_executor",
            {"timeout": settings.code_execution_timeout},
            "timeout",
        )
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Execution timed out after {settings.code_execution_timeout}s",
            truncated=False,
            error_type="timeout",
        )
    except Exception as exc:
        log_error(logger, "code_executor", exc)
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=str(exc),
            truncated=False,
            error_type="runtime",
        )
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
