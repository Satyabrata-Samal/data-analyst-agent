"""Structured logging utilities for the data analysis agent.

Provides a configured logger with console and file output, plus helpers that
emit consistent log lines and return formatted strings for appending to
``agent_log`` in graph state.
"""

import logging
from typing import Any

from app.config.settings import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with console and file handlers attached."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = _build_formatter()

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    return logger


def log_node_entry(logger: logging.Logger, node_name: str, state_summary: dict[str, Any]) -> str:
    """Log node entry and return a line for ``agent_log``."""
    line = f"NODE ENTER | {node_name} | {state_summary}"
    logger.info(line)
    return line


def log_node_exit(logger: logging.Logger, node_name: str, output_summary: dict[str, Any]) -> str:
    """Log node exit and return a line for ``agent_log``."""
    line = f"NODE EXIT | {node_name} | {output_summary}"
    logger.info(line)
    return line


def log_tool_call(
    logger: logging.Logger,
    tool_name: str,
    inputs: dict[str, Any],
    result_summary: str,
) -> str:
    """Log a tool invocation and return a line for ``agent_log``."""
    line = f"TOOL CALL | {tool_name} | inputs={inputs} | result={result_summary}"
    logger.info(line)
    return line


def log_error(logger: logging.Logger, node_name: str, error: Exception) -> str:
    """Log an error at ERROR level and return a line for ``agent_log``."""
    line = f"ERROR | {node_name} | {type(error).__name__}: {error}"
    logger.error(line)
    return line
