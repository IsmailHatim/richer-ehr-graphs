"""Structlog configuration shared by every CLI entry point."""

from __future__ import annotations

import logging

import structlog


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structlog with a colored console renderer."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
