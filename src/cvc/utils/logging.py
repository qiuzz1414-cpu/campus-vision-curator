"""统一日志：带阶段前缀、颜色、可关闭。"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "RESET": "\033[0m",
}


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        reset = _COLORS["RESET"] if color and _use_color() else ""
        prefix = f"{color}{record.levelname:<7}{reset}"
        return f"{prefix} {record.getMessage()}"


def _use_color() -> bool:
    return sys.stderr.isatty() and os.getenv("NO_COLOR") is None


def get_logger(stage: str = "cvc") -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("cvc")
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_Formatter())
        root.addHandler(handler)
        root.setLevel(os.getenv("CVC_LOG_LEVEL", "INFO").upper())
        root.propagate = False
        _CONFIGURED = True
    return root.getChild(stage)


def banner(text: str) -> None:
    """阶段开始时的分隔标题。"""
    line = "─" * 60
    print(f"\n{line}\n  {text}\n{line}", file=sys.stderr)
