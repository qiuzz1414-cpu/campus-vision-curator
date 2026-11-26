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
        # 颜色必须整体开关。早期版本无条件拼接 color、只对 reset 判断 _use_color()，
        # 结果重定向到文件或设置 NO_COLOR 时每行都带一段永不闭合的 ANSI 转义码。
        use_color = _use_color()
        color = _COLORS.get(record.levelname, "") if use_color else ""
        reset = _COLORS["RESET"] if color else ""
        prefix = f"{color}{record.levelname:<7}{reset}"
        msg = f"{prefix} {record.getMessage()}"

        # 必须自己带上堆栈：整个 format 被覆盖后，基类里渲染 exc_info 的逻辑
        # 不再被调用，logger.exception() 会静默丢掉 traceback。
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            msg = f"{msg}\n{self.formatStack(record.stack_info)}"
        return msg


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
