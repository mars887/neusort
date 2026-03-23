from __future__ import annotations

from enum import Enum
from typing import TextIO


class LogLevel(Enum):
    QUIET = "quiet"
    ERROR = "error"
    DEFAULT = "default"
    DEBUG = "debug"


LEVEL_PRIORITY = {
    LogLevel.QUIET: 0,
    LogLevel.ERROR: 1,
    LogLevel.DEFAULT: 2,
    LogLevel.DEBUG: 3,
}


class CustomLogger:
    def __init__(self, level: LogLevel = LogLevel.DEFAULT):
        self.level = level

    def _priority(self, level: LogLevel | int) -> int:
        if isinstance(level, LogLevel):
            return LEVEL_PRIORITY[level]
        return int(level)

    def onlyLvl(self, level: LogLevel | int) -> bool:
        """Return True when the current level matches the requested one."""
        return LEVEL_PRIORITY[self.level] == self._priority(level)

    def lvlp(self, level: LogLevel | int) -> bool:
        """Return True when the current level is equal to or more verbose than requested."""
        return LEVEL_PRIORITY[self.level] >= self._priority(level)

    def info(self, message, end: str | None = None, file: TextIO | None = None):
        if self.lvlp(LogLevel.DEFAULT):
            if end is None and file is None:
                print(message)
            elif file is not None and end is None:
                print(message, file=file)
            elif file is None:
                print(message, end=end)
            else:
                print(message, end=end, file=file)

    def warning(self, message):
        if self.lvlp(LogLevel.DEFAULT):
            print(f"[WARN] {message}")

    def error(self, message):
        if self.lvlp(LogLevel.ERROR):
            print(f"! {message}")

    def debug(self, message):
        if self.lvlp(LogLevel.DEBUG):
            print(f"[DEBUG] {message}")
