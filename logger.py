from enum import Enum

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

    def onlyLvl(self, level: LogLevel) -> bool:
        """Return True when the current level matches the requested one."""
        return self.level == level

    def lvlp(self, level: LogLevel) -> bool:
        """Return True when the current level is equal to or more verbose than requested."""
        return LEVEL_PRIORITY.get(self.level) >= LEVEL_PRIORITY.get(level)
    
    def onlyLvl(self, level: int) -> bool:
        """Return True when the current level matches the requested one."""
        return LEVEL_PRIORITY.get(self.level) == level

    def lvlp(self, level: int) -> bool:
        """Return True when the current level is equal to or more verbose than requested."""
        return LEVEL_PRIORITY.get(self.level) >= level

    def info(self, message, end = None,file = None):
        if self.lvlp(2):
            if end is None and file is None:
                print(message)
            elif file is not None:
                print(message,file=file)
            else:
                print(message,end=end)

    def error(self, message):
        if self.lvlp(1):
            print(f"! {message}")

    def debug(self, message):
        if self.lvlp(3):
            print(f"[DEBUG] {message}")
