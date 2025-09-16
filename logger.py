from enum import Enum

class LogLevel(Enum):
    QUIET = "quiet"
    ERROR = "error"
    DEFAULT = "default"
    DEBUG = "debug"

class CustomLogger:
    def __init__(self, level: LogLevel = LogLevel.DEFAULT):
        self.level = level

    def info(self, message, end = None,file = None):
        if self.level in [LogLevel.DEBUG, LogLevel.DEFAULT]:
            if end is None and file is None:
                print(message)
            elif file is not None:
                print(message,file=file)
            else:
                print(message,end=end)

    def error(self, message):
        if self.level in [LogLevel.DEFAULT, LogLevel.ERROR, LogLevel.DEBUG]:
            print(f"! {message}")

    def debug(self, message):
        if self.level == LogLevel.DEBUG:
            print(f"[DEBUG] {message}")
