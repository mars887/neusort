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
            if end == "":
                print(message)
            elif file != None:
                print(message,file)
            else:
                print(message,end)

    def error(self, message):
        if self.level in [LogLevel.DEFAULT, LogLevel.ERROR, LogLevel.DEBUG]:
            print(f"! {message}")

    def debug(self, message):
        if self.level == LogLevel.DEBUG:
            print(f"[DEBUG] {message}")
