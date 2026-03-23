from __future__ import annotations

import torch

from logger import CustomLogger, LogLevel


LOGGER = CustomLogger(level=LogLevel.DEFAULT)
DEVICE = torch.device("cpu")


def set_runtime(*, logger: CustomLogger | None = None, device: torch.device | None = None) -> None:
    global LOGGER, DEVICE
    if logger is not None:
        LOGGER.level = logger.level
    if device is not None:
        DEVICE = device
