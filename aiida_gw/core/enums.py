from __future__ import annotations

from enum import Enum


EngineType = str


class CalculationMode(str, Enum):
    SINGLE_POINT = "single-point"
    RELAX = "relax"
    GW = "gw"

    def __str__(self) -> str:
        return self.value



