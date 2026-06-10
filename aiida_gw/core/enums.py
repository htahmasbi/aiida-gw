from __future__ import annotations

from enum import Enum


EngineType = str


class CalculationMode(str, Enum):
    SINGLE_POINT = "single-point"
    RELAX = "relax"
    GW = "gw"

    def __str__(self) -> str:
        return self.value


RESOURCE_PRESETS: dict[str, dict] = {
    "default": {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 8,
    },
    "high_memory": {
        "num_machines": 2,
        "num_mpiprocs_per_machine": 8,
    },
    "gpu": {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 4,
    },
    "cp2k_large": {
        "num_machines": 4,
        "num_mpiprocs_per_machine": 16,
    },
}
