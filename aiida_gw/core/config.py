from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class MetadataOptions(BaseModel):
    num_machines: int = Field(default=1, ge=1)
    num_mpiprocs_per_machine: int = Field(default=8, ge=1)
    max_wallclock_seconds: int = Field(default=36000, ge=60)
    withmpi: bool = True
    cpus_per_task: int | None = Field(
        default=None,
        ge=1,
        description="Number of CPUs per task. When set, passed to the scheduler as #SBATCH --cpus-per-task=.",
    )
    memory_per_machine: str | None = Field(
        default=None,
        description="Memory per node (e.g. \"600G\" or \"38400M\"). Passed to the scheduler as #SBATCH --mem=.",
    )
    partition: str | None = Field(
        default=None,
        description="SLURM partition (queue). When set, passed as #SBATCH --partition=.",
    )

    @field_validator("partition")
    @classmethod
    def validate_partition(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"Invalid partition name {v!r} — only alphanumeric, "
                "dashes (-), and underscores (_) are allowed."
            )
        return v

    @field_validator("memory_per_machine")
    @classmethod
    def validate_memory_format(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not v:
            return v
        if v.isdigit():
            return v
        unit = v[-1].upper()
        if unit not in ("G", "M", "K"):
            raise ValueError(
                f"memory_per_machine must be a number (e.g. '38400' = MB) or end with G/M/K (e.g. '600G'), got {v!r}"
            )
        if not v[:-1].isdigit():
            raise ValueError(
                f"memory_per_machine must be a number (e.g. '38400' = MB) or end with G/M/K (e.g. '600G'), got {v!r}"
            )
        return v

    def to_dict(self) -> dict:
        result: dict[str, Any] = {
            "resources": {
                "num_machines": self.num_machines,
                "num_mpiprocs_per_machine": self.num_mpiprocs_per_machine,
            },
            "max_wallclock_seconds": self.max_wallclock_seconds,
            "withmpi": self.withmpi,
        }
        scheduler_lines = []
        if self.cpus_per_task is not None:
            scheduler_lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        if self.memory_per_machine is not None:
            scheduler_lines.append(f"#SBATCH --mem={self.memory_per_machine}")
        if self.partition:
            scheduler_lines.append(f"#SBATCH --partition={self.partition}")
        if scheduler_lines:
            result["custom_scheduler_commands"] = "\n".join(scheduler_lines) + "\n"
        return result


class Cp2kScfConfig(BaseModel):
    max_scf: int = Field(default=200, ge=1)
    eps_scf: float = Field(default=1e-6, gt=0)
    mixing_alpha: float = Field(default=0.2)
    mixing_beta: float = Field(default=0.8)
    mixing_nbroyden: int = Field(default=10)


class Cp2kConfig(BaseModel):
    cutoff: int = Field(default=400, ge=0)
    rel_cutoff: int = Field(default=50, ge=0)
    kpoints_mesh: list[int] | None = Field(
        default=None,
        description="Explicit k-point mesh (e.g. [4,1,4]). When set, overrides kpoints_distance.",
    )
    kpoints_distance: float | None = Field(
        default=None,
        description="Target k-point mesh density (1/Å). Overrides protocol kpoints_distance.",
    )
    basis_set_file: str = "BASIS_MOLOPT"
    potential_file: str = "GTH_POTENTIALS"
    scf: Cp2kScfConfig = Field(default_factory=Cp2kScfConfig)

    @field_validator("kpoints_mesh")
    @classmethod
    def validate_kpoints_mesh(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        if len(v) != 3:
            raise ValueError("kpoints_mesh must have exactly 3 elements")
        if any(x <= 0 for x in v):
            raise ValueError("kpoints_mesh elements must be positive")
        return v


class ElementOverride(BaseModel):
    orb_basis: str | None = Field(
        default=None,
        description="Orbital basis set name for this element (overrides auto-resolution)",
    )
    potential: str | None = Field(
        default=None,
        description="Pseudopotential name for this element (overrides auto-resolution)",
    )
    ri_basis: str | None = Field(
        default=None,
        description="RI auxiliary basis name for this element (overrides auto-resolution)",
    )


class GwConfig(BaseModel):
    kpoints_mesh: list[int] | None = Field(
        default=None,
        description="Explicit k-point mesh for GW DFT (e.g. [6,1,6]). When set, overrides kpoints_distance.",
    )
    kpoints_w_mesh: list[int] | None = Field(
        default=None,
        description="Explicit k-point mesh for GW correction (KPOINTS_W). When set, overrides kpoints_w_distance.",
    )
    kpoints_distance: float | None = Field(
        default=None,
        description="Target k-point mesh density (1/Å) for GW DFT. Overrides protocol kpoints_distance.",
    )
    kpoints_w_distance: float | None = Field(
        default=None,
        description="Target k-point mesh density (1/Å) for GW correction (KPOINTS_W). Overrides protocol kpoints_w_distance.",
    )
    periodic: str = Field(
        default="XZ",
        description="Periodicity for the Poisson solver (DFT.POISSON.PERIODIC). "
        "Typically \"XZ\" for 2D materials with vacuum in Y, or \"XYZ\" for bulk.",
    )
    cell_periodic: str = Field(
        default="XZ",
        description="Cell periodicity (SUBSYS.CELL.PERIODIC). "
        "Typically \"XZ\" for 2D materials with vacuum in Y.",
    )
    poisson_solver: str = Field(default="PERIODIC")
    cutoff: int = Field(default=600, ge=50)
    rel_cutoff: int = Field(default=100, ge=0)
    eps_default: float | None = Field(
        default=None,
        description="QS.EPS_DEFAULT override. When None, the protocol YAML value is used.",
    )
    eps_pgf_orb: float | None = Field(
        default=None,
        description="QS.EPS_PGF_ORB override. When None, the protocol YAML value is used.",
    )
    eps_scf: float | None = Field(
        default=None,
        description="SCF.EPS_SCF override. When None, the protocol YAML value is used.",
    )
    max_scf: int | None = Field(
        default=None,
        description="SCF.MAX_SCF override. When None, the protocol YAML value is used.",
    )
    mixing_alpha: float | None = Field(
        default=None,
        description="MIXING.ALPHA override. When None, the protocol YAML value is used.",
    )
    mixing_beta: float | None = Field(
        default=None,
        description="MIXING.BETA override. When None, the protocol YAML value is used.",
    )
    mixing_nbroyden: int | None = Field(
        default=None,
        description="MIXING.NBROYDEN override. When None, the protocol YAML value is used.",
    )
    num_time_freq: int = Field(default=20, ge=1)
    memory_per_proc: int = Field(
        default=8,
        ge=1,
        description="Memory per MPI process for the GW run, in GB. Set to node_memory / MPI_per_node (e.g. 1000/128 ≈ 8). Too large = no parallelization, OOM. Too small = poor performance.",
    )
    eps_filter: float = Field(default=1.0e-6, gt=0)
    cutoff_radius_ri: int = Field(default=5, ge=1)
    regularization_ri: float = Field(default=0.01, gt=0)
    vacuum: float = Field(default=20.0, ge=5.0)
    supercell: list[int] = Field(default_factory=lambda: [3, 3, 1])
    bs_npoints: int = Field(default=20, ge=1)
    basis_set_file: str = Field(
        default="BASIS_GTH_MOLOPT_AUG_for_excited_states",
        description="Path to augmented basis set file (on cluster)",
    )
    ri_basis_set_file: str = Field(
        default="BASIS_GTH_MOLOPT_AUG_for_excited_states_RI",
        description="Path to RI auxiliary basis set file (on cluster)",
    )
    potential_file: str = Field(
        default="POTENTIAL_UZH",
        description="Path to CP2K potential file (on cluster)",
    )
    orb_basis: str = Field(default="aug-SZV-MOLOPT-GTH-tier-1")
    resolve_from_files: bool = Field(
        default=True,
        description="Resolve orbital/potential/RI from data files instead of YAML atom_data",
    )
    ri_basis_accuracy_target: float | None = Field(
        default=None,
        description="Target accuracy for RI basis selection. Picks entry closest to this error. None = most accurate (smallest error)",
    )
    ri_basis: str | None = None
    potential: str | None = None
    xc_functional: str | None = Field(
        default=None,
        description="XC functional name for potential selection (e.g. \"PBE\", \"OLYP\"). "
        "When set, overrides the XC functional from the protocol YAML when resolving "
        "potential names from the potential file.",
    )
    element_settings: dict[str, ElementOverride] = Field(
        default_factory=dict,
        description="Per-element overrides for basis set, potential, and RI auxiliary basis. "
        "Keys are element symbols (e.g. 'B', 'N'), values are ElementOverride with optional orb_basis, potential, ri_basis.",
    )
    cube_files: list[str] = Field(
        default_factory=lambda: ["V_HARTREE_CUBE", "E_DENSITY_CUBE"],
        description="List of CP2K PRINT cube-section names to write and retrieve. "
        "Common values: 'V_HARTREE_CUBE' (Hartree potential for band alignment), "
        "'E_DENSITY_CUBE' (total charge density, large). "
        "Empty list disables all cube output.",
    )


class ProjectConfig(BaseSettings):
    model_config = {"env_nested_delimiter": "__", "extra": "ignore"}

    code_label: str = Field(default="cp2k@localhost")
    aiida_profile: str | None = None
    cp2k: Cp2kConfig = Field(default_factory=Cp2kConfig)
    gw: GwConfig = Field(default_factory=GwConfig)
    metadata_options: MetadataOptions = Field(default_factory=MetadataOptions)


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    config_data: dict[str, Any] = {}

    if config_path is None:
        search_paths = [
            Path.cwd() / "config.toml",
            Path(__file__).parent.parent / "config.toml",
            Path.home() / ".config" / "aiida-gw" / "config.toml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = path
                break

    if config_path and Path(config_path).exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
        if "default" in config_data:
            defaults = config_data.pop("default")
            for k, v in defaults.items():
                config_data.setdefault(k, v)
        logger.info(f"Loaded configuration from {config_path}")

    return ProjectConfig(**config_data)


_config: ProjectConfig | None = None
_config_lock = threading.Lock()


def get_config() -> ProjectConfig:
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:  # double-check under lock
                _config = load_config()
    return _config


def reset_config() -> None:
    global _config
    with _config_lock:
        _config = None
