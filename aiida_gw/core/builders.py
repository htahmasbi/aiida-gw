from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from aiida.orm import Code, Dict, KpointsData, SinglefileData, StructureData
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_gw.core.config import ProjectConfig

logger = logging.getLogger(__name__)

StructureData = DataFactory("core.structure")
Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")


def get_cp2k_files_path() -> Path:
    """Return the path to the CP2K files directory."""
    return Path(__file__).parent.parent / "codes" / "cp2k" / "cp2k_files"


def load_protocol(protocol_name: str = "protocol_SIRIUS.yml") -> dict:
    """Load a CP2K protocol YAML file from cp2k_files."""
    path = get_cp2k_files_path() / protocol_name
    if not path.exists():
        raise FileNotFoundError(f"Protocol file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_kinds_section_qs(structure, atom_data: dict) -> dict:
    """Build &KIND section for QUICKSTEP."""
    kinds = []
    ase = structure.get_ase()
    symbols = ase.get_chemical_symbols()
    seen = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        kind = {
            "_": symbol,
            "BASIS_SET": atom_data["basis_set"][symbol],
            "POTENTIAL": atom_data["pseudopotential"][symbol],
        }
        kinds.append(kind)
    return {"FORCE_EVAL": {"SUBSYS": {"KIND": kinds}}}


def get_kinds_section_sirius(structure, atom_data: dict) -> dict:
    """Build &KIND section for SIRIUS with UPF potentials."""
    kinds = []
    ase = structure.get_ase()
    symbols = ase.get_chemical_symbols()
    seen = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        filename = Path(atom_data[symbol]["filename"]).stem + ".json"
        kind = {
            "_": symbol,
            "POTENTIAL": f"UPF {filename}",
        }
        kinds.append(kind)
    return {"FORCE_EVAL": {"SUBSYS": {"KIND": kinds}}}


def get_file_section_qs() -> dict:
    """Get CP2K file resources for QUICKSTEP."""
    cp2k_files = get_cp2k_files_path()
    files = {}
    for fname in ["GTH_POTENTIALS", "GTH_BASIS_SETS", "BASIS_MOLOPT", "BASIS_MOLOPT_UCL"]:
        path = cp2k_files / fname
        if path.exists():
            with open(path, "rb") as fh:
                files[fname] = SinglefileData(file=fh)
    return files


def get_file_section_sirius(structure, atom_data: dict) -> dict:
    """Get pseudopotential files for SIRIUS."""
    cp2k_files = get_cp2k_files_path() / "pseudopotentials"
    files = {}
    ase = structure.get_ase()
    symbols = ase.get_chemical_symbols()
    seen = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        filename = atom_data[symbol]["filename"]
        path = cp2k_files / Path(filename).stem + ".json"
        if path.exists():
            with open(path, "rb") as fh:
                files[symbol] = SinglefileData(file=fh)
    return files


def dict_merge(base: dict, merge: dict) -> dict:
    """Recursively merge merge into base."""
    for k, v in merge.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            dict_merge(base[k], v)
        else:
            base[k] = v
    return base


class Cp2kBuilder:
    """Build inputs for CP2K workchains using protocol files."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def build_scf_inputs(
        self,
        structure: StructureData,
        code: Code,
        protocol_section: str = "scf",
        protocol_name: str = "protocol_SIRIUS.yml",
        kpoints_mesh: list[int] | None = None,
        metadata_options: dict | None = None,
    ) -> dict:
        """Build inputs for a CP2K SCF calculation."""
        protocol = load_protocol(protocol_name)
        if protocol_section not in protocol:
            raise KeyError(f"Section '{protocol_section}' not found in {protocol_name}")

        params = copy.deepcopy(protocol[protocol_section])
        basis_pseudo = protocol.get("basis_pseudo", "")
        params["GLOBAL"]["RUN_TYPE"] = "ENERGY_FORCE"
        params["### JOB_TYPE"] = protocol_section

        qs_or_sirius = "SIRIUS"
        cp2k_files = get_cp2k_files_path()

        # Load atom data
        atom_data_path = cp2k_files / basis_pseudo
        if atom_data_path.exists():
            with open(atom_data_path) as f:
                atom_data = json.load(f) if basis_pseudo.endswith(".json") else yaml.safe_load(f)
        else:
            atom_data = {}

        # Set up builder
        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.structure = structure

        # K-points
        if kpoints_mesh and kpoints_mesh != [1, 1, 1]:
            kp = KpointsData()
            kp.set_kpoints_mesh(kpoints_mesh)
            builder.cp2k.kpoints = kp

        # Set cell vectors
        cell = params.get("FORCE_EVAL", {}).get("SUBSYS", {}).get("CELL", {})
        for i, key in enumerate(["A", "B", "C"]):
            if key in cell:
                cell[key] = (
                    f'{cell[key]} {structure.cell[i][0]:<15} '
                    f'{structure.cell[i][1]:<15} {structure.cell[i][2]:<15}'
                )

        # KIND section
        if "SIRIUS" in qs_or_sirius and atom_data:
            dict_merge(params, get_kinds_section_sirius(structure, atom_data))
            builder.cp2k.file = get_file_section_sirius(structure, atom_data)

            # Set cutoffs from pseudopotential data
            if "PW_DFT" in params.get("FORCE_EVAL", {}):
                pw = params["FORCE_EVAL"]["PW_DFT"]["PARAMETERS"]
                gk = [6]
                pw_list = [12]
                ase = structure.get_ase()
                for symbol in set(ase.get_chemical_symbols()):
                    if symbol in atom_data:
                        gk.append(round(0.5 + atom_data[symbol]["cutoff_wfc"] ** 0.5))
                        pw_list.append(round(atom_data[symbol]["cutoff_rho"] ** 0.5))
                        pw_list.append(2 * max(gk))
                pw["PW_CUTOFF"] = max(pw_list)
                pw["GK_CUTOFF"] = max(gk)

        elif atom_data:
            dict_merge(params, get_kinds_section_qs(structure, atom_data))
            builder.cp2k.file = get_file_section_qs()

        # Periodic
        periodic = params.get("FORCE_EVAL", {}).get("SUBSYS", {}).get("CELL", {}).get("PERIODIC")
        if periodic is None:
            if protocol_section in ("dimer", "opt1c", "cluster", "molecule"):
                params.setdefault("FORCE_EVAL", {}).setdefault("SUBSYS", {}).setdefault("CELL", {})["PERIODIC"] = "None"
            else:
                params.setdefault("FORCE_EVAL", {}).setdefault("SUBSYS", {}).setdefault("CELL", {})["PERIODIC"] = "XYZ"

        builder.cp2k.parameters = Dict(dict=params)
        builder.cp2k.code = code
        builder.cp2k.settings = Dict(dict={
            "additional_retrieve_list": [
                "aiida.inp", "aiida-pos-1.xyz", "aiida-frc-1.xyz",
                "aiida-1.cell", "aiida-s_p_forces-1_0.xyz",
                "aiida-s_p_stress_tensor-1_0.stress_tensor",
                "aiida-1.stress", "aiida.coords.xyz",
            ],
        })

        if metadata_options:
            builder.cp2k.metadata.options = metadata_options
        else:
            builder.cp2k.metadata.options = {
                "resources": {
                    "num_machines": self.config.metadata_options.num_machines,
                    "num_mpiprocs_per_machine": self.config.metadata_options.num_mpiprocs_per_machine,
                },
                "max_wallclock_seconds": self.config.metadata_options.max_wallclock_seconds,
                "withmpi": True,
            }

        parser_name = "cp2k_simple_parser" if "CELL_OPT" in str(params.get("GLOBAL", {}).get("RUN_TYPE", "")) else "cp2k_efs_parser"
        builder.cp2k.metadata.options["parser_name"] = parser_name
        builder.cp2k.metadata.label = protocol_section

        builder.handler_overrides = Dict(dict={"restart_incomplete_calculation": {"enabled": False}})
        builder.max_iterations = Int(1)

        return builder

    def build_gw_inputs(
        self,
        structure: StructureData,
        code: Code,
        protocol_name: str = "protocol_GW.yml",
        kpoints_mesh: list[int] | None = None,
        metadata_options: dict | None = None,
    ) -> dict:
        """Build inputs for a CP2K GW calculation."""
        return self.build_scf_inputs(
            structure=structure,
            code=code,
            protocol_section="gw",
            protocol_name=protocol_name,
            kpoints_mesh=kpoints_mesh or self.config.gw.kpoints_mesh,
            metadata_options=metadata_options,
        )
