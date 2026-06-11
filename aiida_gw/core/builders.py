from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from aiida.orm import Code, Dict, Int, KpointsData, SinglefileData, StructureData
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_gw.core.config import ProjectConfig

logger = logging.getLogger(__name__)

StructureData = DataFactory("core.structure")
Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")


def get_kpoints(kpoints_distance: float | None, structure: StructureData) -> KpointsData | None:
    """Compute KpointsData from a target k-point distance using the structure cell."""
    if kpoints_distance is None:
        return None
    KpointsData = DataFactory("array.kpoints")
    mesh = KpointsData()
    mesh.set_cell_from_structure(structure)
    mesh.set_kpoints_mesh_from_density(distance=kpoints_distance)
    return mesh


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


def load_atom_data(basis_pseudo: str) -> dict:
    """Load atom data YAML/JSON from cp2k_files."""
    path = get_cp2k_files_path() / basis_pseudo
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f) if basis_pseudo.endswith(".json") else yaml.safe_load(f)


def dict_merge(base: dict, merge: dict) -> dict:
    """Recursively merge merge into base."""
    for k, v in merge.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            dict_merge(base[k], v)
        else:
            base[k] = v
    return base


def get_kinds_section_qs(
    structure,
    atom_data: dict,
    gw_config=None,
) -> dict:
    """Build &KIND section for QUICKSTEP.

    Uses ``basis_set`` + ``pseudopotential`` from *atom_data* for ORB/POTENTIAL.
    RI_AUX is either read from *atom_data* (``ri_basis_set``) or auto-resolved
    from the configured RI basis file when *gw_config* is given and
    ``auto_resolve_ri`` is ``True``.
    """
    kinds = []
    ase = structure.get_ase()
    symbols = ase.get_chemical_symbols()
    seen = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        kind: dict[str, str] = {
            "_": symbol,
            "BASIS_SET ORB": atom_data["basis_set"][symbol],
            "POTENTIAL": atom_data["pseudopotential"][symbol],
        }
        ri = atom_data.get("ri_basis_set", {}).get(symbol)
        if ri:
            kind["BASIS_SET RI_AUX"] = ri
        elif gw_config and gw_config.auto_resolve_ri:
            resolved = _resolve_ri_for_element(symbol, gw_config)
            if resolved:
                kind["BASIS_SET RI_AUX"] = resolved
        kinds.append(kind)
    return {"FORCE_EVAL": {"SUBSYS": {"KIND": kinds}}}


def _resolve_ri_for_element(symbol: str, gw_config) -> str | None:
    """Auto-resolve RI basis for *symbol* from the configured RI basis file."""
    try:
        from aiida_gw.codes.cp2k.data_reader import resolve_ri_basis_name

        return resolve_ri_basis_name(
            gw_config.ri_basis_set_file,
            symbol,
            accuracy_target=gw_config.ri_basis_accuracy_target,
            orb_basis=gw_config.orb_basis,
        )
    except Exception as exc:
        logger.warning(f"Could not auto-resolve RI basis for {symbol}: {exc}")
        return None


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
        upf_stem = Path(atom_data[symbol]["filename"]).stem
        kind = {
            "_": symbol,
            "POTENTIAL": f"UPF {upf_stem}.json",
        }
        kinds.append(kind)
    return {"FORCE_EVAL": {"SUBSYS": {"KIND": kinds}}}


def get_file_section_qs() -> dict:
    """Get CP2K file resources for QUICKSTEP (GTH potentials + basis sets)."""
    cp2k_files = get_cp2k_files_path()
    files = {}
    for fname in ["GTH_POTENTIALS", "GTH_BASIS_SETS", "BASIS_MOLOPT", "BASIS_MOLOPT_UCL"]:
        path = cp2k_files / fname
        if path.exists():
            with open(path, "rb") as fh:
                files[fname] = SinglefileData(file=fh)
    return files


def get_cutoff_sirius(atom_data: dict, structure, relax_factor: float | None = None) -> tuple[int, int]:
    """Compute PW_CUTOFF and GK_CUTOFF from SIRIUS pseudopotential data.

    Args:
        atom_data: Dict with element keys containing cutoff_wfc and cutoff_rho.
        structure: StructureData to extract element symbols.
        relax_factor: Optional scale factor for relaxation runs (e.g. 0.8).

    Returns:
        (pw_cutoff, gk_cutoff)
    """
    gk = [6]
    pw = [12]
    ase = structure.get_ase()
    for symbol in set(ase.get_chemical_symbols()):
        if symbol in atom_data:
            gk.append(round(0.5 + atom_data[symbol]["cutoff_wfc"] ** 0.5))
            pw.append(round(atom_data[symbol]["cutoff_rho"] ** 0.5))
            pw.append(2 * max(gk))
    pw_cutoff = max(pw)
    gk_cutoff = max(gk)
    if relax_factor is not None:
        pw_cutoff = round(pw_cutoff * relax_factor)
        gk_cutoff = round(gk_cutoff * relax_factor)
        if pw_cutoff < 2 * gk_cutoff:
            pw_cutoff = 2 * gk_cutoff
    return pw_cutoff, gk_cutoff


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
        fname = atom_data[symbol]["filename"]
        path = cp2k_files / f"{Path(fname).stem}.json"
        if path.exists():
            with open(path, "rb") as fh:
                files[symbol] = SinglefileData(file=fh)
    return files


_INPLANE_PATHS: dict[str, list[tuple[str, float, float, float]]] = {
    "hexagonal": [
        ("GAMMA", 0.0, 0.0, 0.0),
        ("M", 0.5, 0.0, 0.0),
        ("K", 1 / 3, 1 / 3, 0.0),
        ("GAMMA", 0.0, 0.0, 0.0),
    ],
    "square": [
        ("GAMMA", 0.0, 0.0, 0.0),
        ("X", 0.5, 0.0, 0.0),
        ("M", 0.5, 0.5, 0.0),
        ("GAMMA", 0.0, 0.0, 0.0),
    ],
    "rectangular": [
        ("GAMMA", 0.0, 0.0, 0.0),
        ("X", 0.5, 0.0, 0.0),
        ("S", 0.5, 0.5, 0.0),
        ("Y", 0.0, 0.5, 0.0),
        ("GAMMA", 0.0, 0.0, 0.0),
    ],
    "oblique": [
        ("GAMMA", 0.0, 0.0, 0.0),
        ("X", 0.5, 0.0, 0.0),
        ("M", 0.5, 0.5, 0.0),
        ("Y", 0.0, 0.5, 0.0),
        ("GAMMA", 0.0, 0.0, 0.0),
    ],
}


def _classify_2d_lattice(structure) -> str:
    """Classify the 2D in-plane Bravais lattice from lattice vectors."""
    import math

    import numpy as np

    matrix = np.array(structure.cell)
    a_len = float(np.linalg.norm(matrix[0]))
    b_len = float(np.linalg.norm(matrix[1]))
    cos_gamma = np.dot(matrix[0], matrix[1]) / (a_len * b_len)
    cos_gamma = max(-1.0, min(1.0, cos_gamma))
    gamma = math.degrees(math.acos(cos_gamma))
    ratio = min(a_len, b_len) / max(a_len, b_len)
    angle_tol = 15.0
    ratio_tol = 0.15

    if (abs(gamma - 120) < angle_tol or abs(gamma - 60) < angle_tol) and (1.0 - ratio) < ratio_tol:
        return "hexagonal"
    if abs(gamma - 90) < angle_tol and (1.0 - ratio) < ratio_tol:
        return "square"
    if abs(gamma - 90) < angle_tol:
        return "rectangular"
    return "oblique"


def get_bandstructure_path(structure) -> list[str]:
    """Generate CP2K SPECIAL_POINT lines from the 2D in-plane Bravais lattice."""
    lattice = _classify_2d_lattice(structure)
    path = _INPLANE_PATHS.get(lattice, _INPLANE_PATHS["oblique"])
    return [
        f"{label}  {k0:f}  {k2:f}  {k1:f}"
        for label, k0, k1, k2 in path
    ]


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
        kpoints_distance: float | None = None,
        metadata_options: dict | None = None,
    ) -> dict:
        """Build inputs for a CP2K calculation (SCF or other)."""
        protocol = load_protocol(protocol_name)
        if protocol_section not in protocol:
            raise KeyError(f"Section '{protocol_section}' not found in {protocol_name}")

        params = copy.deepcopy(protocol[protocol_section])
        basis_pseudo = protocol.get("basis_pseudo", "")

        atom_data = load_atom_data(basis_pseudo)
        method = (
            params.get("FORCE_EVAL", {})
            .get("METHOD", "SIRIUS")
        )

        # Ensure GLOBAL exists
        params.setdefault("GLOBAL", {})
        if "RUN_TYPE" not in params["GLOBAL"]:
            params["GLOBAL"]["RUN_TYPE"] = "ENERGY_FORCE"

        params["### JOB_TYPE"] = protocol_section

        # Set up builder
        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.structure = structure

        # K-points priority: CLI kpoints_mesh > explicit kpoints_distance > protocol kpoints_distance
        kp_obj = None
        if kpoints_mesh and kpoints_mesh != [1, 1, 1]:
            kp_obj = KpointsData()
            kp_obj.set_kpoints_mesh(kpoints_mesh)
        else:
            if kpoints_distance is None:
                kpoints_distance = params.pop("kpoints_distance", None)
            else:
                params.pop("kpoints_distance", None)
            kp_obj = get_kpoints(kpoints_distance, structure)
        if kp_obj is not None:
            builder.cp2k.kpoints = kp_obj
            mesh, _ = kp_obj.get_kpoints_mesh()
        else:
            mesh = None

        # --- Method-specific handling ---
        if "SIRIUS" in method.upper():
            # SIRIUS path
            params.setdefault("FORCE_EVAL", {})
            dft = params["FORCE_EVAL"].setdefault("DFT", {})
            dft.setdefault("XC", {}).setdefault("XC_FUNCTIONAL", {"_": "PBE"})

            if atom_data:
                dict_merge(params, get_kinds_section_sirius(structure, atom_data))
                builder.cp2k.file = get_file_section_sirius(structure, atom_data)

                if "PW_DFT" in params.get("FORCE_EVAL", {}):
                    pw_dft = params["FORCE_EVAL"]["PW_DFT"]
                    pw_params = pw_dft.setdefault("PARAMETERS", {})
                    pw_control = pw_dft.setdefault("CONTROL", {})
                    pw_control["MPI_GRID_DIMS"] = f"1 {self.config.metadata_options.num_mpiprocs_per_machine}"
                    if mesh and mesh != (1, 1, 1):
                        pw_params["NGRIDK"] = f"{mesh[0]} {mesh[1]} {mesh[2]}"

                    pw_cutoff, gk_cutoff = get_cutoff_sirius(atom_data, structure)
                    pw_params["PW_CUTOFF"] = pw_cutoff
                    pw_params["GK_CUTOFF"] = gk_cutoff

        else:
            # QS/GPW path — used for GW and standard Quickstep calculations
            params["FORCE_EVAL"]["METHOD"] = "Quickstep"

            # Load basis/potential filenames and MGRID from config
            gw_cfg = self.config.gw
            dft = params["FORCE_EVAL"].setdefault("DFT", {})
            dft.setdefault("BASIS_SET_FILE_NAME", [])
            dft.setdefault("POTENTIAL_FILE_NAME", "")
            dft.setdefault("MGRID", {})

            # Set basis set file names (cluster paths)
            if gw_cfg.basis_set_file:
                file_list = dft["BASIS_SET_FILE_NAME"]
                if not isinstance(file_list, list):
                    file_list = [file_list]
                if gw_cfg.basis_set_file not in file_list:
                    file_list.append(gw_cfg.basis_set_file)
                dft["BASIS_SET_FILE_NAME"] = file_list

            if gw_cfg.ri_basis_set_file:
                file_list = dft["BASIS_SET_FILE_NAME"]
                if not isinstance(file_list, list):
                    file_list = [file_list]
                if gw_cfg.ri_basis_set_file not in file_list:
                    file_list.append(gw_cfg.ri_basis_set_file)
                dft["BASIS_SET_FILE_NAME"] = file_list

            if gw_cfg.potential_file:
                dft["POTENTIAL_FILE_NAME"] = gw_cfg.potential_file

            # Set MGRID cutoff from config
            dft["MGRID"]["CUTOFF"] = self.config.gw.cutoff
            dft["MGRID"]["REL_CUTOFF"] = self.config.gw.rel_cutoff

            # The QS section should already exist in the protocol, set eps defaults from config
            qs = dft.setdefault("QS", {})
            qs.setdefault("EPS_DEFAULT", self.config.gw.eps_default)
            qs.setdefault("EPS_PGF_ORB", self.config.gw.eps_pgf_orb)

            # SCF section — set eps_scf, max_scf from config if not in protocol
            scf = dft.setdefault("SCF", {})
            scf.setdefault("EPS_SCF", self.config.gw.eps_scf)
            scf.setdefault("MAX_SCF", self.config.gw.max_scf)
            mixing = scf.setdefault("MIXING", {})
            mixing.setdefault("METHOD", "BROYDEN_MIXING")
            mixing.setdefault("ALPHA", self.config.gw.mixing_alpha)
            mixing.setdefault("BETA", self.config.gw.mixing_beta)
            mixing.setdefault("NBROYDEN", self.config.gw.mixing_nbroyden)

            # Generate KIND sections from atom_data (includes RI_AUX if available)
            if atom_data:
                dict_merge(params, get_kinds_section_qs(structure, atom_data, gw_config=self.config.gw))
                if not dft.get("BASIS_SET_FILE_NAME"):
                    builder.cp2k.file = get_file_section_qs()

        # --- Set cell vectors from structure ---
        fe = params.get("FORCE_EVAL", {})
        cell = fe.get("SUBSYS", {}).get("CELL", {})
        for i, key in enumerate(["A", "B", "C"]):
            cell[key] = (
                f'{cell.get(key, "")} {structure.cell[i][0]:<15} '
                f'{structure.cell[i][1]:<15} {structure.cell[i][2]:<15}'
            )

        # Periodic (default XYZ, or from protocol)
        periodic = cell.get("PERIODIC")
        if periodic is None:
            cell["PERIODIC"] = "XYZ"

        # Build the final parameters dict
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

        parser_name = (
            "cp2k_simple_parser"
            if "CELL_OPT" in str(params.get("GLOBAL", {}).get("RUN_TYPE", ""))
            else "cp2k_efs_parser"
        )
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
        kpoints_w_mesh: list[int] | None = None,
        kpoints_distance: float | None = None,
        kpoints_w_distance: float | None = None,
        metadata_options: dict | None = None,
    ) -> dict:
        """Build inputs for a CP2K GW calculation."""
        gw_config = self.config.gw

        builder = self.build_scf_inputs(
            structure=structure,
            code=code,
            protocol_section="gw",
            protocol_name=protocol_name,
            kpoints_mesh=kpoints_mesh,
            kpoints_distance=kpoints_distance,
            metadata_options=metadata_options,
        )

        # Add GW-specific KPOINTS_W and bandstructure path
        # Priority: CLI kpoints_w_mesh > config kpoints_w_distance > protocol kpoints_w_distance
        params = builder.cp2k.parameters.get_dict()
        if kpoints_w_mesh is not None:
            kpoints_w = kpoints_w_mesh
        else:
            if kpoints_w_distance is None:
                kpoints_w_distance = params.pop("kpoints_w_distance", None)
            else:
                params.pop("kpoints_w_distance", None)
            if kpoints_w_distance is not None:
                kw_kp = get_kpoints(kpoints_w_distance, structure)
                if kw_kp is not None:
                    kpoints_w, _ = kw_kp.get_kpoints_mesh()
                    kpoints_w = list(kpoints_w)
                else:
                    kpoints_w = gw_config.kpoints_w_mesh or gw_config.kpoints_mesh
            else:
                kpoints_w = gw_config.kpoints_w_mesh or gw_config.kpoints_mesh

        bs_path = params.setdefault("FORCE_EVAL", {}).setdefault("PROPERTIES", {}).setdefault("BANDSTRUCTURE", {})
        gw_sec = bs_path.setdefault("GW", {})
        gw_sec["KPOINTS_W"] = " ".join(str(k) for k in kpoints_w)

        # Generate bandstructure path
        bs_path.setdefault("DOS", {})
        bs_path.setdefault("BANDSTRUCTURE_PATH", {})
        bs_path["BANDSTRUCTURE_PATH"].setdefault("NPOINTS", gw_config.bs_npoints)
        bs_path["BANDSTRUCTURE_PATH"].setdefault("UNITS", "B_VECTOR")
        bs_path["BANDSTRUCTURE_PATH"].setdefault("SPECIAL_POINT", get_bandstructure_path(structure))

        # Set GW-specific numerical params from config
        gw_sec.setdefault("NUM_TIME_FREQ_POINTS", gw_config.num_time_freq)
        gw_sec.setdefault("MEMORY_PER_PROC", gw_config.memory_per_proc)
        gw_sec.setdefault("EPS_FILTER", gw_config.eps_filter)
        gw_sec.setdefault("CUTOFF_RADIUS_RI", gw_config.cutoff_radius_ri)
        gw_sec.setdefault("REGULARIZATION_RI", gw_config.regularization_ri)

        builder.cp2k.parameters = Dict(dict=params)
        return builder
