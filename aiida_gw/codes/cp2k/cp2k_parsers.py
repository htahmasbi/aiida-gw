import numpy as np

from ase import Atoms
from aiida_cp2k.parsers import Cp2kBaseParser
from aiida.engine import ExitCode
from aiida.common import NotExistent
from aiida.orm import Dict
from aiida.plugins import DataFactory
from aiida_gw.codes.cp2k.parsers import (
    read_structure,
    parse_cp2k_output_simple,
    read_coordinates,
    read_positions,
    read_forces,
    read_stress_tensor,
    read_s_p_forces,
    read_s_p_stress_tensor,
    read_cell_parameters,
    read_lattice_parameters,
    read_bandstructure,
    read_dos_pdos,
)

StructureData = DataFactory("structure")

class Cp2kSimpleParser(Cp2kBaseParser):
    """ AiiDA parser class for the output of CP2K
        Modified for SIRIUS
    """
    def parse(self, **kwargs):
        try:
            _ = self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        exit_code = self._parse_stdout()
        last_structure = None
        try:
            last_structure = self._parse_final_structure()
            if isinstance(last_structure, StructureData):
                self.out("output_structure", last_structure)
        except NotExistent:
            self.logger.warning("No restart file found in the retrieved folder.")
        if exit_code is not None:
            return exit_code
        if isinstance(last_structure, ExitCode):
            return last_structure
        return ExitCode(0)

    def _parse_final_structure(self):
        fname = 'aiida-1.restart'
        if fname not in self.retrieved.base.repository.list_object_names():
            raise NotExistent("No restart file available, so the output trajectory can't be extracted")
        try:
            output_string = self.retrieved.base.repository.get_object_content(fname)
        except OSError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ
        return StructureData(ase=Atoms(**read_structure(output_string)))

    def _parse_stdout(self):
        exit_code, output_string = self._read_stdout()
        if exit_code:
            return exit_code
        # Check the standard output for errors.
        exit_code = self._check_stdout_for_errors(output_string)
        if exit_code:
            return exit_code
        result_dict = parse_cp2k_output_simple(output_string)
        self.out("output_parameters", Dict(dict=result_dict))
        return None

class Cp2kEFSParser(Cp2kBaseParser):
    """ AiiDA parser class for the output of CP2K
        Modified for SIRIUS
    """
    def parse(self, **kwargs):
        try:
            _ = self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        exit_code = self._parse_stdout()
        if exit_code is not None:
            return exit_code
        return ExitCode(0)

    def _parse_stdout(self):
        exit_code, output_string = self._read_stdout()
        if exit_code:
            return exit_code
        exit_code = self._check_stdout_for_errors(output_string)
        if exit_code:
            return exit_code
        result_dict = parse_cp2k_output_simple(output_string)
        exit_code = self._parse_efs(result_dict)
        if exit_code:
            return exit_code
        return None

    def _parse_efs(self, result_dict):
        symbols = positions = cells = forces = stress_tensor = []
        if result_dict["run_type"] in ["GEO_OPT", "CELL_OPT"]:
            if 'aiida-pos-1.xyz' in self.retrieved.list_object_names() and\
               'aiida-frc-1.xyz' in self.retrieved.list_object_names() and\
               'aiida-1.cell' in self.retrieved.list_object_names():
                positions = read_positions(self.retrieved.get_object_content('aiida-pos-1.xyz'))
                forces = read_forces(self.retrieved.get_object_content('aiida-frc-1.xyz'))
                stress_tensor = read_stress_tensor(self.retrieved.get_object_content('aiida-1.stress'))
                cells = read_cell_parameters(self.retrieved.get_object_content('aiida-1.cell'))
                symbols, _ = read_coordinates(self.retrieved.get_object_content('aiida.coords.xyz'))
            else:
                self.logger.warning("Missing trajectory files for %s run; skipping structure output.", result_dict["run_type"])
        if result_dict["run_type"] in ["ENERGY_FORCE"]:
            if 'aiida-s_p_forces-1_0.xyz' in self.retrieved.list_object_names():
                symbols, positions = read_coordinates(self.retrieved.get_object_content('aiida.coords.xyz'))
                forces = read_s_p_forces(self.retrieved.get_object_content('aiida-s_p_forces-1_0.xyz'))
                if result_dict["SIRIUS"]:
                    cells = [result_dict["lattice_vectors"]]
                else:
                    cells = [read_lattice_parameters(self.retrieved.get_object_content('aiida.inp'))]
            else:
                self.logger.warning("Missing SIRIUS force file 'aiida-s_p_forces-1_0.xyz'; skipping force/structure output.")
            if 'aiida-s_p_stress_tensor-1_0.stress_tensor' in self.retrieved.list_object_names():
                stress_tensor = read_s_p_stress_tensor(self.retrieved.get_object_content('aiida-s_p_stress_tensor-1_0.stress_tensor'))
            else:
                self.logger.warning("Missing SIRIUS stress tensor file; skipping stress output.")
        if result_dict["run_type"] in ["ENERGY"]:
            for fname in ('aiida-1.restart', 'aiida-RESTART.kp'):
                try:
                    output_string = self.retrieved.base.repository.get_object_content(fname)
                    ase_struct = Atoms(**read_structure(output_string))
                    symbols = ase_struct.get_chemical_symbols()
                    positions = [ase_struct.get_positions()]
                    cells = [ase_struct.get_cell().array]
                    forces = [np.zeros((len(symbols), 3))]
                    stress_tensor = [np.zeros(9)]
                    break
                except Exception as exc:
                    self.logger.warning("Failed to parse %s run type restart file '%s': %s", result_dict["run_type"], fname, exc)
            else:
                self.logger.warning("No usable restart file for %s run; falling back to 'aiida.coords.xyz'.", result_dict["run_type"])
                if 'aiida.coords.xyz' in self.retrieved.list_object_names():
                    try:
                        symbols, positions = read_coordinates(self.retrieved.get_object_content('aiida.coords.xyz'))
                        if result_dict["SIRIUS"]:
                            cells = [result_dict["lattice_vectors"]]
                        else:
                            cells = [read_lattice_parameters(self.retrieved.get_object_content('aiida.inp'))]
                        forces = [np.zeros((len(symbols), 3))]
                        stress_tensor = [np.zeros(9)]
                    except Exception as exc:
                        self.logger.warning("Failed to parse fallback coordinates file 'aiida.coords.xyz': %s", exc)
                else:
                    self.logger.warning("No coordinate file available for %s run; skipping structure output.", result_dict["run_type"])

        if symbols and positions and cells and forces and stress_tensor:
            result_dict['motion_step_info'].update({'symbols': symbols, 'positions': positions, 'cells': cells, 'forces': forces, 'stress_tensor': stress_tensor})
            cell_pbc = [True, True, True] #result_dict['cell_pbc']
            try:
                cell = np.asarray(cells[-1], dtype=np.float64).reshape(3, 3)
                output_structure = StructureData(ase=Atoms(symbols = symbols, positions = positions[-1], cell = cell, pbc = cell_pbc))
                self.out("output_structure", output_structure)
            except Exception as exc:
                self.logger.error("Failed to construct output structure from parsed data: %s", exc)
        else:
            self.logger.warning("Incomplete structure data for %s run; skipping structure output.", result_dict["run_type"])

        self._parse_gw_outputs(result_dict)

        self.out("output_parameters", Dict(dict=result_dict))
        return None

    def _parse_gw_outputs(self, result_dict):
        """Parse GW-specific output files (bandstructure, DOS/PDOS).

        Handles both standard CP2K naming (aiida-BANDSTRUCTURE*, aiida-dos.dat, aiida-pdos*)
        and custom naming conventions (bandstructure_*, DOS_PDOS_*.out).
        """
        available = set(self.retrieved.list_object_names())

        bandstructure_files = {
            "bandstructure_SCF_and_G0W0": "g0w0",
            "bandstructure_SCF_and_G0W0_plus_SOC": "g0w0_soc",
            "aiida-BANDSTRUCTURE_1-1.dat": "kpoints",
        }
        for fname, key in bandstructure_files.items():
            if fname in available:
                try:
                    content = self.retrieved.get_object_content(fname)
                    bs_data = read_bandstructure(content)
                    result_dict[f"bandstructure_{key}"] = {
                        "kpoints": bs_data["kpoints"].tolist(),
                        "kpoint_labels": bs_data["kpoint_labels"],
                        "units": bs_data["units"],
                    }
                    for level_name, eigenvalues in bs_data["eigenvalues"].items():
                        result_dict[f"bandstructure_{key}"][f"eigenvalues_{level_name}"] = eigenvalues.tolist()
                    self.logger.info("Parsed %s bandstructure data", fname)
                except Exception as exc:
                    self.logger.warning("Failed to parse bandstructure file '%s': %s", fname, exc)

        eigenvalue_files = [
            "aiida-BANDSTRUCTURE_1-1_0.dat",
            "aiida-BANDSTRUCTURE_1-1_SOC.dat",
            "aiida-BANDSTRUCTURE_1-1_0_GW0.dat",
        ]
        for fname in eigenvalue_files:
            if fname in available:
                try:
                    content = self.retrieved.get_object_content(fname)
                    bs_data = read_bandstructure(content)
                    level_key = "scf" if "SCF" in fname.upper() or "_0" in fname else "g0w0"
                    if "SOC" in fname.upper():
                        level_key += "_soc"
                    key = f"{level_key}_bandstructure"
                    if key not in result_dict:
                        result_dict[key] = {"kpoints": [], "kpoint_labels": [], "units": "eV"}
                    for level_name, eigenvalues in bs_data["eigenvalues"].items():
                        result_dict[key][f"eigenvalues_{level_name}"] = eigenvalues.tolist()
                    self.logger.info("Parsed eigenvalue file %s", fname)
                except Exception as exc:
                    self.logger.warning("Failed to parse eigenvalue file '%s': %s", fname, exc)

        dos_files = {
            "DOS_PDOS_SCF.out": "scf",
            "DOS_PDOS_G0W0.out": "g0w0",
            "DOS_PDOS_SCF_SOC.out": "scf_soc",
            "DOS_PDOS_G0W0_SOC.out": "g0w0_soc",
        }
        for fname, key in dos_files.items():
            if fname in available:
                try:
                    content = self.retrieved.get_object_content(fname)
                    dos_data = read_dos_pdos(content)
                    result_dict[f"{key}_dos"] = {
                        "energy": dos_data["energy"].tolist(),
                        "total_dos": dos_data["total_dos"].tolist(),
                        "units": dos_data["units"],
                    }
                    if "fermi_energy" in dos_data:
                        result_dict[f"{key}_dos"]["fermi_energy"] = dos_data["fermi_energy"]
                    if dos_data["pdos"]:
                        result_dict[f"{key}_dos"]["pdos"] = {k: v.tolist() for k, v in dos_data["pdos"].items()}
                    self.logger.info("Parsed %s DOS/PDOS data", fname)
                except Exception as exc:
                    self.logger.warning("Failed to parse DOS/PDOS file '%s': %s", fname, exc)

        if "aiida-dos.dat" in available:
            try:
                content = self.retrieved.get_object_content("aiida-dos.dat")
                dos_data = read_dos_pdos(content)
                result_dict["dos"] = {
                    "energy": dos_data["energy"].tolist(),
                    "total_dos": dos_data["total_dos"].tolist(),
                    "units": dos_data["units"],
                }
                if "fermi_energy" in dos_data:
                    result_dict["dos"]["fermi_energy"] = dos_data["fermi_energy"]
                self.logger.info("Parsed aiida-dos.dat")
            except Exception as exc:
                self.logger.warning("Failed to parse aiida-dos.dat: %s", exc)

        pdos_files = [f for f in available if f.startswith("aiida-pdos")]
        for fname in pdos_files:
            try:
                content = self.retrieved.get_object_content(fname)
                dos_data = read_dos_pdos(content)
                pdos_key = fname.replace("aiida-", "").replace(".dat", "")
                result_dict[f"pdos_{pdos_key}"] = {
                    "energy": dos_data["energy"].tolist(),
                    "total_dos": dos_data["total_dos"].tolist(),
                    "pdos": {k: v.tolist() for k, v in dos_data["pdos"].items()},
                    "units": dos_data["units"],
                }
                self.logger.info("Parsed %s", fname)
            except Exception as exc:
                self.logger.warning("Failed to parse %s: %s", fname, exc)
