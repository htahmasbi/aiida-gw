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
    read_lattice_parameters)

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
                self.logger.warning("No usable restart file for %s run; skipping structure output.", result_dict["run_type"])

        if symbols and positions and cells and forces and stress_tensor:
            result_dict['motion_step_info'].update({'symbols': symbols, 'positions': positions, 'cells': cells, 'forces': forces, 'stress_tensor': stress_tensor})
            cell_pbc = [True, True, True] #result_dict['cell_pbc']
            output_structure = StructureData(ase=Atoms(symbols = symbols, positions = positions[-1], cell = cells[-1], pbc = cell_pbc))
            self.out("output_structure", output_structure)
        else:
            self.logger.warning("Incomplete structure data for %s run; skipping structure output.", result_dict["run_type"])
        self.out("output_parameters", Dict(dict=result_dict))
        return None
