import os
from random import uniform
import collections
from copy import deepcopy
import json
import yaml
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.engine import WorkChain
from aiida.orm import Int, Str, Dict, Code, SinglefileData, load_group
import aiida_datagen.workflows.settings as settings

results_step1_group = load_group(label='results_step1')
results_step3_group = load_group(label='results_step3')
StructureData = DataFactory('structure')

def dict_merge(dct, merge_dct):
    """ Taken from https://gist.github.com/angstwad/
    """
    for k in merge_dct.keys():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], collections.abc.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def get_options():
    job_script = settings.job_script
    resources = {
        'num_machines': job_script['geopt']['nodes'],
        'num_mpiprocs_per_machine': job_script['geopt']['ntasks']}
    if job_script['geopt']['ncpu']:
        resources['num_cores_per_mpiproc'] = job_script['geopt']['ncpu']
    options = {'resources': resources,
               'max_wallclock_seconds': job_script['geopt']['time']}
    if job_script['geopt']['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
    return options

def get_kinds_section_QS(structure, basis_pseudo, magnetization_tags=None):
    """ Write the &KIND section
        Taken from aiida-commonworkflow
    """
    kinds = []
    with open(os.path.join(settings.CP2K_input_files_path, basis_pseudo), 'rb') as fhandle:
        atom_data = yaml.safe_load(fhandle)
    ase_structure = structure.get_ase()
    symbol_tag = {
    (symbol, str(tag)) for symbol, tag in zip(ase_structure.get_chemical_symbols(), ase_structure.get_tags())
    }
    for symbol, tag in symbol_tag:
        new_atom = {
            '_': symbol if tag == '0' else symbol + tag,
        }
        new_atom['BASIS_SET'] = atom_data['basis_set'][symbol]
        new_atom['POTENTIAL'] = atom_data['pseudopotential'][symbol]

        if magnetization_tags:
            new_atom['MAGNETIZATION'] = magnetization_tags[tag]
        kinds.append(new_atom)
    return {'FORCE_EVAL': {'SUBSYS': {'KIND': kinds}}}

def get_kinds_section_SIRIUS(structure, basis_pseudo, magnetization_tags=None):
    """ Write the &KIND section
        Taken from aiida-commonworkflow
    """
    kinds = []
    with open(os.path.join(settings.CP2K_input_files_path, basis_pseudo), 'rb') as fhandle:
        atom_data = json.loads(fhandle.read())
    ase_structure = structure.get_ase()
    symbol_tag = {
    (symbol, str(tag)) for symbol, tag in zip(ase_structure.get_chemical_symbols(), ase_structure.get_tags())
    }
    for symbol, tag in symbol_tag:
        new_atom = {
            '_': symbol if tag == '0' else symbol + tag,
        }
        filename = os.path.splitext(atom_data[symbol]['filename'])[0]+'.json'
        new_atom = {
            '_': symbol if tag == '0' else symbol + tag,
            'POTENTIAL': 'UPF ' + filename,
        }

        if magnetization_tags:
            new_atom['MAGNETIZATION'] = magnetization_tags[tag]
        kinds.append(new_atom)
    return {'FORCE_EVAL': {'SUBSYS': {'KIND': kinds}}}

def get_cutoff_SIRIUS(structure, basis_pseudo):
    gk_cutoff = [6]
    pw_cutoff = [12]
    with open(os.path.join(settings.CP2K_input_files_path, basis_pseudo), 'rb') as fhandle:
        atom_data = json.loads(fhandle.read())
    ase_structure = structure.get_ase()
    symbol_tag = {
    (symbol, str(tag)) for symbol, tag in zip(ase_structure.get_chemical_symbols(), ase_structure.get_tags())
    }
    for symbol, tag in symbol_tag:
        gk_cutoff.append(round(0.5 + (atom_data[symbol]['cutoff_wfc'])**0.5))
        pw_cutoff.append(round((atom_data[symbol]['cutoff_rho'])**0.5))
        pw_cutoff.append(round(2 * max(gk_cutoff)))
    return max(pw_cutoff), max(gk_cutoff)

def get_file_section_QS():
    """ Potential files for QS
    """
    files_dict =  {}
    with open(os.path.join(settings.CP2K_input_files_path,'GTH_POTENTIALS'), 'rb') as handler:
        potential = SinglefileData(file=handler)
    files_dict['potential'] = potential
    with open(os.path.join(settings.CP2K_input_files_path,'GTH_BASIS_SETS'), 'rb') as handle:
        basis_gth = SinglefileData(file=handle)
    files_dict['basis_gth'] = basis_gth
    with open(os.path.join(settings.CP2K_input_files_path,'BASIS_MOLOPT'), 'rb') as handle:
        basis_molopt = SinglefileData(file=handle)
    files_dict['basis_molopt'] = basis_molopt
    with open(os.path.join(settings.CP2K_input_files_path,'BASIS_MOLOPT_UCL'), 'rb') as handle:
        basis_molopt_ucl = SinglefileData(file=handle)
    files_dict['basis_molopt_ucl'] = basis_molopt_ucl
    return files_dict

def get_file_section_SIRIUS(structure, basis_pseudo):
    """ Potential files for SIRIUS
    """
    files_dict =  {}
    with open(os.path.join(settings.CP2K_input_files_path,basis_pseudo), 'rb') as fhandle:
        atom_data = json.loads(fhandle.read())
    ase_structure = structure.get_ase()
    symbol_tag = {
    (symbol, str(tag)) for symbol, tag in zip(ase_structure.get_chemical_symbols(), ase_structure.get_tags())
    }
    for symbol, tag in symbol_tag:
        filename = os.path.splitext(atom_data[symbol]['filename'])[0]+'.json'
        with open(os.path.join(settings.CP2K_input_files_path,'pseudopotentials',filename), 'rb') as fhandle:
            files_dict[symbol] = SinglefileData(file=fhandle)
    return files_dict

def get_kpoints(kpoints_distance, structure):
    """  kpoints for SIRIUS/QS
    """
    KpointsData = DataFactory('array.kpoints')
    if kpoints_distance:
        kpoints_mesh = KpointsData()
        kpoints_mesh.set_cell_from_structure(structure)
        kpoints_mesh.set_kpoints_mesh_from_density(distance=kpoints_distance)
        return kpoints_mesh
    return None

def construct_builder(structure, parameters, basis_pseudo, QSorSIRIUS):
    job_type = parameters['### JOB_TYPE']
    Workflow = WorkflowFactory('cp2k.base')
    builder = Workflow.get_builder()
    builder.cp2k.structure = structure
    kpoints_distance = parameters.pop('kpoints_distance')
    kpoints = get_kpoints(kpoints_distance, structure)
    mesh, _ = kpoints.get_kpoints_mesh()
    if 'QS' in QSorSIRIUS:
        if mesh != [1, 1, 1]:
            builder.cp2k.kpoints = kpoints
        dict_merge(parameters, get_kinds_section_QS(structure, basis_pseudo))
        builder.cp2k.file = get_file_section_QS()
    if 'SIRIUS' in QSorSIRIUS:
        if mesh != [1, 1, 1]:
            parameters['FORCE_EVAL']['PW_DFT']['PARAMETERS']['NGRIDK'] = f'{mesh[0]} {mesh[1]} {mesh[2]}'
        parameters['FORCE_EVAL']['PW_DFT']['CONTROL']['MPI_GRID_DIMS'] = f'{1} {settings.job_script["geopt"]["ntasks"]}'
        cell = parameters['FORCE_EVAL']['SUBSYS']['CELL']
        for i, keys in enumerate(cell.keys()):
            cell[keys] = f'{cell[keys]} {round(structure.cell[i][0],14):<15} {round(structure.cell[i][1],14):<15} {round(structure.cell[i][2],14):<15}'
        dict_merge(parameters, get_kinds_section_SIRIUS(structure, basis_pseudo))
        builder.cp2k.file = get_file_section_SIRIUS(structure, basis_pseudo)
        pw_cutoff, gk_cutoff = get_cutoff_SIRIUS(structure, basis_pseudo)
        if job_type in ['opt1vc', 'opt1c']:
            pw_cutoff = round(pw_cutoff * 0.8)
            gk_cutoff = round(gk_cutoff * 0.8)
            if pw_cutoff < 2 * gk_cutoff:
                pw_cutoff = 2 * gk_cutoff
        parameters['FORCE_EVAL']['PW_DFT']['PARAMETERS']['PW_CUTOFF'] = pw_cutoff
        parameters['FORCE_EVAL']['PW_DFT']['PARAMETERS']['GK_CUTOFF'] = gk_cutoff
        dict_merge(parameters, get_kinds_section_SIRIUS(structure, basis_pseudo))
    if job_type in ['opt1c', 'opt2_cluster', 'single_point_cluster', 'molecule', 'dimer']:
        periodic = None
    else:
        periodic = 'XYZ'
    parameters['FORCE_EVAL']['SUBSYS']['CELL']['PERIODIC'] = periodic
    builder.cp2k.parameters = Dict(dict=parameters)
    builder.cp2k.code = Code.get_from_string(settings.configs['aiida_settings']['DFT_code_string'])
    builder.cp2k.settings = Dict(dict={
        'additional_retrieve_list': ['aiida.inp',
                                     'aiida-pos-1.xyz',
                                     'aiida-frc-1.xyz',
                                     'aiida-1.cell',
                                     'aiida-s_p_forces-1_0.xyz',
                                     'aiida-s_p_stress_tensor-1_0.stress_tensor',
                                     'aiida-1.stress',
                                     'aiida.coords.xyz']})
    builder.cp2k.metadata.options = get_options()
    if job_type in ['opt1vc', 'opt1c']:
        builder.cp2k.metadata.options['parser_name'] = 'cp2k_simple_parser'
    else:
        builder.cp2k.metadata.options['parser_name'] = 'cp2k_efs_parser'
    builder.handler_overrides = Dict(dict={'restart_incomplete_calculation': {'enabled': False}})
    builder.max_iterations = Int(1)
    builder.cp2k.metadata['label'] = job_type
    return builder


def get_scaled_structure(scale_factor, structure):
    pymatgen_structure = structure.get_pymatgen()
    pymatgen_structure.scale_lattice(pymatgen_structure.volume*scale_factor)
    return StructureData(pymatgen=pymatgen_structure)

class DimerGeOptWorkChain(WorkChain):
    """ CP2K WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('QSorSIRIUS', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_dimer,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        if 'QS' in self.inputs.QSorSIRIUS.value:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_QS.yml')
        else:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_SIRIUS.yml')
        with open(protocol, 'r', encoding='utf8') as fhandle:
            self.ctx.protocol = yaml.safe_load(fhandle)

    def run_dimer(self):
        structure = self.inputs.structure
        QSorSIRIUS = self.inputs.QSorSIRIUS.value
        # parameters
        parameters = self.ctx.protocol['dimer']
        basis_pseudo = self.ctx.protocol['basis_pseudo']
        parameters['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'
        parameters['### JOB_TYPE'] = 'dimer'
        # builder
        builder = construct_builder(structure, parameters, basis_pseudo, QSorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'dimer': future})

    def inspect_calculation(self):
        if not self.ctx['dimer'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['dimer']
        results_step1_group.add_nodes(a_node)

class Scheme1GeOptWorkChain(WorkChain):
    """ CP2K WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('QSorSIRIUS', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_opt1vc,
            cls.inspect_calculation_1,
            cls.run_single_point_1,
            cls.inspect_calculation_2,
            cls.run_single_point_2,
            cls.inspect_calculation_3)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        if 'QS' in self.inputs.QSorSIRIUS.value:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_QS.yml')
        else:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_SIRIUS.yml')
        with open(protocol, 'r', encoding='utf8') as fhandle:
            self.ctx.protocol = yaml.safe_load(fhandle)

    def run_opt1vc(self):
        structure = self.inputs.structure
        QSorSIRIUS = self.inputs.QSorSIRIUS.value
        # parameters
        basis_pseudo = self.ctx.protocol['basis_pseudo']
        parameters = self.ctx.protocol['opt1vc']
        parameters['GLOBAL']['RUN_TYPE'] = 'CELL_OPT'
        parameters['MOTION']['CELL_OPT']['MAX_FORCE'] = "[bohr^-1*hartree] 0.00194469" # 0.1 eV/A
        parameters['### JOB_TYPE'] = 'opt1vc'
        # builder
        builder = construct_builder(structure, parameters, basis_pseudo, QSorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1vc': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1vc'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_single_point_1(self):
        structure = self.ctx['opt1vc'].outputs['output_structure']
        QSorSIRIUS = self.inputs.QSorSIRIUS.value
        # parameters
        basis_pseudo = self.ctx.protocol['basis_pseudo']
        parameters = deepcopy(self.ctx.protocol['single_point'])
        parameters['GLOBAL']['RUN_TYPE'] = 'ENERGY_FORCE'
        parameters['### JOB_TYPE'] = 'single_point_bulk'
        # builder
        builder = construct_builder(structure, parameters, basis_pseudo, QSorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'single_point_bulk': future})

    def inspect_calculation_2(self):
        if not self.ctx['single_point_bulk'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['single_point_bulk']
        results_step3_group.add_nodes(a_node)

    def run_single_point_2(self):
        structure = self.ctx['opt1vc'].outputs['output_structure']
        scaled_structure = get_scaled_structure(uniform(0.75,0.95), structure)
        QSorSIRIUS = self.inputs.QSorSIRIUS.value
        # parameters
        basis_pseudo = self.ctx.protocol['basis_pseudo']
        parameters = deepcopy(self.ctx.protocol['single_point'])
        parameters['GLOBAL']['RUN_TYPE'] = 'ENERGY_FORCE'
        parameters['### JOB_TYPE'] = 'single_point_scaled_bulk'
        # builder
        builder = construct_builder(scaled_structure, parameters, basis_pseudo, QSorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'scaled_bulk': future})

    def inspect_calculation_3(self):
        if not self.ctx['scaled_bulk'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['scaled_bulk']
        results_step3_group.add_nodes(a_node)

class Scheme2GeOptWorkChain(WorkChain):
    """ CP2K WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('QSorSIRIUS', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_opt1vc,
            cls.inspect_calculation_1,
            cls.run_single_point_1,
            cls.inspect_calculation_2,
            cls.run_single_point_2,
            cls.inspect_calculation_3)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        if 'QS' in self.inputs.QSorSIRIUS.value:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_QS.yml')
        else:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_SIRIUS.yml')
        with open(protocol, 'r', encoding='utf8') as fhandle:
            self.ctx.protocol = yaml.safe_load(fhandle)

    def run_opt1vc(self):
        structure = self.inputs.structure
        QSorSIRIUS = self.inputs.QSorSIRIUS.value
        # parameters
        basis_pseudo = self.ctx.protocol['basis_pseudo']
        parameters = self.ctx.protocol['opt1vc']
        parameters['GLOBAL']['RUN_TYPE'] = 'CELL_OPT'
        parameters['### JOB_TYPE'] = 'opt1vc'
        # builder
        builder = construct_builder(structure, parameters, basis_pseudo, QSorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1vc': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1vc'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_single_point_1(self):
        structure = self.ctx['opt1vc'].outputs['output_structure']
        scaled_structure = get_scaled_structure(uniform(0.75,0.85), structure)
        QSorSIRIUS = self.inputs.QSorSIRIUS.value
        # parameters
        basis_pseudo = self.ctx.protocol['basis_pseudo']
        parameters = deepcopy(self.ctx.protocol['single_point'])
        parameters['GLOBAL']['RUN_TYPE'] = 'ENERGY_FORCE'
        parameters['### JOB_TYPE'] = 'single_point_scaled_bulk'
        # builder
        builder = construct_builder(scaled_structure, parameters, basis_pseudo, QSorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'scaled_bulk_1': future})

    def inspect_calculation_2(self):
        if not self.ctx['scaled_bulk_1'].is_finished_ok:
            self.report('the calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['scaled_bulk_1']
        results_step3_group.add_nodes(a_node)

    def run_single_point_2(self):
        structure = self.ctx['opt1vc'].outputs['output_structure']
        scaled_structure = get_scaled_structure(uniform(0.65,0.75), structure)
        QSorSIRIUS = self.inputs.QSorSIRIUS.value
        # parameters
        basis_pseudo = self.ctx.protocol['basis_pseudo']
        parameters = deepcopy(self.ctx.protocol['single_point'])
        parameters['GLOBAL']['RUN_TYPE'] = 'ENERGY_FORCE'
        parameters['### JOB_TYPE'] = 'single_point_scaled_bulk'
        # builder
        builder = construct_builder(scaled_structure, parameters, basis_pseudo, QSorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'scaled_bulk_2': future})

    def inspect_calculation_3(self):
        if not self.ctx['scaled_bulk_2'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['scaled_bulk_2']
        results_step3_group.add_nodes(a_node)

class Scheme3GeOptWorkChain(WorkChain):
    """ CP2K WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('QSorSIRIUS', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_bulk,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        if 'QS' in self.inputs.QSorSIRIUS.value:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_QS.yml')
        else:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_SIRIUS.yml')
        with open(protocol, 'r', encoding='utf8') as fhandle:
            self.ctx.protocol = yaml.safe_load(fhandle)

    def run_bulk(self):
        structure = self.inputs.structure
        QSorSIRIUS = self.inputs.QSorSIRIUS.value
        # parameters
        parameters = self.ctx.protocol['bulk']
        basis_pseudo = self.ctx.protocol['basis_pseudo']
        parameters['GLOBAL']['RUN_TYPE'] = 'CELL_OPT'
        parameters['### JOB_TYPE'] = 'bulk'
        # builder
        builder = construct_builder(structure, parameters, basis_pseudo, QSorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'bulk': future})

    def inspect_calculation(self):
        if not self.ctx['bulk'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['bulk']
        results_step3_group.add_nodes(a_node)

class ClusterGeOptWorkChain(WorkChain):
    """ CP2K WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('QSorSIRIUS', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_opt1c,
            cls.inspect_calculation_1,
            cls.run_single_point,
            cls.inspect_calculation_2)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        if 'QS' in self.inputs.QSorSIRIUS.value:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_QS.yml')
        else:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_SIRIUS.yml')
        with open(protocol, 'r', encoding='utf8') as fhandle:
            self.ctx.protocol = yaml.safe_load(fhandle)

    def run_opt1c(self):
        structure = self.inputs.structure
        QSorSIRIUS = self.inputs.QSorSIRIUS.value
        # input parameters
        parameters = self.ctx.protocol['opt1']
        basis_pseudo = self.ctx.protocol['basis_pseudo']
        parameters['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'
        parameters['### JOB_TYPE'] = 'opt1c'
        # builder
        builder = construct_builder(structure, parameters, basis_pseudo, QSorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1c': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1c'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_single_point(self):
        structure = self.ctx['opt1c'].outputs['output_structure']
        QSorSIRIUS = self.inputs.QSorSIRIUS.value
        # input parameters
        parameters = self.ctx.protocol['single_point']
        basis_pseudo = self.ctx.protocol['basis_pseudo']
        parameters['kpoints_distance'] = 10
        parameters['GLOBAL']['RUN_TYPE'] = 'ENERGY_FORCE'
        parameters['### JOB_TYPE'] = 'single_point_cluster'
        # builder
        builder = construct_builder(structure, parameters, basis_pseudo, QSorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'single_point_cluster': future})

    def inspect_calculation_2(self):
        if not self.ctx['single_point_cluster'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['single_point_cluster']
        results_step3_group.add_nodes(a_node)

class MoleculeGeOptWorkChain(WorkChain):
    """ CP2K WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('QSorSIRIUS', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_molecule,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        if 'QS' in self.inputs.QSorSIRIUS.value:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_QS.yml')
        else:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_SIRIUS.yml')
        with open(protocol, 'r', encoding='utf8') as fhandle:
            self.ctx.protocol = yaml.safe_load(fhandle)

    def run_molecule(self):
        structure = self.inputs.structure
        QSorSIRIUS = self.inputs.QSorSIRIUS.value
        # input parameters
        parameters = self.ctx.protocol['bulk']
        basis_pseudo = self.ctx.protocol['basis_pseudo']
        parameters['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'
        parameters['### JOB_TYPE'] = 'molecule'
        # builder
        builder = construct_builder(structure, parameters, basis_pseudo, QSorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'molecule': future})

    def inspect_calculation(self):
        if not self.ctx['molecule'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['molecule']
        results_step3_group.add_nodes(a_node)

class SinglePointtWorkChain(WorkChain):
    """ CP2K WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('QSorSIRIUS', valid_type=Str)
        spec.input('bc', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_single_point,
            cls.inspect_calculation_1)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        if 'QS' in self.inputs.QSorSIRIUS.value:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_QS.yml')
        else:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_SIRIUS.yml')
        with open(protocol, 'r', encoding='utf8') as fhandle:
            self.ctx.protocol = yaml.safe_load(fhandle)

    def run_single_point(self):
        structure = self.inputs.structure
        QSorSIRIUS = self.inputs.QSorSIRIUS.value
        # parameters
        parameters = self.ctx.protocol['single_point']
        basis_pseudo = self.ctx.protocol['basis_pseudo']
        parameters['GLOBAL']['RUN_TYPE'] = 'ENERGY_FORCE'

        bc = self.inputs.bc.value
        if 'free' in bc:
            job_type = 'single_point_cluster'
            parameters['kpoints_distance'] = 10
        else:
            job_type = 'single_point_bulk'
        parameters['### JOB_TYPE'] = job_type
        # builder
        builder = construct_builder(structure, parameters, basis_pseudo, QSorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'single_point': future})

    def inspect_calculation_1(self):
        if not self.ctx['single_point'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['single_point']
        results_step3_group.add_nodes(a_node)
