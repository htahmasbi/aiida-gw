import os
from random import uniform
import yaml
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.engine import WorkChain
from aiida.orm import Int, Str, List, Dict, Code, Bool, SinglefileData, load_group
import aiida_datagen.workflows.settings as settings

results_step1_group = load_group(label='results_step1')
results_step3_group = load_group(label='results_step3')
StructureData = DataFactory('structure')

def get_options():
    job_script = settings.job_script
    resources = {
        'num_machines': job_script['geopt']['nodes'],
        'num_mpiprocs_per_machine': job_script['geopt']['ntasks']}
    if job_script['geopt']['ncpu']:
        resources['num_cores_per_mpiproc'] = job_script['geopt']['ncpu']
    options = {'resources': resources,
               'max_wallclock_seconds': job_script['geopt']['time'],
               }
    if job_script['geopt']['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
    return options

def construct_builder(structure, protocol, potential_family, potential_mapping):
    Workflow = WorkflowFactory('vasp.vasp')
    builder = Workflow.get_builder()
    builder.structure = structure
    builder.parameters = Dict(dict={'incar':protocol['incar']})
    builder.potential_family = Str(potential_family)
    builder.potential_mapping = Dict(dict=potential_mapping)
    if 'LUSE_VDW' in protocol['incar'].keys() and protocol['incar']['LUSE_VDW']:
        with open(os.path.join(settings.VASP_input_files_path,'vdw_kernel.bindat'), 'rb') as handle:
            vdw_kernel = SinglefileData(file=handle)
        builder.file = {'vdw_kernel':vdw_kernel.bindat}
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    if protocol['name'] in ['opt1c', 'molecule', 'single_point_cluster']:
        kpoints.set_kpoints_mesh_from_density(10)
    else:
        kpoints.set_kpoints_mesh_from_density(protocol['kpoint_distance'])
    builder.kpoints = kpoints
    builder.code = Code.get_from_string(settings.configs['aiida_settings']['DFT_code_string'])
    if protocol['name'] in ['opt1vc', 'opt1c']:
        parser_settings = {'add_structure': True}
    else:
        parser_settings = {'add_structure': True,
                           'add_trajectory': True,
                           'add_energies': True,
                           }
    builder.settings = Dict(dict={'parser_settings': parser_settings})
    if 'dimer' in protocol['name']:
        builder.dynamics.positions_dof = List([[False, False, False], [False, False, True]])
#    builder.settings = Dict(dict={'CHECK_IONIC_CONVERGENCE': False})
    builder.options = Dict(dict=get_options())
    builder.clean_workdir = Bool(False)
    if protocol['name'] in ['bulk', 'dimer']:
        builder.max_iterations = Int(2)
    else:
        builder.max_iterations = Int(1)
    builder.verbose = Bool(True)
    builder.metadata['label'] = protocol['name']
    return builder

def get_scaled_structure(scale_factor, structure):
    pymatgen_structure = structure.get_pymatgen()
    pymatgen_structure.scale_lattice(pymatgen_structure.volume*scale_factor)
    return StructureData(pymatgen=pymatgen_structure)

class DimerGeOptWorkChain(WorkChain):
    """ VASP WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=(DataFactory('structure')))
        spec.outline(
            cls.initialize,
            cls.run_dimer,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential = yaml.safe_load(fhandle)

    def run_dimer(self):
        structure = self.inputs.structure
        protocol = self.ctx.vasp_protocol['dimer']
        protocol['name'] = 'dimer'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(structure, protocol, potential_family, potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'dimer': future})

    def inspect_calculation(self):
        if not self.ctx['dimer'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['dimer']
        results_step1_group.add_nodes(a_node)

class RefGeOptWorkChain(WorkChain):
    """ VASP WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=(DataFactory('structure')))
        spec.outline(
            cls.initialize,
            cls.run_bulk,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential = yaml.safe_load(fhandle)

    def run_bulk(self):
        structure = self.inputs.structure
        protocol = self.ctx.vasp_protocol['bulk']
        protocol['name'] = 'bulk'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(structure, protocol, potential_family, potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'bulk': future})

    def inspect_calculation(self):
        if not self.ctx['bulk'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['bulk']
        results_step1_group.add_nodes(a_node)

class Scheme1GeOptWorkChain(WorkChain):
    """ VASP WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
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
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential = yaml.safe_load(fhandle)

    def run_opt1vc(self):
        structure = self.inputs.structure
        protocol = self.ctx.vasp_protocol['opt1vc']
        protocol['incar']['EDIFFG'] = -0.1
        protocol['name'] = 'opt1vc'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(structure, protocol, potential_family, potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1vc': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1vc'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_single_point_1(self):
        structure = self.ctx['opt1vc'].outputs['structure']
        protocol = self.ctx.vasp_protocol['single_point']
        protocol['name'] = 'single_point_bulk'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(structure, protocol, potential_family, potential_mapping)
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
        structure = self.ctx['opt1vc'].outputs['structure']
        scaled_structure = get_scaled_structure(uniform(0.75,0.95), structure)
        protocol = self.ctx.vasp_protocol['single_point']
        protocol['name'] = 'single_point_scaled_bulk'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(scaled_structure, protocol, potential_family, potential_mapping)
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
    """ VASP WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
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
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential = yaml.safe_load(fhandle)

    def run_opt1vc(self):
        structure = self.inputs.structure
        protocol = self.ctx.vasp_protocol['opt1vc']
        protocol['name'] = 'opt1vc'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(structure, protocol, potential_family, potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1vc': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1vc'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_single_point_1(self):
        structure = self.ctx['opt1vc'].outputs['structure']
        scaled_structure = get_scaled_structure(uniform(0.75,0.85), structure)
        protocol = self.ctx.vasp_protocol['single_point']
        protocol['name'] = 'single_point_scaled_bulk'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(scaled_structure, protocol, potential_family, potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'scaled_bulk_1': future})

    def inspect_calculation_2(self):
        if not self.ctx['scaled_bulk_1'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['scaled_bulk_1']
        results_step3_group.add_nodes(a_node)

    def run_single_point_2(self):
        structure = self.ctx['opt1vc'].outputs['structure']
        scaled_structure = get_scaled_structure(uniform(0.65,0.75), structure)
        protocol = self.ctx.vasp_protocol['single_point']
        protocol['name'] = 'single_point_scaled_bulk'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(scaled_structure, protocol, potential_family, potential_mapping)
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
    """ VASP WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=(DataFactory('structure')))
        spec.outline(
            cls.initialize,
            cls.run_bulk,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential = yaml.safe_load(fhandle)

    def run_bulk(self):
        structure = self.inputs.structure
        protocol = self.ctx.vasp_protocol['bulk']
        protocol['name'] = 'bulk'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(structure, protocol, potential_family, potential_mapping)
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
    """ VASP WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
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
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential = yaml.safe_load(fhandle)

    def run_opt1c(self):
        structure = self.inputs.structure
        protocol = self.ctx.vasp_protocol['opt1']
        protocol['name'] = 'opt1c'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(structure, protocol, potential_family, potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1c': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1c'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_single_point(self):
        structure = self.ctx['opt1c'].outputs['structure']
        protocol = self.ctx.vasp_protocol['single_point']
        protocol['name'] = 'single_point_cluster'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(structure, protocol, potential_family, potential_mapping)
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
    """ VASP WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=(DataFactory('structure')))
        spec.outline(
            cls.initialize,
            cls.run_molecule,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential = yaml.safe_load(fhandle)

    def run_bulk(self):
        structure = self.inputs.structure
        protocol = self.ctx.vasp_protocol['molecule']
        protocol['name'] = 'molecule'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(structure, protocol, potential_family, potential_mapping)
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
    """ VASP WorkChain
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('bc', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_single_point,
            cls.inspect_calculation_1)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential = yaml.safe_load(fhandle)

    def run_single_point(self):
        structure = self.inputs.structure
        bc = self.inputs.bc.value
        protocol = self.ctx.vasp_protocol['single_point']
        if 'free' in bc:
            protocol['name'] = 'single_point_cluster'
        else:
            protocol['name'] = 'single_point_bulk'
        potential_family = self.ctx.potential['potential_family']
        potential_mapping = self.ctx.potential['potential_mapping']
        # builder
        builder = construct_builder(structure, protocol, potential_family, potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'single_point': future})

    def inspect_calculation_1(self):
        if not self.ctx['single_point'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        a_node = self.ctx['single_point']
        results_step3_group.add_nodes(a_node)
