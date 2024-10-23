import os
import collections
from random import randint
from itertools import combinations_with_replacement
import yaml
from aiida.orm import Group, Dict, Str, Int, Code
from aiida.plugins import CalculationFactory
from aiida.engine import WorkChain
from pymatgen.core.composition import Composition
import aiida_datagen.workflows.settings as settings

def dict_merge(dct, merge_dct):
    """ Taken from https://gist.github.com/angstwad/
    """
    for k in merge_dct.keys():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], collections.abc.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def get_options(job_type, max_wallclock = False):
    job_script = settings.job_script
    if 'averdist' in job_type:
        resources = {
            'num_machines': 1,
            'num_mpiprocs_per_machine': job_script['QBC']['ntasks']}
        job_type = 'QBC'
    else:
        resources = {
            'num_machines': job_script[job_type]['nodes'],
            'num_mpiprocs_per_machine': job_script[job_type]['ntasks']}
    options = {'resources': resources}
    if job_script[job_type]['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
    if max_wallclock:
        options['max_wallclock_seconds'] = max_wallclock
    else:
        options['max_wallclock_seconds'] = job_script[job_type]['time']
    return options

class GenSymCrysWorkChain(WorkChain):
    """ FLAME calculation for generating crystal structures
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('composition', valid_type=Str)
        spec.input('n_atom', valid_type=Int)
        spec.outline(
            cls.initialize,
            cls.run_gensymcrys,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message="The calculation did not finish successfully")

    def initialize(self):
        """ Initialize
        """
        composition = self.inputs.composition.value
        with open(os.path.join(settings.FLAME_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.flame_in = yaml.safe_load(fhandle)
        self.ctx.elements = []
        self.ctx.n_elmnt = []
        for elmnt, nelmnt in Composition(composition).items():
            self.ctx.elements.append(str(elmnt))
            self.ctx.n_elmnt.append(int(nelmnt))

    def run_gensymcrys(self):
        """ Run
        """
        n_atom = self.inputs.n_atom.value
        # input parameters
        parameters = self.ctx.flame_in['gensymcrys']
        additional_parameters = self._get_additional_parameters(self.ctx.elements, self.ctx.n_elmnt, n_atom)
        dict_merge(parameters, additional_parameters)
        # builder
        builder = self._construct_builder(parameters)
        # submit
        future = self.submit(builder)
        self.to_context(**{'gensymcrys': future})

    def inspect_calculation(self):
        """ Inspect
        """
        if not self.ctx['gensymcrys'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    @staticmethod
    def _construct_builder(parameters):
        Workflow = CalculationFactory('flame')
        builder = Workflow.get_builder()
        builder.parameters = Dict(dict=parameters)
        builder.job_type_info = Dict(dict={'gensymcrys':{}})
        builder.code = Code.get_from_string(settings.configs['aiida_settings']['FLAME_code_string'])
        builder.settings = Dict(dict={
            'additional_retrieve_list': ['posout.yaml'],
            'retrieve_temporary_list':[]})
        builder.metadata['label'] = 'gensymcrys'
        builder.metadata.options = get_options('gensymcrys')
        builder.metadata.options.parser_name = 'datagen_gensymcrys_parser'
        return builder

    @staticmethod
    def _get_additional_parameters(elements, n_elmnt, n_atom):
        vpas = []
        dimers = {}
        known_structures_group = Group.collection.get(label='known_structures')
        for a_node in known_structures_group.nodes:
            if 'vpas' in a_node.label:
                vpas = a_node.get_list()
            if 'dimers' in a_node.label:
                dimers = a_node.get_dict()
        pairs = {}
        min_d_prefactor = min(0.90, settings.inputs['min_distance_prefactor'])
        for a_pair in combinations_with_replacement(elements,2):
            pairs[''.join(a_pair)] = dimers['-'.join(a_pair)] * min_d_prefactor

        additional_parameters = {}
        additional_parameters['main'] = {}
        additional_parameters['main']['types'] = ' '.join(elements)
        additional_parameters['main']['seed'] = randint(1,10**randint(1,6))
        additional_parameters['genconf'] = {}
        additional_parameters['genconf']['volperatom_bounds'] = [vpas[0], vpas[1]]
        additional_parameters['genconf']['nat_types_fu'] = n_elmnt
        additional_parameters['genconf']['list_fu'] = [int(n_atom/sum(n_elmnt))]
        additional_parameters['genconf']['nconf'] = 230
        additional_parameters['genconf']['rmin_pairs'] = pairs
        return additional_parameters
