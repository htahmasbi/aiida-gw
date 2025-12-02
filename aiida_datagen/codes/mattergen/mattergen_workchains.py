from aiida.engine import BaseRestartWorkChain, while_
from aiida.orm import Int, Str, Dict, Code
from aiida.plugins import CalculationFactory
from aiida_datagen.workflows import settings

def get_options():
    """Return scheduler options"""
    job_script = settings.configs['job_script']['mattergen']
    resources = {
        'num_machines': job_script['nodes'],
        'num_mpiprocs_per_machine': job_script['ntasks'],
        'num_cores_per_mpiproc': job_script['ncpu'],
    }
    options = {
        'resources': resources,
        'max_wallclock_seconds': job_script['time'],
        'parser_name': 'mattergen_parser',
        'append_text': 'python aiida.py'
    }
    if job_script['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
    return options

def get_cmdline(chemical_system, job_info, diffusion_guidance_factor = 2.0):
    """Construct MatterGen command line"""

    return [
        "$RESULTS_PATH",
        f"--pretrained-name={job_info['model_name']}",
        f"--batch_size={job_info['batch_size']}",
        f"--num_batches={job_info['num_batches']}",
        f"--properties_to_condition_on={{'energy_above_hull': {job_info['energy_above_hull']}, 'chemical_system': '{chemical_system}'}}",
        f"--diffusion_guidance_factor={diffusion_guidance_factor}",
        "--record-trajectories=False"
    ]

MatterGenCalculation = CalculationFactory('mattergen')

class MatterGenBaseWorkChain(BaseRestartWorkChain):
    """BaseRestartWorkChain to run MatterGenCalculation with automatic restarts"""

    _process_class = MatterGenCalculation

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Declare the inputs needed for this workchain:
        spec.input('chemical_system', valid_type=Str)
        spec.input('batch_size', valid_type=Int)
        spec.input('num_batches', valid_type=Int)

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.exit_code(
            400,
            'ERROR_MAX_RESTARTS_EXCEEDED',
            message='Maximum number of restarts exceeded for MatterGenBaseWorkChain.'
        )

    def setup(self):
        """Initialize context before first calculation."""
        super().setup()
        chemical_system = self.inputs.chemical_system.value
        job_info = {'model_name': settings.configs['MatterGen_model_name'],
                    'batch_size': self.inputs.batch_size.value,
                    'num_batches': self.inputs.num_batches.value,
                    'energy_above_hull': 0.5,
                    'diffusion_guidance_factor': 2
                    }
        cmdline = get_cmdline(chemical_system, job_info)

        self.ctx.inputs = {
            'code': Code.get_from_string(settings.configs['aiida_settings']['MatterGen_code_string']),
            'parameters': Dict(dict={'cmdline_params': cmdline}),
            'metadata': {
                'options': get_options(),
                'label': 'MatterGen calculation'
            }
        }
