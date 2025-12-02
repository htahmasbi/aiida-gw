from aiida.orm import Int, Str, List, Dict
from aiida.plugins import WorkflowFactory
from aiida.engine import WorkChain
from aiida_datagen.workflows import settings

def get_code(model_key):
    """
    Helper to fetch builder.code, model_path, and device
    """
    return load_code(
        settings.configs["codes"][model_key]["code_string"])

class GenWorkChain(WorkChain):
    """Work chain for generating structures"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("chemical_systems", valid_type=List)
        spec.input("ML_model", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.generative_calcs,
            cls.inspect_gen_calcs,
            cls.store_results,
        )

        spec.exit_code(301, "ERROR_GENERATIVE_FAILED", message="MatterGen generative calculation failed")
        spec.exit_code(302, "ERROR_ML_RELAX_FAILED", message="ML relaxation failed")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_systems = self.inputs.chemical_systems.get_list()
        self.ctx.ML_model = self.inputs.ML_model.value
        self.ctx.failed_ml_e = []
        self.report(f"Launching MatterGenWorkChain for {self.ctx.chemical_systems}")

    def generative_calcs(self):
        """Run MatterGen workchains for each unique chemical system"""
        for chemical_system in self.ctx.chemical_systems:
            builder = self._construct_mattergen_gen_builder(chemical_system)
            future = self.submit(builder)
            self.to_context(**{f"{chemical_system}_mattergen": future})

    def inspect_gen_calcs(self):
        """Check for any failures among MatterGen calculations"""
        for chemical_system in self.ctx.chemical_systems:
            calculation = self.ctx[f"{chemical_system}_mattergen"]
            if not calculation.is_finished_ok:
                return self.exit_codes.ERROR_GENERATIVE_FAILED

    def store_results(self):
        """Predict the energies of the structures with the given ML model"""
        chemical_systems = self.ctx.chemical_systems
        for chemical_system in chemical_systems:

            wch = self.ctx[f"{chemical_system}_mattergen"]
            output_structures = (
                    wch.called[-1]
                    .outputs.output_dict["structures"]
            )
################################################################################
    @staticmethod
    def _construct_mattergen_gen_builder(chemical_system):
        """MatterGen gen Builder"""
        Workflow = WorkflowFactory("mattergen.base")
        builder = Workflow.get_builder()
        builder.chemical_system = Str(chemical_system)
        builder.code = get_code("MatterGen")

        builder.job_info = Dict(
                {"job_type": "gen",
                 "model_name":  settings.configs["codes"]["MatterGen"]["model_name"],
                 "energy_above_hull": settings.inputs["MatterGen_generate"]["energy_above_hull"],
                 "batch_size": settings.inputs["MatterGen_generate"]["batch_size"],
                 "num_batches": settings.inputs["MatterGen_generate"]["num_batches"]
                }
        )
        builder.max_iterations = Int(2)
        return builder
