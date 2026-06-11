from __future__ import annotations

import logging

from aiida.engine import WorkChain, submit
from aiida.orm import Code, Dict, Int, List, Str
from aiida.plugins import WorkflowFactory

from aiida_gw.core.config import ProjectConfig, get_config
from aiida_gw.core.exceptions import WorkflowError

logger = logging.getLogger(__name__)

Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")


class SinglePointWorkChain(WorkChain):
    """Workflow for a single-point CP2K calculation using protocol files."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=Cp2kBaseWorkChain._spec.inputs["structure"])
        spec.input("code", valid_type=Code)
        spec.input("parameters", valid_type=Dict, required=False)
        spec.input(
            "protocol_name",
            valid_type=Str,
            default=Str("protocol_SIRIUS.yml"),
        )
        spec.input("section", valid_type=Str, default=Str("single_point"))
        spec.input("kpoints_mesh", valid_type=List, required=False)
        spec.input("metadata_options", valid_type=Dict, required=False)
        spec.outline(
            cls.setup,
            cls.run_cp2k,
            cls.finalize,
        )
        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="CP2K calculation failed")
        spec.output("output_structure", valid_type=Cp2kBaseWorkChain._spec.inputs["structure"])
        spec.output("output_parameters", valid_type=Dict)

    def setup(self):
        self.ctx.config = get_config()

    def run_cp2k(self):
        from aiida_gw.core.builders import Cp2kBuilder

        builder = Cp2kBuilder(self.ctx.config)
        kpoints_mesh = (
            self.inputs.kpoints_mesh.get_list()
            if "kpoints_mesh" in self.inputs
            else self.ctx.config.cp2k.kpoints_mesh
        )

        inputs = builder.build_scf_inputs(
            structure=self.inputs.structure,
            code=self.inputs.code,
            protocol_section=self.inputs.section.value,
            protocol_name=self.inputs.protocol_name.value,
            kpoints_mesh=kpoints_mesh,
        )
        if self.inputs.metadata_options:
            inputs.cp2k.metadata.options = self.inputs.metadata_options.get_dict()

        future = self.submit(inputs)
        self.to_context(cp2k_calc=future)

    def finalize(self):
        if not self.ctx.cp2k_calc.is_finished_ok:
            return self.exit_codes.ERROR_CALCULATION_FAILED
        self.out("output_structure", self.ctx.cp2k_calc.outputs.output_structure)
        self.out("output_parameters", self.ctx.cp2k_calc.outputs.output_parameters)


def run_single_point(
    structure,
    code_label: str = "cp2k@localhost",
    config: ProjectConfig | None = None,
) -> int:
    """Run a single-point calculation and return the workchain PK."""
    from aiida.engine import submit
    from aiida.orm import load_code

    from aiida_gw.core.config import get_config

    cfg = config or get_config()
    code = load_code(code_label)

    builder = SinglePointWorkChain.get_builder()
    builder.structure = structure
    builder.code = code
    builder.protocol_name = Str("protocol_SIRIUS.yml")
    builder.section = Str("single_point")

    node = submit(builder)
    logger.info(f"Submitted SinglePointWorkChain<{node.pk}>")
    return node.pk
