from __future__ import annotations

import logging

from aiida.engine import WorkChain, submit
from aiida.orm import Code, Dict, List, Str, StructureData
from aiida.plugins import WorkflowFactory

from aiida_gw.core.config import ProjectConfig, config_from_inputs, get_config
from aiida_gw.core.nodes import get_output_node

logger = logging.getLogger(__name__)

Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")


class SinglePointWorkChain(WorkChain):
    """Workflow for a single-point CP2K calculation using protocol files."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData)
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
        spec.input("config", valid_type=Dict, required=False)
        spec.outline(
            cls.setup,
            cls.run_cp2k,
            cls.finalize,
        )
        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="CP2K calculation failed")
        spec.output("output_structure", valid_type=StructureData)
        spec.output("output_parameters", valid_type=Dict)

    def setup(self):
        self.ctx.config = config_from_inputs(self.inputs)

    def run_cp2k(self):
        from aiida_gw.core.builders import Cp2kBuilder

        builder = Cp2kBuilder(self.ctx.config)
        cp2k_cfg = self.ctx.config.cp2k
        kpoints_mesh = (
            self.inputs.kpoints_mesh.get_list()
            if "kpoints_mesh" in self.inputs
            else cp2k_cfg.kpoints_mesh
        )

        inputs = builder.build_scf_inputs(
            structure=self.inputs.structure,
            code=self.inputs.code,
            protocol_section=self.inputs.section.value,
            protocol_name=self.inputs.protocol_name.value,
            kpoints_mesh=kpoints_mesh,
            kpoints_distance=cp2k_cfg.kpoints_distance,
        )
        if self.inputs.metadata_options is not None:
            inputs.cp2k.metadata.options = self.inputs.metadata_options.get_dict()

        self.report("Submitting SinglePointWorkChain for structure %s", self.inputs.structure.label)
        future = self.submit(inputs)
        self.to_context(cp2k_calc=future)

    def finalize(self):
        if not self.ctx.cp2k_calc.is_finished_ok:
            self.logger.error("CP2K calculation failed with exit code %s", self.ctx.cp2k_calc.exit_status)
            return self.exit_codes.ERROR_CALCULATION_FAILED
        self.report("Single-point calculation finished successfully")

        output_structure = get_output_node(self.ctx.cp2k_calc, "output_structure")
        if output_structure is not None:
            self.out("output_structure", output_structure)
        else:
            self.report("CP2K calculation produced no output structure; emitting input structure")
            self.out("output_structure", self.inputs.structure)

        output_parameters = get_output_node(self.ctx.cp2k_calc, "output_parameters")
        if output_parameters is None:
            self.logger.error("CP2K calculation produced no output_parameters")
            return self.exit_codes.ERROR_CALCULATION_FAILED
        self.out("output_parameters", output_parameters)


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
    builder.config = Dict(dict=cfg.model_dump())

    node = submit(builder)
    self.report(f"Submitted SinglePointWorkChain<{node.pk}>")
    return node.pk
