from __future__ import annotations

import logging

from aiida.engine import WorkChain
from aiida.orm import Bool, Code, Dict, List, Str, StructureData
from aiida.plugins import WorkflowFactory

from aiida_gw.core.config import config_from_inputs
from aiida_gw.core.nodes import get_output_node

logger = logging.getLogger(__name__)

Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")


class RelaxWorkChain(WorkChain):
    """Workflow for geometry relaxation with CP2K."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=StructureData)
        spec.input("code", valid_type=Code)
        spec.input("protocol_name", valid_type=Str, default=Str("protocol_SIRIUS.yml"))
        spec.input("section", valid_type=Str, default=Str("opt1vc"))
        spec.input("cell_opt", valid_type=Bool, default=Bool(True))
        spec.input("kpoints_mesh", valid_type=List, required=False)
        spec.input("config", valid_type=Dict, required=False)
        spec.outline(
            cls.setup,
            cls.run_relax,
            cls.inspect_relax,
            cls.run_scf,
            cls.inspect_scf,
        )
        spec.exit_code(300, "ERROR_RELAX_FAILED", message="Relaxation calculation failed")
        spec.exit_code(301, "ERROR_SCF_FAILED", message="Post-relax SCF failed")
        spec.output("output_structure", valid_type=StructureData)
        spec.output("output_parameters", valid_type=Dict)

    def setup(self):
        self.ctx.config = config_from_inputs(self.inputs)

    def run_relax(self):
        from aiida_gw.core.builders import Cp2kBuilder

        cp2k_cfg = self.ctx.config.cp2k
        kpoints_mesh = (
            self.inputs.kpoints_mesh.get_list()
            if "kpoints_mesh" in self.inputs
            else cp2k_cfg.kpoints_mesh
        )
        builder = Cp2kBuilder(self.ctx.config)
        section = "opt1vc" if self.inputs.cell_opt.value else "opt1"
        inputs = builder.build_scf_inputs(
            structure=self.inputs.structure,
            code=self.inputs.code,
            protocol_section=section,
            protocol_name=self.inputs.protocol_name.value,
            kpoints_mesh=kpoints_mesh,
            kpoints_distance=cp2k_cfg.kpoints_distance,
        )
        future = self.submit(inputs)
        self.to_context(relax_calc=future)

    def inspect_relax(self):
        if not self.ctx.relax_calc.is_finished_ok:
            self.logger.error("Relax calculation failed with exit code %s", self.ctx.relax_calc.exit_status)
            return self.exit_codes.ERROR_RELAX_FAILED
        self.report("Relaxation finished successfully")

    def run_scf(self):
        from aiida_gw.core.builders import Cp2kBuilder

        cp2k_cfg = self.ctx.config.cp2k
        kpoints_mesh = (
            self.inputs.kpoints_mesh.get_list()
            if "kpoints_mesh" in self.inputs
            else cp2k_cfg.kpoints_mesh
        )
        relaxed_structure = get_output_node(self.ctx.relax_calc, "output_structure")
        if relaxed_structure is None:
            self.logger.error("Relax calculation missing output_structure")
            return self.exit_codes.ERROR_RELAX_FAILED
        builder = Cp2kBuilder(self.ctx.config)
        inputs = builder.build_scf_inputs(
            structure=relaxed_structure,
            code=self.inputs.code,
            protocol_section="single_point",
            protocol_name=self.inputs.protocol_name.value,
            kpoints_mesh=kpoints_mesh,
            kpoints_distance=cp2k_cfg.kpoints_distance,
        )
        future = self.submit(inputs)
        self.to_context(scf_calc=future)

    def inspect_scf(self):
        if not self.ctx.scf_calc.is_finished_ok:
            self.logger.error("Post-relax SCF failed with exit code %s", self.ctx.scf_calc.exit_status)
            return self.exit_codes.ERROR_SCF_FAILED
        self.report("Post-relax SCF finished successfully")
        output_structure = get_output_node(self.ctx.scf_calc, "output_structure")
        if output_structure is not None:
            self.out("output_structure", output_structure)
        else:
            self.report("SCF calculation produced no output structure; emitting relaxed structure")
            relaxed_structure = get_output_node(self.ctx.relax_calc, "output_structure")
            if relaxed_structure is not None:
                self.out("output_structure", relaxed_structure)
            else:
                self.out("output_structure", self.inputs.structure)

        output_parameters = get_output_node(self.ctx.scf_calc, "output_parameters")
        if output_parameters is None:
            self.logger.error("Post-relax SCF produced no output_parameters")
            return self.exit_codes.ERROR_SCF_FAILED
        self.out("output_parameters", output_parameters)
