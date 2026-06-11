from __future__ import annotations

import logging

from aiida.engine import WorkChain
from aiida.orm import Bool, Code, Dict, List, Str
from aiida.plugins import WorkflowFactory

from aiida_gw.core.config import get_config
from aiida_gw.core.exceptions import WorkflowError

logger = logging.getLogger(__name__)

Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")


class RelaxWorkChain(WorkChain):
    """Workflow for geometry relaxation with CP2K."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure")
        spec.input("code", valid_type=Code)
        spec.input("protocol_name", valid_type=Str, default=Str("protocol_SIRIUS.yml"))
        spec.input("section", valid_type=Str, default=Str("opt1vc"))
        spec.input("cell_opt", valid_type=Bool, default=Bool(True))
        spec.input("kpoints_mesh", valid_type=List, required=False)
        spec.outline(
            cls.setup,
            cls.run_relax,
            cls.inspect_relax,
            cls.run_scf,
            cls.inspect_scf,
        )
        spec.exit_code(300, "ERROR_RELAX_FAILED", message="Relaxation calculation failed")
        spec.exit_code(301, "ERROR_SCF_FAILED", message="Post-relax SCF failed")
        spec.output("output_structure")
        spec.output("output_parameters", valid_type=Dict)

    def setup(self):
        self.ctx.config = get_config()

    def run_relax(self):
        from aiida_gw.core.builders import Cp2kBuilder

        kpoints_mesh = (
            self.inputs.kpoints_mesh.get_list()
            if "kpoints_mesh" in self.inputs
            else self.ctx.config.cp2k.kpoints_mesh
        )
        builder = Cp2kBuilder(self.ctx.config)
        section = "opt1vc" if self.inputs.cell_opt.value else "opt1"
        inputs = builder.build_scf_inputs(
            structure=self.inputs.structure,
            code=self.inputs.code,
            protocol_section=section,
            protocol_name=self.inputs.protocol_name.value,
            kpoints_mesh=kpoints_mesh,
        )
        future = self.submit(inputs)
        self.to_context(relax_calc=future)

    def inspect_relax(self):
        if not self.ctx.relax_calc.is_finished_ok:
            return self.exit_codes.ERROR_RELAX_FAILED

    def run_scf(self):
        from aiida_gw.core.builders import Cp2kBuilder

        kpoints_mesh = (
            self.inputs.kpoints_mesh.get_list()
            if "kpoints_mesh" in self.inputs
            else self.ctx.config.cp2k.kpoints_mesh
        )
        relaxed_structure = self.ctx.relax_calc.outputs.output_structure
        builder = Cp2kBuilder(self.ctx.config)
        inputs = builder.build_scf_inputs(
            structure=relaxed_structure,
            code=self.inputs.code,
            protocol_section="single_point",
            protocol_name=self.inputs.protocol_name.value,
            kpoints_mesh=kpoints_mesh,
        )
        future = self.submit(inputs)
        self.to_context(scf_calc=future)

    def inspect_scf(self):
        if not self.ctx.scf_calc.is_finished_ok:
            return self.exit_codes.ERROR_SCF_FAILED
        self.out("output_structure", self.ctx.scf_calc.outputs.output_structure)
        self.out("output_parameters", self.ctx.scf_calc.outputs.output_parameters)
