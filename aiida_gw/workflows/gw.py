from __future__ import annotations

import logging

from aiida.engine import WorkChain
from aiida.orm import Code, Dict, Float, Int, List, Str
from aiida.plugins import WorkflowFactory

from aiida_gw.core.config import get_config
from aiida_gw.core.exceptions import WorkflowError

logger = logging.getLogger(__name__)

Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")


class GwWorkChain(WorkChain):
    """Workflow for CP2K GW calculation on a pre-relaxed structure."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure")
        spec.input("code", valid_type=Code)
        spec.input("protocol_name", valid_type=Str, default=Str("protocol_GW.yml"))
        spec.input("kpoints_mesh", valid_type=List, required=False)
        spec.input("kpoints_w_mesh", valid_type=List, required=False)
        spec.input("bandstructure_path", valid_type=List, required=False)
        spec.input("ri_basis_accuracy_target", valid_type=Float, required=False)
        spec.outline(
            cls.setup,
            cls.run_gw,
            cls.finalize,
        )
        spec.exit_code(300, "ERROR_GW_FAILED", message="GW calculation failed")
        spec.output("output_structure")
        spec.output("output_parameters", valid_type=Dict, required=False)

    def setup(self):
        self.ctx.config = get_config()

    def run_gw(self):
        from aiida_gw.core.builders import Cp2kBuilder

        builder = Cp2kBuilder(self.ctx.config)
        gw_config = self.ctx.config.gw

        kmesh = (
            self.inputs.kpoints_mesh.get_list()
            if "kpoints_mesh" in self.inputs
            else gw_config.kpoints_mesh
        )
        kwmesh = (
            self.inputs.kpoints_w_mesh.get_list()
            if "kpoints_w_mesh" in self.inputs
            else gw_config.kpoints_w_mesh
        )

        bs_path = (
            self.inputs.bandstructure_path.get_list()
            if "bandstructure_path" in self.inputs
            else None
        )

        inputs = builder.build_gw_inputs(
            structure=self.inputs.structure,
            code=self.inputs.code,
            protocol_name=self.inputs.protocol_name.value,
            kpoints_mesh=kmesh,
            kpoints_w_mesh=kwmesh,
            kpoints_distance=gw_config.kpoints_distance,
            kpoints_w_distance=gw_config.kpoints_w_distance,
            bandstructure_path=bs_path,
        )
        future = self.submit(inputs)
        self.to_context(gw_calc=future)

    def finalize(self):
        if not self.ctx.gw_calc.is_finished_ok:
            return self.exit_codes.ERROR_GW_FAILED
        self.out("output_structure", self.ctx.gw_calc.outputs.output_structure)
        if hasattr(self.ctx.gw_calc.outputs, "output_parameters"):
            self.out("output_parameters", self.ctx.gw_calc.outputs.output_parameters)
