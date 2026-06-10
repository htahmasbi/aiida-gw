import yaml
from aiida.engine import CalcJob
from aiida.orm import Dict, SinglefileData
from aiida.common import CalcInfo, CodeInfo

class FlameCalculation(CalcJob):
    """ A subclass of JobCalculation, to prepare input for FLAME calculations
    """
    _INPUT_FILE = "flame_in.yaml"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("parameters", valid_type=Dict)
        spec.input("job_type_info", valid_type=Dict)
        spec.input_namespace("file", valid_type=(SinglefileData), required=False, dynamic=True)
        spec.input("settings", valid_type=Dict, required=False)
        spec.input("metadata.options.withmpi", valid_type=bool, default=True)

        spec.output("output_parameters", valid_type=Dict, required=False)

        spec.exit_code(
            200,
            "ERROR_NO_RETRIEVED_FOLDER",
            message="The retrieved folder data node can not be accessed"
        )
        spec.exit_code(
            302, "ERROR_OUTPUT_PARSE",
            message="The output file can not be parsed."
        )
        spec.exit_code(
            303, "ERROR_OUTPUT_INCOMPLETE",
            message="The output file is incomplete."
        )

    def prepare_for_submission(self, folder):
        """ Create input files
        """
        inp = self.inputs.parameters.get_dict()

        with folder.open(self._INPUT_FILE, 'w', encoding='utf-8') as fhandle:
            yaml.dump(inp, fhandle, default_flow_style=False)

        provenance_exclude_list = []

        settings = self.inputs.settings.get_dict() if "settings" in self.inputs else {}
        # Code info.
        codeinfo = CodeInfo()
        codeinfo.cmdline_params = []
        codeinfo.withmpi = self.metadata.options.withmpi
        codeinfo.code_uuid = self.inputs.code.uuid
        # Calc info.
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.cmdline_params = codeinfo.cmdline_params
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list += settings.pop("additional_retrieve_list", [])
        calcinfo.retrieve_temporary_list = settings.pop("retrieve_temporary_list", [])
        calcinfo.codes_info = [codeinfo]
        if "file" in self.inputs:
            calcinfo.local_copy_list = []
            for _, obj in self.inputs.file.items():
                if isinstance(obj, SinglefileData):
                    calcinfo.local_copy_list.append((obj.uuid, obj.filename, obj.filename))
        calcinfo.provenance_exclude_list = provenance_exclude_list
        return calcinfo
