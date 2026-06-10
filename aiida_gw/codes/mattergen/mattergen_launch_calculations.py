from aiida.plugins import DataFactory
from aiida.orm import Str, Int
from aiida_submission_controller import BaseSubmissionController
from aiida_gw.codes.mattergen.mattergen_workchains import MatterGenBaseWorkChain

StructureData = DataFactory('structure')

class MatterGenSubmissionController(BaseSubmissionController):
    """ SubmissionController
    """
    def __init__(self,
            data_dict,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dict = data_dict

    def get_extra_unique_keys(self):
        """ Return a tuple of the keys of the unique extras that
            will be used to uniquely identify your workchains
        """
        return ('a_chemsys', 'batch_size', 'num_batches')

    def get_all_extras_to_submit(self):
        """ Return a *set* of the values of all extras uniquely
            identifying all simulations that you want to submit.
            Each entry of the set must be a tuple, in same order
            as the keys returned by get_extra_unique_keys().
            Note: for each item, pass extra values as tuples
        """
        data_dict = self.data_dict
        all_extras = set()
        for a_chemsys in data_dict.keys():
            batch_size = data_dict[a_chemsys][0]
            num_batches = data_dict[a_chemsys][1]
            all_extras.add((a_chemsys, batch_size, num_batches))
        return all_extras

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """ Return the inputs and the process class for the process
            to run, associated a given tuple of extras values.
            Param: extras_values: a tuple of values of the extras,
            in same order as the keys returned by get_extra_unique_keys().
        """
        inputs = {'chemical_system' : Str(extras_values[0]),
                  'batch_size' : Int(extras_values[1]),
                  'num_batches': Int(extras_values[2])
                 }
        return inputs, MatterGenBaseWorkChain
