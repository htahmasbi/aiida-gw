from aiida.orm import Str
from aiida_submission_controller import FromGroupSubmissionController
from aiida_datagen.codes.cp2k.cp2k_workchains import (
    DimerGeOptWorkChain,
    Scheme1GeOptWorkChain,
    Scheme2GeOptWorkChain,
    Scheme3GeOptWorkChain,
    ClusterGeOptWorkChain,
    MoleculeGeOptWorkChain,
    SinglePointtWorkChain)

class CP2KSubmissionController(FromGroupSubmissionController):
    """ A SubmissionController
    """
    def __init__(self,
              QSorSIRIUS,
              *args,
              **kwargs):
        super().__init__(*args, **kwargs)
        self.QSorSIRIUS = QSorSIRIUS

    def get_extra_unique_keys(self):
        """ Return a tuple of the keys of the unique extras that
            will be used to uniquely identify your workchains
        """
        return ('job', )

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """ Return the inputs and the process class for the process to run,
            associated a given tuple of extras values.
            Param: extras_values: a tuple of values of the extras,
            in same order as the keys returned by get_extra_unique_keys().
        """
        parent_node = self.get_parent_node_from_extras(extras_values)
        structure = parent_node
        inputs = {'structure': structure, 'QSorSIRIUS': Str(self.QSorSIRIUS)}
        process_class = None
        if 'reference' in parent_node.label:
            process_class = RefGeOptWorkChain
        if 'dimer' in parent_node.label:
            process_class = DimerGeOptWorkChain
        if 'scheme1' in parent_node.label:
            process_class = Scheme1GeOptWorkChain
        if 'scheme2' in parent_node.label:
            process_class = Scheme2GeOptWorkChain
        if 'scheme3' in parent_node.label:
            process_class = Scheme3GeOptWorkChain
        if 'cluster' in parent_node.label:
            process_class = ClusterGeOptWorkChain
        if 'molecule' in parent_node.label:
            process_class = MoleculeGeOptWorkChain
        return inputs, process_class

class CP2KSPSubmissionController(FromGroupSubmissionController):
    """ A SubmissionController
    """
    def __init__(self,
              QSorSIRIUS,
              *args,
              **kwargs):
        super().__init__(*args, **kwargs)
        self._process_class = SinglePointtWorkChain
        self.QSorSIRIUS = QSorSIRIUS

    def get_extra_unique_keys(self):
        """ Return a tuple of the keys of the unique extras that
            will be used to uniquely identify your workchains
        """
        return ('job', )

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """ Return the inputs and the process class for the process to run,
            associated a given tuple of extras values.
            Param: extras_values: a tuple of values of the extras,
            in same order as the keys returned by get_extra_unique_keys().
        """
        parent_node = self.get_parent_node_from_extras(extras_values)
        structure = parent_node
        if 'cluster' in parent_node.label:
            bc = 'free'
        else:
            bc = 'bulk'
        inputs = {'structure': structure, 'bc': Str(bc), 'QSorSIRIUS': Str(self.QSorSIRIUS)}
        return inputs, self._process_class
