import yaml
from aiida.parsers import Parser
from aiida.common import exceptions

class GenSymCrysParser(Parser):
    """ AiiDA parser class for the output of GenSymCrys
    """
    def parse(self, **kwargs):
        """Parse
        """
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            with retrieved_folder.open('posout.yaml', 'r') as fhandle:
                status = self._read_stdout(fhandle)
        except (OSError, IOError):
            return self.exit_codes.ERROR_OUTPUT_PARSE
        if not status:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
        return None

    @staticmethod
    def _read_stdout(stdout):
        """ Parse stdout
        """
        if len(list(yaml.safe_load_all(stdout))) == 0:
            return False
        return True
