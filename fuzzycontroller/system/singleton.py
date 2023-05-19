from fuzzycontroller.utils.json_handler import JsonHandler
from fuzzycontroller.linguistic.variables import LinguisticVariable


class SingletonFIS:

    def __init__(self) -> None:
        self.json_handler = JsonHandler()
        self.input_variables = {}

    def _load_input_linguistic_variable(self, name: str,
                                        variable_data: dict) -> None:
        """
        Load the linguistic variable from a dictionary.
        Should be of the form {'universe': universe, 'terms': terms}

        Args:
            name: name of the linguistic variable.
            variable_data: dictionary containing the linguistic variable info.
        """
        self.input_variables[name] = LinguisticVariable(name, variable_data)

    def load_data(self, input_file: str):
        json_data = self.json_handler.read(input_file)
        inputs = json_data["inputs"]
        for key, data in inputs.items():
            self._load_input_linguistic_variable(key, data)
        for key, data in json_data["output"].items():
            self.output_variable = LinguisticVariable(key, data)

    def get_all_firing_strengths(self, crisp_inputs: dict) -> dict:
        """
        Compute the firing strength of all the linguistic terms from inputs.
        Inputs should be of the form {'input1': 1.0, 'input2': 2.0}
        where the keys are the names of the linguistic variables and the values
        are the crisp inputs.

        Args:
            crisp_inputs: dictionary containing the crisp inputs.

        Returns:
            dictionary containing the firing strengths of all the linguistic
            terms. Will be of the form {'input1': [('term1', 0.5), ...],
            'input2': ...}

        """
        return {key: self.input_variables[key].compute_all_firing_strengths(
                crisp_inputs[key], "singleton")
                for key in crisp_inputs.keys()}
