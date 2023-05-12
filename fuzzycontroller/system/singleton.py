from fuzzycontroller.utils.json_handler import JsonHandler
from fuzzycontroller.linguistic.variables import LinguisticVariable


class SingletonFIS:

    def __init__(self):
        self.json_handler = JsonHandler()
        self.input_variables = {}

    def _load_input_linguistic_variable(self, name: str, variable_data: dict):
        self.input_variables[name] = LinguisticVariable(name, variable_data)

    def load_data(self, input_file: str):
        json_data = self.json_handler.read(input_file)
        inputs = json_data["inputs"]
        for key, data in inputs.items():
            self._load_input_linguistic_variable(key, data)
        for key, data in json_data["output"].items():
            self.output_variable = LinguisticVariable(key, data)

    def get_all_firing_strengths(self, crisp_inputs: dict):
        return {key: self.input_variables[key].compute_all_firing_strengths(
                crisp_inputs[key], "singleton")
                for key in crisp_inputs.keys()}
