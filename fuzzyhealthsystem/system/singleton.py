from fuzzyhealthsystem.utils.json_handler import JsonHandler
from fuzzyhealthsystem.linguistic.variables import LinguisticVariable


class SingletonFIS:

    def __init__(self):
        self.json_handler = JsonHandler()
        self.input_variables = {}

    def _load_linguistic_variable(self, name: str, variable_data: dict):
        self.input_variables[name] = LinguisticVariable(name, variable_data)

    def load_data(self, input_file: str):
        data = self.json_handler.read(input_file)
        inputs = data["inputs"]
        for key, data in inputs.items():
            self._load_linguistic_variable(key, data)
