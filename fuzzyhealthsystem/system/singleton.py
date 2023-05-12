from fuzzyhealthsystem.utils.json_handler import JsonHandler
from fuzzyhealthsystem.linguistic.variables import LinguisticVariable


class SingletonFIS:

    def __init__(self):
        self.json_handler = JsonHandler()
        self.input_variables = {}

    def _load_linguistic_variable(self, linguistic_variable: dict):
        name = linguistic_variable["name"]
        universe = linguistic_variable["universe"]
        self.input_variables[name] = LinguisticVariable(name, universe)

    def load_data(self, input_file: str):
        data = self.json_handler.read(input_file)
        inputs = data["inputs"]
        for inpt_var in inputs:
            self._load_linguistic_variable(inpt_var)
