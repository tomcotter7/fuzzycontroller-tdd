from __future__ import annotations
from ..utils.json_handler import JsonHandler
from ..linguistic.variables import LinguisticVariable
from ..rule.rules import Rules
import numpy as np
import skfuzzy as fuzz


class SingletonFIS:

    def __init__(self) -> None:
        self.json_handler = JsonHandler()
        self.variables = {}
        self.output_variable = None

    def _load_linguistic_variable(self, name: str,
                                  variable_data: dict) -> None:
        """
        Load the linguistic variable from a dictionary.
        Should be of the form {'universe': universe, 'terms': terms}

        Args:
            name: name of the linguistic variable.
            variable_data: dictionary containing the linguistic variable info.
        """
        self.variables[name] = LinguisticVariable(name, variable_data)

    def load_data(self, input_file: str):
        json_data = self.json_handler.read(input_file)
        inputs = json_data["inputs"]
        for key, data in inputs.items():
            self._load_linguistic_variable(key, data)
        for key, data in json_data["output"].items():
            self.output_variable = key
            self.variables[key] = LinguisticVariable(key, data)

        self.rg = Rules(json_data["rules"], self.variables)

    def get_all_firing_strengths(self, crisp_inputs: dict) \
            -> dict[str, dict[str, float]]:
        """
        Compute the firing strength of all the linguistic terms from inputs.
        Inputs should be of the form {'input1': 1.0, 'input2': 2.0}
        where the keys are the names of the linguistic variables and the values
        are the crisp inputs.

        Args:
            crisp_inputs: dictionary containing the crisp inputs.

        Returns:
            dictionary containing the firing strengths of all the linguistic
            terms. Will be of the form {'input1': {('term1', 0.5), ...},
            'input2': ...}

        """
        return {key: self.variables[key].compute_memberships(
                crisp_inputs[key], "singleton")
                for key in crisp_inputs.keys()}

    def compute_output_sets(self, crisp_inputs: dict) -> dict:
        fs = self.get_all_firing_strengths(crisp_inputs)
        self.rg.get_correct_output_sets(fs)
        return self.rg.output_sets

    def compute_aggregate_set(self, crisp_inputs):
        output_sets = self.compute_output_sets(crisp_inputs)
        return np.fmax.reduce(list(output_sets.values()))

    def compute_defuzzified_output(self, crisp_inputs,
                                   defuzzication_method="centroid"):
        aggregate_set = self.compute_aggregate_set(crisp_inputs)
        return fuzz.defuzz(self.variables[self.output_variable].universe,
                           aggregate_set, defuzzication_method)
