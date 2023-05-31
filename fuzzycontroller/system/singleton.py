from __future__ import annotations

from numpy.typing import NDArray
from ..utils.json_handler import JsonHandler
from ..linguistic.variables import LinguisticVariable
from ..rule.rules import Rules
import numpy as np
import skfuzzy as fuzz


class SingletonFIS:
    """A Singleton Inference System.

    A singleton inference system calculates firing strengths / degrees of
    membership using a singleton fuzzy set / a spike for each crisp input.

    Attributes:
        variables: dictionary of linguistic variables.
        output_variable: name of the output variable.
        rg: rules of the system.
        json_handler: :class: `.JsonHandler` object.

    """

    def __init__(self) -> None:
        """Initializes the singleton inference system."""
        self.json_handler = JsonHandler()
        self.variables = {}
        self.output_variable = None

    def _load_linguistic_variable(self, name: str,
                                  variable_data: dict) -> None:
        """Loads the linguistic variable from a dictionary.
        variable_data should be of the form {'universe': universe,
        'terms': terms}

        Args:
            name: name of the linguistic variable.
            variable_data: dictionary containing the linguistic variable info.
        """
        self.variables[name] = LinguisticVariable(name, variable_data)

    def load_data(self, input_file: str):
        """Loads the data from a json file

        Input file should correspond to 'example.json' in the
        root of the repository.

        Args:
            input_file: path to the json file containing the data.

        """
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
        """Computes the firing strength of all the linguistic terms
        from inputs. Inputs should be of the form {'input1': 1.0,
        'input2': 2.0} where the keys are the names of the linguistic
        variables and the values are the crisp inputs.

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

    def compute_output_sets(self, crisp_inputs: dict) -> dict[str, NDArray]:
        """Computes the output sets for the given crisp inputs.

        There will be one output set for each rule.

        Args:
            crisp_inputs: dictionary containing the crisp inputs.

        Returns:
            dictionary containing the output sets for each rule.

        """
        fs = self.get_all_firing_strengths(crisp_inputs)
        self.rg.get_correct_output_sets(fs)
        return self.rg.output_sets

    def compute_aggregate_set(self, crisp_inputs) -> NDArray:
        """Computes the aggregate set for the given crisp inputs.

        Args:
            crisp_inputs: dictionary containing the crisp inputs.

        Returns:
            aggregate set for the given crisp inputs.
        """
        output_sets = self.compute_output_sets(crisp_inputs)
        return np.fmax.reduce(list(output_sets.values()))

    def compute_defuzzified_output(self, crisp_inputs,
                                   defuzzication_method="centroid"):
        """Computes the defuzzified output for the given crisp inputs.

        Args:
            crisp_inputs: dictionary containing the crisp inputs.
            defuzzication_method: method used to defuzzify the output
                default: centroid

        Returns:
            defuzzified output for the given crisp inputs.

        """
        aggregate_set = self.compute_aggregate_set(crisp_inputs)
        return fuzz.defuzz(self.variables[self.output_variable].universe,
                           aggregate_set, defuzzication_method)
