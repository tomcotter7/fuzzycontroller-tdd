from __future__ import annotations
from abc import ABC, abstractmethod
from ..utils.json_handler import JsonHandler
from ..rule.rules import Rules
from ..linguistic.variables import LinguisticVariable
from numpy.typing import NDArray
import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt


class FIS(ABC):
    """A Abstract Mamdani Fuzzy Inference System

    Attributes:
        json_handler: JsonHandler object used to read the json file.
        variables: dictionary containing the linguistic variables.
        output_variable: name of the output variable.
        rules: Rules object containing the rules.
        type: type of the inference system - used in implementation.
    """

    @property
    def type(self):
        pass

    def __init__(self):
        """Initializes the Mamdani Fuzzy Inference System."""
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

        self.rules = Rules(json_data["rules"], self.variables)

    @abstractmethod
    def get_all_firing_strengths(self, inputs) -> dict[str, dict[str, float]]:
        pass

    def compute_output_sets(self, crisp_inputs: dict) -> dict[str, NDArray]:
        """Computes the output sets for the given crisp inputs.

        There will be one output set for each rule.

        Args:
            crisp_inputs: dictionary containing the crisp inputs.

        Returns:
            dictionary containing the output sets for each rule.

        """
        fs = self.get_all_firing_strengths(crisp_inputs)
        self.rules.get_correct_output_sets(fs)
        return self.rules.output_sets

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

    def graph_membership_functions(self):
        """Graphs the membership functions for all the linguistic variables."""
        fig, axs = plt.subplots(nrows=len(self.variables), figsize=(15, 5))
        for i, (key, variable) in enumerate(self.variables.items()):
            variable.graph(axs[i])
            axs[i].set_title(key)
            axs[i].legend()

        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()
        plt.show()
