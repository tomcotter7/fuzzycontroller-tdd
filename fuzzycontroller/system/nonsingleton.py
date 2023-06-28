from __future__ import annotations
from .fis import FIS
from ..membership.membership_functions import GauAngleMF


class NonSingletonFIS(FIS):
    """A NonSingleton Fuzzy Inference System

    A non-singleton inferences system calculates the firing strengths /
    degrees of membership using non-singleton fuzzy sets. Essentially the
    inputs are type-1 fuzzy sets. Inherits from :class:`FIS`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._type = "non-singleton"

    @property
    def type(self) -> str:
        """Returns the type of the FIS"""
        return self._type

    def create_input_sets(self, inputs: dict[str, dict[str, float]]) -> dict:
        input_sets = {}
        for var_name, var in inputs.items():
            universe = self.variables[var_name].universe
            mean = (inputs[var_name]["start"] + inputs[var_name]["end"]) / 2
            sd = (inputs[var_name]["end"] - inputs[var_name]["start"]) / 4
            input_sets[var_name] = GauAngleMF(universe, [mean, sd],
                                              var['start'], var['end'])

        return input_sets

    def get_all_firing_strengths(self, inputs: dict[str, dict[str, float]]) \
            -> dict[str, dict[str, float]]:

        input_sets = self.create_input_sets(inputs)

        return {key: self.variables[key].compute_memberships(
                input_sets[key], self.type) for key in inputs.keys()}
