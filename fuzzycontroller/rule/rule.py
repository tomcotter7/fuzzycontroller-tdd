from __future__ import annotations
from fuzzycontroller.rule.propositions import Consequent, Antecedents
from fuzzycontroller.linguistic.variables import LinguisticVariable
import numpy as np


class Rule():

    def __init__(self, rule: dict,
                 linguistic_variables: dict[str, LinguisticVariable]):
        self.antecedents = Antecedents(rule['antecedent'],
                                       linguistic_variables)
        self.consequent = Consequent(rule['consequent'], linguistic_variables)

    def to_string(self) -> str:
        ants = self.antecedents.to_string()
        cons = self.consequent.to_string()
        return f"IF {ants} THEN {cons}"

    def apply_rule(self, firing_strengths: dict[str, dict[str, float]]) \
            -> np.ndarray:

        cylindrical_extension = self.antecedents \
                .get_cylindrical_extension(firing_strengths)

        return self.consequent.compute_output_set(cylindrical_extension)
