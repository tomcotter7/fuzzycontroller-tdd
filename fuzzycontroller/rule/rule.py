from __future__ import annotations
from fuzzycontroller.rule.propositions import Consequent, Antecedents
from fuzzycontroller.linguistic.variables import LinguisticVariable


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
