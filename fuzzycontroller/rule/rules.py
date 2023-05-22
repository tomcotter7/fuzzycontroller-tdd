from __future__ import annotations
from fuzzycontroller.rule.rule import Rule
from fuzzycontroller.linguistic.variables import LinguisticVariable


class Rules():

    def __init__(self, rules: dict,
                 linguistic_variables: dict[str, LinguisticVariable]):
        self.rules = {}
        for rule_name, rule in rules.items():
            self.rules[rule_name] = Rule(rule, linguistic_variables)
