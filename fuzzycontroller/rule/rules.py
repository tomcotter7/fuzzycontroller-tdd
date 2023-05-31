from __future__ import annotations
from .rule import Rule
from ..linguistic.variables import LinguisticVariable
import numpy as np


class Rules():
    """Multiple rules for the whole system

    Defined as multiple :class: `.Rule` objects.

    Attributes:
        output_sets: dictionary of output sets of the rules. The keys are the
            name of the rule.
        lvs: dictionary of linguistic variables.
        rules: dictionary of rules. The keys are the name of the rule.
    """

    def __init__(self, rules: dict,
                 linguistic_variables: dict[str, LinguisticVariable]):
        """Initializes the rules.

        Args:
            rules: dict of rules. These are in raw form and need to be parsed.
            linguistic_variables: dict of linguistic variables.
        """
        self._output_sets = None
        self.lvs = linguistic_variables
        self.rules = {}
        for rule_name, rule in rules.items():
            self.rules[rule_name] = Rule(rule, linguistic_variables)

    @property
    def output_sets(self) -> dict[str, np.ndarray]:
        """The output sets of the rules.

        Raises:
            AttributeError if the output sets have not been computed.
        """
        if self._output_sets is not None:
            return self._output_sets
        raise AttributeError("Output sets not computed")

    def get_correct_output_sets(self,
                                firing_strengths: dict[str, dict[str, float]]):
        """Calculates the correct output sets for the given firing strengths.

        Args:
            firing_strengths: dict of firing strengths for each
                linguistic term.
        """
        output_sets = {}
        for rule_name, rule in self.rules.items():
            output_sets[rule_name] = rule.apply_rule(firing_strengths)

        self._output_sets = output_sets
