from __future__ import annotations
from .propositions import Consequent, Antecedents
from ..linguistic.variables import LinguisticVariable
import numpy as np


class Rule():
    """A rule.

    This is of the form: IF <antecedent> THEN <consequent>, where
    <antecedent> is an :class: `.Antecedents` and <consequent> is a
    :class: `.Consequent`.

    Attributes:
        antecedents: antecedent of the rule.
        consequent: consequent of the rule.
    """

    def __init__(self, rule: dict,
                 linguistic_variables: dict[str, LinguisticVariable]):
        """Initializes a rule.

        The input rule dict should be of the form: {'antecedent': data,
        'consequent': data}, where data is a dictionary representation of the
        antecedent and consequent respectively.

        Args:
            rule: dictionary with the antecedent and consequent of the rule.
            linguistic_variables: dictionary of linguistic variables.
        """
        self.antecedents = Antecedents(rule['antecedent'],
                                       linguistic_variables)
        self.consequent = Consequent(rule['consequent'], linguistic_variables)

    def to_string(self) -> str:
        """Returns a string representation of the rule."""
        ants = self.antecedents.to_string()
        cons = self.consequent.to_string()
        return f"IF {ants} THEN {cons}"

    def apply_rule(self, firing_strengths: dict[str, dict[str, float]]) \
            -> np.ndarray:
        """Combines the firing strengths of the antecedents with
        respect to the rule. This output set is then computed.

        Args:
            firing_strengths: dictionary of firing strengths of the
                antecedents linguistic terms. The keys are the names of the
                linguistic variables and the values are dictionaries with the
                names of the linguistic terms as keys and the firing strengths
                as values.

        Returns:
            output_set: output set of the consequent.
        """

        cylindrical_extension = self.antecedents \
            .get_cylindrical_extension(firing_strengths)

        return self.consequent.compute_output_set(cylindrical_extension)
