from __future__ import annotations
from fuzzycontroller.linguistic.variables import LinguisticVariable
from fuzzycontroller.linguistic.terms import LinguisticTerm


class Antecedent:
    def __init__(self, ant: str, lvs: dict[str, LinguisticVariable]):
        """
        Args:
            ant: string representation of the antecedent.
            lvs: dictionary of linguistic variables.
        """
        self.name, self.term = self.parse_antecedent(ant, lvs)

    def parse_antecedent(self, ant: str, lvs: dict[str, LinguisticVariable]) \
            -> tuple[str, LinguisticTerm]:
        """
        Args:
            ant: string representation of the antecedent.
            lvs: dict of linguistic variables.

        Returns:
            name: name of the linguistic variable.
            term: linguistic term.
        """
        name, term = ant.split(" IS ")
        try:
            return name, lvs[name].get_term(term)
        except KeyError:
            raise ValueError(f"Antecedent {ant} not found in {lvs}")

    def to_string(self) -> str:
        """
        e.g. "temperature IS cold"

        Returns:
            string representation of the antecedent.
        """
        return f"{self.name} IS {self.term.name}"


class Antecedents():

    def __init__(self, antecedents: dict,
                 lvs: dict[str, LinguisticVariable]):
        self.antecedents = self._load_antecedents(lvs, antecedents)

    def _load_antecedents(self, lvs: dict[str, LinguisticVariable],
                          antecedents: dict[str, str or dict]) -> dict:
        """
        Load the antecedents from a dictionary.
        Should be of the form {'antecedent1': str or dict,
        'operator': 'AND', 'antecedent2': str or dict}

        Args:
            antecedents: dictionary containing the antecedents info.
        """
        if 'antecedent2' not in antecedents:
            return {'antecedent1': Antecedent(antecedents['antecedent1'], lvs)}

        ant1 = antecedents['antecedent1']
        ant2 = antecedents['antecedent2']
        operator = antecedents['operator']

        if isinstance(ant1, dict):
            ant1 = Antecedents(ant1, lvs)
        else:
            ant1 = Antecedent(ant1, lvs)

        if isinstance(ant2, dict):
            ant2 = Antecedents(ant2, lvs)
        else:
            ant2 = Antecedent(ant2, lvs)

        return {'antecedent1': ant1, 'operator': operator, 'antecedent2': ant2}

    def to_string(self):
        """
        e.g. "temperature IS cold AND humidity IS high"

        Returns:
            string representation of the antecedents.
        """

        return "(" + "".join([f" {self.antecedents[key]} " if key == 'operator'
                             else self.antecedents[key].to_string()
                              for key in self.antecedents.keys()]) + ")"
