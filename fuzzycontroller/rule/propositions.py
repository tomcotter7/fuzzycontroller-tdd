from __future__ import annotations
from fuzzycontroller.linguistic.variables import LinguisticVariable
from fuzzycontroller.linguistic.terms import LinguisticTerm
from abc import ABC, abstractmethod
import numpy as np


class Proposition(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the linguistic variable."""
        pass

    @property
    @abstractmethod
    def term(self) -> LinguisticTerm:
        """Return the linguistic term of the proposition."""
        pass

    def parse_proposition(self, prop: str, lvs: dict[str, LinguisticVariable])\
            -> tuple[str, LinguisticTerm]:
        """Parse a proposition string into a linguistic variable and term.

        Args:
            prop: string representation of the proposition.
            lvs: dict of linguistic variables.

        Returns:
            name: name of the linguistic variable.
            term: linguistic term.
        """
        name, term = prop.split(' IS ')
        try:
            return name, lvs[name].get_term(term)
        except KeyError:
            raise ValueError("Linguistic Variable not found")

    def to_string(self) -> str:
        """Return a string representation of the proposition."""
        return f"{self.name} IS {self.term.name}"


class Consequent(Proposition):

    def __init__(self, conq: str, lvs: dict[str, LinguisticVariable]):
        """
        Args:
            conq: string representation of the consequent.
            lvs: dict of linguistic variables.
        """
        self._name, self._term = super().parse_proposition(conq, lvs)

    @property
    def name(self) -> str:
        """Return the name of the linguistic variable."""
        return self._name

    @property
    def term(self) -> LinguisticTerm:
        """Return the linguistic term of the proposition."""
        return self._term


class Antecedent(Proposition):

    def __init__(self, ant: str, lvs: dict[str, LinguisticVariable]):
        """
        Args:
            ant: string representation of the antecedent.
            lvs: dictionary of linguistic variables.
        """
        # self._name, self._term = self.parse_antecedent(ant, lvs)
        self._name, self._term = super().parse_proposition(ant, lvs)

    @property
    def name(self) -> str:
        """Return the name of the linguistic variable."""
        return self._name

    @property
    def term(self) -> LinguisticTerm:
        """Return the linguistic term of the proposition."""
        return self._term

    def get_cylindrical_extension(self, firing_strength: float):
        return firing_strength


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

    def get_cylindrical_extension(self, firing_strengths: dict[str, float]):
        pass

