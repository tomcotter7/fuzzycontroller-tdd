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
                ex. 'linguistic_variable IS term'
            lvs: dict of linguistic variables. Keys are the names of the
                linguistic variables.

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

    def compute_output_set(self, cylindrical_extension: float or np.ndarray) \
            -> np.ndarray:
        """Compute the output set of the consequent.

        Args:
            cylindrical_extension: cylindrical extension of the antecedents.

        Returns:
            output_set: output set of the consequent.
        """
        return self.term.apply_fmax_or_fmin(np.fmin, cylindrical_extension)


class Antecedent(Proposition):

    def __init__(self, ant: str, lvs: dict[str, LinguisticVariable]):
        """
        Args:
            ant: string representation of the antecedent.
            lvs: dictionary of linguistic variables.
        """
        self._negate = False
        if "NOT" in ant:
            self._negate = True
            ant = ant.replace("NOT ", "")
        self._name, self._term = super().parse_proposition(ant, lvs)

    @property
    def negate(self) -> bool:
        """Return true if we should negate this"""
        return self._negate

    @property
    def name(self) -> str:
        """Return the name of the linguistic variable."""
        return self._name

    @property
    def term(self) -> LinguisticTerm:
        """Return the linguistic term of the proposition."""
        return self._term

    def to_string(self) -> str:
        """Return a string representation of the proposition."""
        if self.negate:
            return f"NOT {super().to_string()}"
        else:
            return super().to_string()

    def get_cylindrical_extension(self, firing_strengths:
                                  dict[str, dict[str, float]]) -> float:
        """
        Get the cylindrical extension of a single antecedent.
        Also applies NOT if this is the case

        Args:
            firing_strengths: dict of firing strengths.

        Returns:
            cylindrical_extension: cylindrical extension of the antecedent.
        """
        fss = firing_strengths[self.name]
        if self.negate:
            return 1 - fss[self.term.name]
        return fss[self.term.name]


class Antecedents():

    def __init__(self, antecedents: dict,
                 lvs: dict[str, LinguisticVariable]):
        self.antecedents = self._load_antecedents(lvs, antecedents)

    @property
    def single_antecedent(self) -> bool:
        """Return true if there is only one antecedent."""
        return self._single_antecedent

    def _load_antecedents(self, lvs: dict[str, LinguisticVariable],
                          antecedents: dict[str, str or dict]) -> dict:
        """
        Load the antecedents from a dictionary.
        Should be of the form {'antecedent1': str or dict,
        'operator': 'AND', 'antecedent2': str or dict}

        Args:
            antecedents: dictionary containing the antecedents info.
        """
        self._single_antecedent = False
        if 'antecedent2' not in antecedents:
            self._single_antecedent = True
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

    def get_cylindrical_extension(self, firing_strengths:
                                  dict[str, dict[str, float]]) \
            -> float or np.ndarray:
        """
        Get the cylindrical extension of the antecedents.

        Args:
            firing_strengths: dictionary of firing strengths.
                of the form {variable_name: {term_name: firing_strength}}

        Returns:
            cylindrical_extension: cylindrical extension of the antecedents.
        """

        if self.single_antecedent:
            return self.antecedents['antecedent1'].get_cylindrical_extension(
                firing_strengths)

        if self.antecedents['operator'] == 'OR':
            return np.fmax(self.antecedents['antecedent1']
                           .get_cylindrical_extension(firing_strengths),
                           self.antecedents['antecedent2']
                           .get_cylindrical_extension(firing_strengths))

        return np.fmin(self.antecedents['antecedent1']
                       .get_cylindrical_extension(firing_strengths),
                       self.antecedents['antecedent2']
                       .get_cylindrical_extension(firing_strengths))
