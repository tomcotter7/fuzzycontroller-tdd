from __future__ import annotations

from numpy._typing import NDArray
from ..linguistic.terms import LinguisticTerm
import numpy as np


class LinguisticVariable:
    """A linguistic variable

    An example of a linguistic variable is 'temperature'. In the literature
    this is defined as a triplet -> name, set of terms, universe. For example:
    (temperature, {hot, cold}, [0-100])

    Attributes:
        name: A string representation of the name.
        universe: A numpy array representation of the universe
            of discourse
        terms: A dictionary to store the associated LinguisticTerms.
            Of the form {'hot': LinguisticTerm}
    """

    def __init__(self, name: str, data: dict):
        """Initializes the LinguisticVariable based on a name
        and a dictionary

        Args:
            name: the name of the variable
            data: a dictionary defining the linguistic variable
                should be of the form {'universe': dict, 'terms':
                dict}
        """
        self.name = name
        self.universe = self._load_universe(data['universe'])
        self.terms = self._load_terms(data['terms'])

    def _load_universe(self, universe: dict) -> NDArray:
        """Loads the universe from a dictionary.
        Should be of the form {'start': 0, 'end': 10, 'step': 0.1}

        Args:
            universe: dictionary containing the universe info.

        Returns:
            A numpy array representation of the universe.
        """
        return np.arange(float(universe['start']),
                         float(universe['end']),
                         float(universe['step']))

    def _load_terms(self, terms) -> dict[str, LinguisticTerm]:
        """Loads the linguistic terms from a dictionary.
        Should be of the form {'term1': {'name': 'term1', 'mf': mf1},
        'term2': {'name': 'term2', 'mf': mf2}}

        Args:
            terms: dict containing the linguistic terms info.

        Returns:
            A new dictionary representing the linguistic terms.
                Of the form {'name': LinguisticTerm}
        """

        loaded_terms = {}
        for item in terms.items():
            lt = LinguisticTerm(self.universe, item[1])
            loaded_terms[lt.name] = lt

        return loaded_terms

    def compute_memberships(self, crisp_input: float or np.ndarray,
                            input_type: str) -> dict[str, float or np.ndarray]:
        """Computes the degree of membership of all the linguistic terms.

        Args:
            crisp_input: crisp input value.
            input_type: type of the input, either 'singleton' or
                'non-singleton'

        Returns:
            A dictionary of the memberships of each term for the input
                crisp value.
        """

        memberships = {}
        for item in self.terms.items():
            memberships[item[0]] = item[1].compute_membership(
                crisp_input, input_type)

        return memberships

    def get_term(self, search_term: str) -> LinguisticTerm:
        """Returns a linguistic term by name

        Args:
            search_term: name of the linguistic term.
        """
        return self.terms[search_term]
