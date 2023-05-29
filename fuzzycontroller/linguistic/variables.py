from __future__ import annotations
from ..linguistic.terms import LinguisticTerm
import numpy as np


class LinguisticVariable:

    def __init__(self, name, data):
        self.name = name
        self.universe = self._load_universe(data['universe'])
        self.terms = self._load_terms(data['terms'])

    def _load_universe(self, universe: dict):
        """
        Load the universe from a dictionary.
        Should be of the form {'start': 0, 'end': 10, 'step': 0.1}

        Args:
            universe: dictionary containing the universe info.
        """
        return np.arange(float(universe['start']),
                         float(universe['end']),
                         float(universe['step']))

    def _load_terms(self, terms) -> dict[str, LinguisticTerm]:
        """
        Load the linguistic terms from a dictionary.
        Should be of the form {'term1': {'name': 'term1', 'mf': mf1},
        'term2': {'name': 'term2', 'mf': mf2}}

        Args:
            terms: dict containing the linguistic terms info.
        """

        loaded_terms = {}
        for item in terms.items():
            lt = LinguisticTerm(self.universe, item[1])
            loaded_terms[lt.name] = lt

        return loaded_terms

    def compute_all_firing_strengths(self, crisp_input: float or np.ndarray,
                                     input_type: str) \
            -> dict[str, float or np.ndarray]:
        """
        Compute the firing strength of all the linguistic terms.

        Args:
            crisp_input: crisp input value.
            input_type: type of the input, either 'singleton' or
                'non-singleton'
        """

        firing_strengths = {}
        for item in self.terms.items():
            firing_strengths[item[0]] = item[1].compute_membership(
                crisp_input, input_type)

        return firing_strengths

    def get_term(self, search_term: str) -> LinguisticTerm:
        """
        Get a linguistic term by name.

        Args:
            search_term: name of the linguistic term.
        """
        return self.terms[search_term]
