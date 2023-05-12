from fuzzycontroller.linguistic.terms import LinguisticTerm
import numpy as np


class LinguisticVariable:

    def __init__(self, name, data):
        self.name = name
        self.universe = self._load_universe(data['universe'])
        self.terms = self._load_terms(data['terms'])

    def _load_universe(self, universe):
        return np.arange(float(universe['start']),
                         float(universe['end']),
                         float(universe['step']))

    def _load_terms(self, terms):
        return [LinguisticTerm(self.universe, item[1])
                for item in terms.items()]

    def compute_all_firing_strengths(self, crisp_input: float or np.ndarray,
                                     input_type: str):
        return [(term.name, term.compute_membership(crisp_input, input_type))
                for term in self.terms]
