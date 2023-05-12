from fuzzyhealthsystem.linguistic.terms import LinguisticTerm
import numpy as np


class LinguisticVariable:

    def __init__(self, name, data):
        self.name = name
        self.universe = self._load_universe(data['universe'])
        self.terms = self._load_terms(data['terms'])

    def _load_universe(self, universe):
        return np.arange(universe['start'], universe['end'], universe['step'])

    def _load_terms(self, terms):
        return [LinguisticTerm(item[0], self.universe, item[1])
                for item in terms.items()]
