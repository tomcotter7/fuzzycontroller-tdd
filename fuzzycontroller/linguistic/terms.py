from fuzzycontroller.membership.membership_functions import TriangularMF, \
        TrapezoidalMF, GauAngleMF
import numpy as np


class LinguisticTerm:

    def __init__(self, universe, term):
        self.name = term['name']
        self.universe = universe
        self.mf = self._load_mf(term['mf'])

    def __eq__(self, other):
        if isinstance(other, LinguisticTerm):
            return self.name == other.name and \
                    np.array_equal(self.universe, other.universe) and \
                    self.mf == other.mf
        return False

    def _load_mf(self, mf):
        if mf['type'] == "trimf":
            return TriangularMF(self.universe, mf['params'])
        elif mf['type'] == "trapmf":
            return TrapezoidalMF(self.universe, mf['params'])
        return GauAngleMF(self.universe, mf['params'],
                          float(mf['start']), float(mf['end']))

    def compute_membership(self, crisp_input: float or np.ndarray,
                           input_type: str):
        if input_type == "singleton":
            return self.mf.singleton_interp_mem(crisp_input)
