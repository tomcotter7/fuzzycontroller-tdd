from fuzzyhealthsystem.membership.membership_functions import TriangularMF, \
        TrapezoidalMF, GauAngleMF
import numpy as np


class LinguisticTerm:

    def __init__(self, name, universe, mf):
        self.name = name
        self.universe = universe
        self.mf = self._load_mf(mf)

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
        elif mf['type'] == "gauanglemf":
            return GauAngleMF(self.universe, mf['params'],
                              mf['start'], mf['end'])
