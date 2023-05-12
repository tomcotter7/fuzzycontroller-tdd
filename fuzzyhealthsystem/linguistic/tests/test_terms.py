from fuzzyhealthsystem.linguistic.terms import LinguisticTerm
from fuzzyhealthsystem.membership.membership_functions import TriangularMF
import numpy as np


def test_load_mf():

    lt = LinguisticTerm("test", np.arange(0, 60, 0.1), {'type': "trimf",
                                                        'params': [0, 10, 20]})
    assert lt.mf == TriangularMF(np.arange(0, 60, 0.1), [0, 10, 20])
