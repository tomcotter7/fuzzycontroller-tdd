from ..terms import LinguisticTerm
from ...membership.membership_functions import TriangularMF
import numpy as np


def test_load_mf():

    lt = LinguisticTerm(np.arange(0, 60, 0.1), {"name": "cold",
                                                "mf": {"type": "trimf",
                                                       "params": [0, 10, 20]}})
    assert lt.mf == TriangularMF(np.arange(0, 60, 0.1), [0, 10, 20])
