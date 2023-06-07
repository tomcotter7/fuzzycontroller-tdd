from ..terms import LinguisticTerm
from ...membership.membership_functions import TriangularMF
import numpy as np


def test_load_mf():

    lt = LinguisticTerm(np.arange(0, 60, 0.1), {"name": "cold",
                                                "mf": {"type": "trimf",
                                                       "params": [0, 10, 20]}})
    assert lt.mf == TriangularMF(np.arange(0, 60, 0.1), [0, 10, 20])


def test_compute_membership_singleton():
    lt = LinguisticTerm(np.arange(0, 11, 1), {"name": "cold",
                                              "mf": {"type": "trimf",
                                                     "params": [0, 5, 10]}})
    assert lt.compute_membership(5.0, "singleton") == 1.0


def test_compute_membership_nonsingleton():
    expected = 0.18644067796610167
    lt = LinguisticTerm(np.arange(0, 11, 1), {"name": "cold",
                                              "mf": {"type": "trimf",
                                                     "params": [0, 5, 10]}})

    input_mf = TriangularMF(np.arange(0, 11, 1), [0, 2, 4])
    actual = lt.compute_membership(input_mf, "non-singleton")
    assert expected == actual
