from fuzzyhealthsystem.linguistic.variables import LinguisticVariable
from fuzzyhealthsystem.linguistic.terms import LinguisticTerm
import numpy as np


def test_load_name():

    lv = LinguisticVariable("temp",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {"cold": {"type": "trimf",
                                                "params": [0, 10, 20]}}})
    assert lv.name == "temp"


def test_load_universe():
    lv = LinguisticVariable("temp",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {"cold": {"type": "trimf",
                                                "params": [0, 10, 20]}}})
    assert np.array_equal(lv.universe, np.arange(0, 60, 0.1))


def test_load_terms():
    lv = LinguisticVariable("temp",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {"cold": {"type": "trimf",
                                                "params": [0, 10, 20]}}})
    assert lv.terms == [LinguisticTerm("cold", np.arange(0, 60, 0.1),
                                       {'type': "trimf",
                                        'params': [0, 10, 20]})]


def test_load_terms_gauangle():
    lv = LinguisticVariable("temp",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {"cold": {"type": "gauanglemf",
                                                "params": [0, 10, 20],
                                                "start": 0,
                                                "end": 60}}})
    assert lv.terms == [LinguisticTerm("cold", np.arange(0, 60, 0.1),
                                       {'type': "gauanglemf",
                                        'params': [0, 10, 20],
                                        'start': 0,
                                        'end': 60})]
