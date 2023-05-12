from fuzzycontroller.linguistic.variables import LinguisticVariable
from fuzzycontroller.linguistic.terms import LinguisticTerm
import numpy as np


def test_load_name():

    lv = LinguisticVariable("temp",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {
                                 "cold": {
                                    "name": "cold",
                                    "mf": {"type": "trimf",
                                           "params": [0, 10, 20]}
                                     }
                                 }
                             })

    assert lv.name == "temp"


def test_load_universe():

    lv = LinguisticVariable("temp",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {
                                 "cold": {
                                    "name": "cold",
                                    "mf": {"type": "trimf",
                                           "params": [0, 10, 20]}
                                     }
                                 }
                             })

    assert np.array_equal(lv.universe, np.arange(0, 60, 0.1))


def test_load_terms():
    lv = LinguisticVariable("temp",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {
                                 "cold": {
                                    "name": "cold",
                                    "mf": {"type": "trimf",
                                           "params": [0, 10, 20]}
                                     }
                                }
                             })

    assert lv.terms == [LinguisticTerm(np.arange(0, 60, 0.1),
                                       {"name": "cold",
                                        "mf": {
                                            "type": "trimf",
                                            "params": [0, 10, 20]}})]


def test_load_terms_gauangle():
    lv = LinguisticVariable("temp",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {
                                 "cold": {
                                     "name": "cold",
                                     "mf": {
                                        "type": "gauanglemf",
                                        "params": [0, 10, 20],
                                        "start": 0,
                                        "end": 60}}}})

    assert lv.terms == [LinguisticTerm(np.arange(0, 60, 0.1),
                                       {"name": "cold",
                                        "mf": {
                                            "type": "gauanglemf",
                                            "params": [0, 10, 20],
                                            "start": 0,
                                            "end": 60}})]


def test_get_all_firing_strengths_singleton():
    lv = LinguisticVariable("temp",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {"cold": {
                                 "name": "cold",
                                 "mf": {"type": "trimf", "params": [0, 10, 20]}
                                 }
                                       }
                             })
    fs = lv.compute_all_firing_strengths(10, "singleton")
    assert fs == [("cold", 1.0)]
