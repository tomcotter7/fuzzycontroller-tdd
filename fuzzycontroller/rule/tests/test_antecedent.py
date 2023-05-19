from fuzzycontroller.rule.antecedent import Antecedent, Antecedents
from fuzzycontroller.linguistic.variables import LinguisticVariable
import numpy as np


def test_load_antecedent():

    lv = LinguisticVariable("temperature",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {
                                 "cold": {
                                    "name": "cold",
                                    "mf": {"type": "trimf",
                                           "params": [0, 10, 20]}
                                     }
                                 }
                             })

    ant = Antecedent("temperature IS cold", {'temperature': lv})
    assert ant.to_string() == "temperature IS cold"


def test_load_antecedents_with_OR():

    lv = LinguisticVariable("temperature",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {
                                 "cold": {
                                    "name": "cold",
                                    "mf": {"type": "trimf",
                                           "params": [0, 10, 20]}
                                     }
                                 }
                             })

    antecedents = Antecedents({"antecedent1": "temperature IS cold",
                               "operator": "OR",
                               "antecedent2": "temperature IS cold"},
                              {'temperature': lv})
    assert antecedents.to_string() == \
        "(temperature IS cold OR temperature IS cold)"


def test_load_antecedents_with_AND():

    lv = LinguisticVariable("temperature",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {
                                 "cold": {
                                    "name": "cold",
                                    "mf": {"type": "trimf",
                                           "params": [0, 10, 20]}
                                     }
                                 }
                             })

    antecedents = Antecedents({"antecedent1": "temperature IS cold",
                               "operator": "AND",
                               "antecedent2": "temperature IS cold"},
                              {'temperature': lv})

    assert antecedents.to_string() == \
        "(temperature IS cold AND temperature IS cold)"


def test_load_antecedents_with_sub_antecedent():
    lv = LinguisticVariable("temperature",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {
                                 "cold": {
                                    "name": "cold",
                                    "mf": {"type": "trimf",
                                           "params": [0, 10, 20]}
                                     }
                                 }
                             })

    antecedents = Antecedents({"antecedent1": "temperature IS cold",
                               "operator": "AND",
                               "antecedent2": {
                                   "antecedent1": "temperature IS cold",
                                   "operator": "OR",
                                   "antecedent2": "temperature IS cold"}},
                              {'temperature': lv})
    assert antecedents.to_string() == \
        "(temperature IS cold AND \
(temperature IS cold OR temperature IS cold))"


def test_load_antecedents_with_one_antecedent():
    lv = LinguisticVariable("temperature",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {
                                 "cold": {
                                    "name": "cold",
                                    "mf": {"type": "trimf",
                                           "params": [0, 10, 20]}
                                     }
                                 }
                             })

    antecedents = Antecedents({"antecedent1": "temperature IS cold"},
                              {"temperature": lv})
    assert antecedents.to_string() == \
        "(temperature IS cold)"
