from fuzzycontroller.rule.propositions import Antecedent, Consequent, \
        Antecedents
from fuzzycontroller.linguistic.variables import LinguisticVariable


def test_load_consequent():

    lv = LinguisticVariable("layers",
                            {"universe": {"start": 0, "end": 5, "step": 1},
                             "terms": {
                                 "many": {
                                    "name": "many",
                                    "mf": {"type": "trimf",
                                           "params": [3, 5, 5]}
                                     }
                                 }
                             })

    conq = Consequent("layers IS many", {'layers': lv})
    assert conq.to_string() == "layers IS many"
    assert conq.term == lv.terms['many']


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
    assert ant.term == lv.terms['cold']


def test_get_cylindrical_extension_antecedent():
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
    assert ant.get_cylindrical_extension(0.5) == 0.5


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


def test_get_cylindrical_extension_antecedent_with_or():
    lv = LinguisticVariable("temperature",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {
                                 "cold": {
                                    "name": "cold",
                                    "mf": {"type": "trimf",
                                           "params": [0, 10, 20]}
                                     }
                                 },
                                "very_cold": {
                                     "name": "very_cold",
                                     "mf": {"type": "trimf",
                                            "params": [0, 5, 10]}
                                    }
                             })

    antecedents = Antecedents({"antecedent1": "temperature IS cold",
                               "operator": "OR",
                               "antecedent2": "temperature IS very_cold"},
                              {'temperature': lv})
    assert antecedents.get_cylindrical_extension({"cold": 0.5, "very_cold": 0.75}) == 0.75
