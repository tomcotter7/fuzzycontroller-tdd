from fuzzycontroller.rule.propositions import Antecedent, Consequent, \
        Antecedents
from fuzzycontroller.linguistic.variables import LinguisticVariable
import pytest


@pytest.fixture
def setup():

    # SETUP

    lv = LinguisticVariable("temperature",
                            {"universe": {"start": 0, "end": 60, "step": 0.1},
                             "terms": {
                                 "cold": {
                                     "name": "cold",
                                     "mf": {"type": "trimf",
                                            "params": [0, 10, 20]}
                                     },
                                 "very_cold": {
                                     "name": "very_cold",
                                     "mf": {"type": "trimf",
                                            "params": [0, 0, 10]}
                                    }
                                }
                             })

    yield lv

    lv = None


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


def test_load_antecedent(setup):

    ant = Antecedent("temperature IS cold", {'temperature': setup})
    assert ant.to_string() == "temperature IS cold"
    assert ant.term == setup.terms['cold']


def test_get_cylindrical_extension_antecedent_without_NOT(setup):

    ant = Antecedent("temperature IS cold", {'temperature': setup})
    assert ant.get_cylindrical_extension({"temperature":
                                          {"cold": 0.3,
                                           "very_cold": 0.2}}) == 0.3


def test_get_cylindrical_extension_antecedent_with_NOT(setup):

    ant = Antecedent("NOT temperature IS cold", {"temperature": setup})
    assert ant.get_cylindrical_extension({"temperature":
                                          {"cold": 0.3,
                                           "very_cold": 0.2}}) == 0.7


def test_load_antecedent_with_NOT(setup):

    ant = Antecedent("NOT temperature IS cold", {"temperature": setup})
    assert ant.to_string() == \
        "NOT temperature IS cold"


def test_load_antecedents_with_OR(setup):

    antecedents = Antecedents({"antecedent1": "temperature IS cold",
                               "operator": "OR",
                               "antecedent2": "temperature IS very_cold"},
                              {'temperature': setup})
    assert antecedents.to_string() == \
        "(temperature IS cold OR temperature IS very_cold)"


def test_load_antecedents_with_AND(setup):

    antecedents = Antecedents({"antecedent1": "temperature IS cold",
                               "operator": "AND",
                               "antecedent2": "temperature IS very_cold"},
                              {'temperature': setup})

    assert antecedents.to_string() == \
        "(temperature IS cold AND temperature IS very_cold)"


def test_load_antecedents_with_sub_antecedent(setup):
    antecedents = Antecedents({"antecedent1": "temperature IS cold",
                               "operator": "AND",
                               "antecedent2": {
                                   "antecedent1": "temperature IS cold",
                                   "operator": "OR",
                                   "antecedent2": "temperature IS cold"}},
                              {'temperature': setup})
    assert antecedents.to_string() == \
        "(temperature IS cold AND \
(temperature IS cold OR temperature IS cold))"


def test_load_antecedents_with_one_antecedent(setup):
    antecedents = Antecedents({"antecedent1": "temperature IS cold"},
                              {"temperature": setup})
    assert antecedents.to_string() == \
        "(temperature IS cold)"


def test_load_antecedents_with_not_antecedent(setup):
    antecedents = Antecedents({"antecedent1": "NOT temperature IS cold",
                               "operator": "AND",
                               "antecedent2": "temperature IS very_cold"},
                              {'temperature': setup})
    assert antecedents.to_string() == \
        "(NOT temperature IS cold AND temperature IS very_cold)"


def test_get_cylindrical_extension_antecedents_with_OR(setup):

    antecedents = Antecedents({"antecedent1": "temperature IS cold",
                               "operator": "OR",
                               "antecedent2": "temperature IS very_cold"},
                              {'temperature': setup})
    assert antecedents.get_cylindrical_extension({"temperature":
                                                  {"cold": 0.3,
                                                   "very_cold": 0.2}}) == 0.3


def test_get_cylindrical_extension_antecedents_with_AND(setup):

    antecedents = Antecedents({"antecedent1": "temperature IS cold",
                               "operator": "AND",
                               "antecedent2": "temperature IS very_cold"},
                              {'temperature': setup})
    assert antecedents.get_cylindrical_extension({"temperature":
                                                  {"cold": 0.3,
                                                   "very_cold": 0.2}}) == 0.2


def test_get_cylindrical_extension_antecedents_one_antecedent(setup):

    antecedents = Antecedents({"antecedent1": "temperature IS cold"},
                              {"temperature": setup})
    assert antecedents.get_cylindrical_extension({"temperature":
                                                  {"cold": 0.3,
                                                   "very_cold": 0.2}}) == 0.3


def test_nested_antecedents_get_cylindrical_extension(setup):

    antecedents = Antecedents({"antecedent1": "temperature IS very_cold",
                               "operator": "AND",
                               "antecedent2": {
                                   "antecedent1": "temperature IS cold",
                                   "operator": "OR",
                                   "antecedent2": "temperature IS very_cold"}},
                              {"temperature": setup})
    assert antecedents.get_cylindrical_extension({"temperature":
                                                  {"cold": 0.3,
                                                   "very_cold": 0.2}}) == 0.2
