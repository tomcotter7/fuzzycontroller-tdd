from ..rule import Rule
from ...linguistic.variables import LinguisticVariable
import pytest
import numpy as np


@pytest.fixture
def setup():

    lv_ant = LinguisticVariable("temperature",
                                {"universe": {"start": 0, "end": 60,
                                              "step": 0.1},
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
    lv_conq = LinguisticVariable("layers",
                                 {"universe": {"start": 0, "end": 5,
                                               "step": 1},
                                  "terms": {
                                      "many": {
                                          "name": "many",
                                          "mf": {"type": "trimf",
                                                 "params": [3, 5, 5]}
                                          }
                                      }
                                  })
    yield [lv_ant, lv_conq]

    lv_ant = None
    lv_conq = None


def test_load_rule(setup):

    rule = {"antecedent": {"antecedent1": "temperature IS cold"},
            "consequent": "layers IS many"}

    r = Rule(rule,
             {'temperature': setup[0], 'layers': setup[1]})
    assert r.to_string() == "IF (temperature IS cold) THEN layers IS many"
    assert r.antecedents.antecedents['antecedent1'].term \
        == setup[0].get_term('cold')
    assert r.consequent.term == setup[1].get_term('many')


def test_apply_rule(setup):

    rule = {"antecedent": {"antecedent1": "temperature IS cold"},
            "consequent": "layers IS many"}

    r = Rule(rule,
             {'temperature': setup[0], 'layers': setup[1]})

    assert np.array_equal(r.apply_rule({'temperature': {'cold': 1.0}}),
                          setup[1].terms['many'].mf.mf)
