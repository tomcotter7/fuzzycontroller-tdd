from fuzzycontroller.rule.rules import Rules
from fuzzycontroller.linguistic.variables import LinguisticVariable
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
                                     "hot": {
                                         "name": "hot",
                                         "mf": {"type": "trimf",
                                                "params": [40, 50, 60]}
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
                                        },
                                      "few": {
                                          "name": "few",
                                          "mf": {"type": "trimf",
                                                 "params": [0, 0, 2]}
                                        }
                                    }
                                  })
    yield [lv_ant, lv_conq]

    lv_ant = None
    lv_conq = None


def test_load_rules(setup):

    rules = {"rule1": {"antecedent": {"antecedent1": "temperature IS cold"},
                       "consequent": "layers IS many"},
             "rule2": {"antecedent": {"antecedent1": "temperature IS hot"},
                       "consequent": "layers IS few"}}
    rg = Rules(rules, {'temperature': setup[0], 'layers': setup[1]})
    assert rg.rules['rule1'].to_string() \
        == "IF (temperature IS cold) THEN layers IS many"
    assert rg.rules['rule2'].to_string() \
        == "IF (temperature IS hot) THEN layers IS few"


def test_get_correct_output_sets(setup):

    rules = {"rule1": {"antecedent": {"antecedent1": "temperature IS cold"},
                       "consequent": "layers IS many"},
             "rule2": {"antecedent": {"antecedent1": "temperature IS hot"},
                       "consequent": "layers IS few"}}
    rg = Rules(rules, {'temperature': setup[0], 'layers': setup[1]})
    rg.get_correct_output_sets({'temperature': {'cold': 1.0,
                                                'hot': 0.0}})

    assert np.array_equal(rg.output_sets['rule1'],
                          setup[1].get_term('many').mf.mf)
    assert np.array_equal(rg.output_sets['rule2'],
                          np.zeros(len(setup[1].universe)))
