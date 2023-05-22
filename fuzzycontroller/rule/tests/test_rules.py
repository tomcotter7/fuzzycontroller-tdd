from fuzzycontroller.rule.rules import Rules
from fuzzycontroller.linguistic.variables import LinguisticVariable


def test_load_rules():

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

    rules = {"rule1": {"antecedent": {"antecedent1": "temperature IS cold"},
                       "consequent": "layers IS many"},
             "rule2": {"antecedent": {"antecedent1": "temperature IS hot"},
                       "consequent": "layers IS few"}}
    rg = Rules(rules, {'temperature': lv_ant, 'layers': lv_conq})
    assert rg.rules['rule1'].to_string() \
        == "IF (temperature IS cold) THEN layers IS many"
    assert rg.rules['rule2'].to_string() \
        == "IF (temperature IS hot) THEN layers IS few"
