from fuzzycontroller.rule.rule import Rule
from fuzzycontroller.linguistic.variables import LinguisticVariable


def test_load_rule():

    lv_ant = LinguisticVariable("temperature",
                                {"universe": {"start": 0, "end": 60,
                                              "step": 0.1},
                                 "terms": {
                                     "cold": {
                                         "name": "cold",
                                         "mf": {"type": "trimf",
                                                "params": [0, 10, 20]}
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

    rule = {"antecedent": {"antecedent1": "temperature IS cold"},
            "consequent": "layers IS many"}

    r = Rule(rule,
             {'temperature': lv_ant, 'layers': lv_conq})
    assert r.to_string() == "IF (temperature IS cold) THEN layers IS many"
    assert r.antecedents.antecedents['antecedent1'].term \
        == lv_ant.get_term('cold')
    assert r.consequent.term == lv_conq.get_term('many')
