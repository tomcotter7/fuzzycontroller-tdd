from ..singleton import SingletonFIS
import numpy as np
import skfuzzy as fuzz
import pytest


@pytest.fixture
def sfis():
    sfis = SingletonFIS()
    sfis.load_data("fuzzycontroller/system/tests/data.json")
    yield sfis

    sfis = None


def test_load_linguistic_variable(sfis):
    assert sfis.variables["temperature"].name == "temperature"
    assert np.array_equal(sfis.variables["temperature"].universe,
                          np.arange(0, 60.1, 0.1))


def test_load_linguistic_variable_with_multiple_terms(sfis):
    assert "cold" in sfis.variables["temperature"].terms
    assert "very_cold" in sfis.variables["temperature"].terms


def test_load_multiple_linguistic_variables(sfis):
    assert "temperature" in sfis.variables
    assert "headache" in sfis.variables


def test_get_all_firing_strengths(sfis):
    fs = sfis.get_all_firing_strengths(({"temperature": 30.0}))
    assert fs["temperature"]["very_cold"] == 1.0
    assert fs["temperature"]["cold"] == 0.0
    assert fs["temperature"]["standard"] == 0.0


def test_load_output(sfis):
    assert sfis.output_variable == "urgency"


def test_load_rule(sfis):
    s = "IF (temperature IS very_cold OR temperature IS very_hot)" + \
        " THEN urgency IS emergency"
    assert sfis.rules.rules['rule1'].to_string() == s


def test_compute_output_sets(sfis):
    output_sets = sfis.compute_output_sets({"temperature": 30.0,
                                            "headache": 5.0,
                                            "age": 65.0})
    assert np.array_equal(output_sets['rule1'],
                          fuzz.trapmf(np.arange(0, 100.1, 0.1),
                                      [85, 95, 100, 100]))


def test_compute_aggregate_sets():
    sfis = SingletonFIS()
    sfis.load_data("fuzzycontroller/system/tests/smaller_data.json")
    aggregate_set = sfis.compute_aggregate_set({"temperature": 37.0,
                                                "headache": 5})
    assert np.array_equal(aggregate_set,
                          fuzz.trapmf(np.arange(0, 100, 0.1),
                                      [0, 10, 20, 30]))


def test_compute_defuzzified_output():
    sfis = SingletonFIS()
    sfis.load_data("fuzzycontroller/system/tests/smaller_data.json")
    output = sfis.compute_defuzzified_output({"temperature": 37.0,
                                              "headache": 5})
    u = np.arange(0, 100, 0.1)
    assert output == fuzz.defuzz(u, fuzz.trapmf(u, [0, 10, 20, 30]),
                                 'centroid')


def test_full_urgency_system(sfis):
    output = sfis.compute_defuzzified_output({"temperature": 34.0,
                                              "headache": 4,
                                              "age": 65})
    assert output == 93.34755751076897
