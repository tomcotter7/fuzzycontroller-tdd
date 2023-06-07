from ..nonsingleton import NonSingletonFIS
import numpy as np
import pytest
import skfuzzy as fuzz


@pytest.fixture
def nsfis():
    nsfis = NonSingletonFIS()
    nsfis.load_data("fuzzycontroller/system/tests/data.json")
    yield nsfis

    nsfis = None


def test_load_linguistic_variable(nsfis):
    assert nsfis.variables["temperature"].name == "temperature"
    assert np.array_equal(nsfis.variables["temperature"].universe,
                          np.arange(0, 60.1, 0.1))


def test_load_linguistic_variable_with_multiple_terms(nsfis):
    assert "cold" in nsfis.variables["temperature"].terms
    assert "very_cold" in nsfis.variables["temperature"].terms


def test_load_multiple_linguistic_variables(nsfis):
    assert "temperature" in nsfis.variables
    assert "headache" in nsfis.variables


def test_create_input_fuzzy_sets(nsfis):
    inputs = {"temperature": {"start": 34, "end": 36.5}}
    temp = nsfis.create_input_sets(inputs)["temperature"]
    mean = (36.5 + 34) / 2
    sd = (36.5 - 34) / 4
    temp_universe = np.arange(0, 60.1, 0.1)
    gaussian_fs = fuzz.gaussmf(temp_universe, mean, sd)
    assert temp.singleton_interp_mem(35) == \
        fuzz.interp_membership(temp_universe, gaussian_fs, 35)
    assert temp.singleton_interp_mem(33.9) == 0


def test_get_all_firing_strengths_temp_cold(nsfis):
    inputs = {"temperature": {"start": 34, "end": 36.5}}
    fs = nsfis.get_all_firing_strengths(inputs)
    assert round(fs["temperature"]["very_cold"], 5) == 0.00455
    assert round(fs["temperature"]["cold"], 5) == 0.34157
    assert round(fs["temperature"]["standard"], 5) == 0.04793


def test_get_all_firing_strengths_temp_hot(nsfis):
    inputs = {"temperature": {"start": 38, "end": 40}}
    fs = nsfis.get_all_firing_strengths(inputs)
    assert round(fs["temperature"]["hot"], 5) == 0.64613
    assert round(fs["temperature"]["very_hot"], 5) == 0.00891


def test_defuzzified_output(nsfis):
    inputs = {"temperature": {"start": 36.5, "end": 38},
              "headache": {"start": 3, "end": 4},
              "age": {"start": 10, "end": 15}}
    output = nsfis.compute_defuzzified_output(inputs)
    assert round(output, 5) == 35.44026
