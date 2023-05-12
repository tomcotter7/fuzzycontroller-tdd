from fuzzycontroller.system.singleton import SingletonFIS
import numpy as np


def test_load_linguistic_variable():
    sfis = SingletonFIS()
    sfis.load_data("fuzzycontroller/system/tests/data.json")
    assert sfis.input_variables["temperature"].name == "temperature"
    assert np.array_equal(sfis.input_variables["temperature"].universe,
                          np.arange(0, 60.1, 0.1))


def test_load_linguistic_variable_with_multiple_terms():
    sfis = SingletonFIS()
    sfis.load_data("fuzzycontroller/system/tests/data.json")
    assert len(sfis.input_variables["temperature"].terms) == 2
    assert sfis.input_variables["temperature"].terms[1].name == "Cold"


def test_load_multiple_linguistic_variables():
    sfis = SingletonFIS()
    sfis.load_data("fuzzycontroller/system/tests/data.json")
    assert len(sfis.input_variables) == 2


def test_get_all_firing_strengths():
    sfis = SingletonFIS()
    sfis.load_data("fuzzycontroller/system/tests/data.json")
    assert sfis.get_all_firing_strengths({"temperature": 30}) == \
        {"temperature": [("Very Cold", 1.0), ("Cold", 0.0)]}


def test_load_output():
    sfis = SingletonFIS()
    sfis.load_data("fuzzycontroller/system/tests/data.json")
    assert sfis.output_variable.name == "urgency"
