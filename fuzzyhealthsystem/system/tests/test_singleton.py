from fuzzyhealthsystem.system.singleton import SingletonFIS
import numpy as np


def test_load_linguistic_variable():
    sfis = SingletonFIS()
    sfis.load_data("fuzzyhealthsystem/system/tests/data.json")
    assert sfis.input_variables["temperature"].name == "temperature"
    assert np.array_equal(sfis.input_variables["temperature"].universe,
                          np.arange(0, 60.1, 0.1))


def test_load_linguistic_variable_with_multiple_terms():
    sfis = SingletonFIS()
    sfis.load_data("fuzzyhealthsystem/system/tests/data.json")
    assert len(sfis.input_variables["temperature"].terms) == 2
    assert sfis.input_variables["temperature"].terms[1].name == "Cold"

