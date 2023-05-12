from fuzzyhealthsystem.system.singleton import SingletonFIS
import numpy as np
import os


def test_load_linguistic_variable():
    sfis = SingletonFIS()
    sfis.load_data("fuzzyhealthsystem/system/tests/data.json")
    assert sfis.input_variables["temperature"].name == "temperature"
    assert np.array_equal(sfis.input_variables["temperature"].universe,
                          np.arange(0, 60, 0.1))
