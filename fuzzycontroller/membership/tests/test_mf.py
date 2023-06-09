from ..membership_functions import TriangularMF, \
        TrapezoidalMF, GauAngleMF
import skfuzzy as fuzz
import numpy as np
import pytest


@pytest.fixture
def tmf():
    universe = np.arange(0, 11, 1)
    tmf = TriangularMF(universe, [0, 5, 10])
    yield tmf
    tmf = None


@pytest.fixture
def trapmf():
    universe = np.arange(0, 11, 1)
    trapmf = TrapezoidalMF(universe, [0, 2, 8, 10])
    yield trapmf
    trapmf = None


def test_singleton_interp_mem_tri_peak(tmf):
    assert tmf.singleton_interp_mem(5) == 1


def test_singleton_interp_mem_tri(tmf):
    universe = np.arange(0, 11, 1)
    mf = fuzz.trimf(universe, [0, 5, 10])
    x = 3
    expected = fuzz.interp_membership(universe, mf, x)
    actual = tmf.singleton_interp_mem(x)
    assert expected == actual


def test_singleton_interp_mem_x_outside_universe_tri(tmf):
    x = 11
    with pytest.raises(ValueError):
        tmf.singleton_interp_mem(x)


def test_nonsingleton_interp_mem_similarity_tri(tmf):
    universe = np.arange(0, 11, 1)
    mf = fuzz.trimf(universe, [0, 5, 10])
    input_mf = fuzz.trimf(universe, [0, 2, 4])
    expected = np.sum(np.fmin(input_mf, mf)) / np.sum(np.fmax(input_mf, mf))
    actual = tmf.nonsingleton_interp_mem(input_mf, "similarity")
    print(expected)
    assert expected == actual


def test_nonsingleton_interp_mem_centroid_tri(tmf):
    universe = np.arange(0, 11, 1)
    mf = fuzz.trimf(universe, [0, 5, 10])
    input_mf = fuzz.trimf(universe, [0, 2, 4])
    x = fuzz.defuzz(universe, np.fmin(input_mf, mf), "centroid")
    expected = fuzz.interp_membership(universe, mf, x)
    actual = tmf.nonsingleton_interp_mem(input_mf, "centroid")
    assert expected == actual


def test_nonsingleton_interp_mem_centroid_no_overlap_tri():
    universe = np.arange(0, 11, 1)
    input_mf = fuzz.trimf(universe, [6, 7, 8])
    expected = 0
    tmf = TriangularMF(universe, [0, 1, 2])
    actual = tmf.nonsingleton_interp_mem(input_mf, "centroid")
    assert expected == actual


def test_singleton_interp_mem_trap(trapmf):
    universe = np.arange(0, 11, 1)
    mf = fuzz.trapmf(universe, [0, 2, 8, 10])
    x = 3
    expected = fuzz.interp_membership(universe, mf, x)
    actual = trapmf.singleton_interp_mem(x)
    assert expected == actual


def test_singleton_interp_mem_trap_outside_universe(trapmf):
    x = 11
    with pytest.raises(ValueError):
        trapmf.singleton_interp_mem(x)


def test_nonsingleton_interp_mem_similarity_trap():
    universe = np.arange(0, 11, 1)
    mf = fuzz.trapmf(universe, [0, 2, 8, 10])
    input_mf = fuzz.trapmf(universe, [0, 1, 4, 5])
    expected = np.sum(np.fmin(input_mf, mf)) / np.sum(np.fmax(input_mf, mf))
    tmf = TrapezoidalMF(universe, [0, 2, 8, 10])
    actual = tmf.nonsingleton_interp_mem(input_mf, "similarity")
    assert expected == actual


def test_nonsingleton_interp_mem_centroid_trap():
    universe = np.arange(0, 11, 1)
    mf = fuzz.trapmf(universe, [0, 2, 8, 10])
    input_mf = fuzz.trapmf(universe, [0, 1, 4, 5])
    x = fuzz.defuzz(universe, np.fmin(input_mf, mf), "centroid")
    expected = fuzz.interp_membership(universe, mf, x)
    tmf = TrapezoidalMF(universe, [0, 2, 8, 10])
    actual = tmf.nonsingleton_interp_mem(input_mf, "centroid")
    assert expected == actual


def test_nonsingleton_interp_mem_centroid_no_overlap_trap():
    universe = np.arange(0, 11, 1)
    input_mf = fuzz.trapmf(universe, [6, 7, 8, 9])
    expected = 0
    tmf = TrapezoidalMF(universe, [0, 1, 2, 3])
    actual = tmf.nonsingleton_interp_mem(input_mf, "centroid")
    assert expected == actual


def test_singleton_interp_mem_gauangle_inside_gaussian():
    univsere = np.arange(0, 11, 1)
    x = 3
    mf = fuzz.gaussmf(univsere, 3, 1)
    expected = fuzz.interp_membership(univsere, mf, x)
    gamf = GauAngleMF(univsere, [3, 1], 0, 5)
    actual = gamf.singleton_interp_mem(x)
    assert expected == actual


def test_singleton_interp_mem_gauanle_with_float_start_stop():
    universe = np.arange(0, 11, 0.1)
    gamf = GauAngleMF(universe, [3, 1], 2.1, 4.4)
    gmf = fuzz.gaussmf(universe, 3, 1)
    assert gamf.singleton_interp_mem(1.9) == 0
    assert gamf.singleton_interp_mem(2.2) == \
        fuzz.interp_membership(universe, gmf, 2.2)
    assert gamf.singleton_interp_mem(4.3) == \
        fuzz.interp_membership(universe, gmf, 4.3)
    assert gamf.singleton_interp_mem(4.6) == 0


def test_singleton_interp_mem_gauangle_with_float_start_stop_temp():
    universe = np.arange(0, 60.1, 0.1)
    gamf = GauAngleMF(universe, [35.3, 0.25], 35, 35.8)
    assert gamf.mf[350] == 0
    assert gamf.mf[358] == 0
    assert gamf.mf[351] != 0
    assert gamf.mf[357] != 0


def test_singleton_interp_mem_gauangle_outside_range_lower():
    universe = np.arange(0, 11, 1)
    x = 2
    gamf = GauAngleMF(universe, [4, 1], 2, 6)
    assert gamf.singleton_interp_mem(x) == 0


def test_singleton_interp_mem_gauangle_outside_range_higher():
    univser = np.arange(0, 11, 1)
    x = 6
    gamf = GauAngleMF(univser, [4, 1], 2, 6)
    assert gamf.singleton_interp_mem(x) == 0


def test_singleton_interp_mem_gauangle_outside_range_lower_high_resolution():
    universe = np.arange(0, 11, 0.1)
    x = 2
    gamf = GauAngleMF(universe, [4, 1], 2, 6)
    assert gamf.singleton_interp_mem(x) == 0


def test_gauangle_with_no_start():
    universe = np.arange(0, 11, 1)
    gamf = GauAngleMF(universe, [4, 1], -1, 6)
    assert gamf.mf[0] != 0
    assert gamf.singleton_interp_mem(4) == 1.0


def test_gauangle_with_no_end():
    universe = np.arange(0, 11, 1)
    gamf = GauAngleMF(universe, [4, 1], 0, -1)
    assert gamf.mf[-1] != 0
    assert gamf.singleton_interp_mem(4) == 1.0
