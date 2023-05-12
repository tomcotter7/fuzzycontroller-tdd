from fuzzycontroller.membership.membership_functions import TriangularMF, \
        TrapezoidalMF, GauAngleMF
import skfuzzy as fuzz
import numpy as np
import pytest


def test_singleton_interp_mem_tri():
    universe = np.arange(0, 11, 1)
    mf = fuzz.trimf(universe, [0, 5, 10])
    x = 3
    expected = fuzz.interp_membership(universe, mf, x)
    tmf = TriangularMF(universe, [0, 5, 10])
    actual = tmf.singleton_interp_mem(x)
    assert expected == actual


def test_singleton_interp_mem_x_outside_universe_tri():
    universe = np.arange(0, 11, 1)
    tmf = TriangularMF(universe, [0, 5, 10])
    x = 11
    with pytest.raises(ValueError):
        tmf.singleton_interp_mem(x)


def test_nonsingleton_interp_mem_similarity_tri():
    universe = np.arange(0, 11, 1)
    mf = fuzz.trimf(universe, [0, 5, 10])
    input_mf = fuzz.trimf(universe, [0, 2, 4])
    expected = np.sum(np.fmin(input_mf, mf)) / np.sum(np.fmax(input_mf, mf))
    tmf = TriangularMF(universe, [0, 5, 10])
    actual = tmf.nonsingleton_interp_mem(input_mf, "similarity")
    assert expected == actual


def test_nonsingleton_interp_mem_centroid_tri():
    universe = np.arange(0, 11, 1)
    mf = fuzz.trimf(universe, [0, 5, 10])
    input_mf = fuzz.trimf(universe, [0, 2, 4])
    x = fuzz.defuzz(universe, np.fmin(input_mf, mf), "centroid")
    expected = fuzz.interp_membership(universe, mf, x)
    tmf = TriangularMF(universe, [0, 5, 10])
    actual = tmf.nonsingleton_interp_mem(input_mf, "centroid")
    assert expected == actual


def test_nonsingleton_interp_mem_centroid_no_overlap_tri():
    universe = np.arange(0, 11, 1)
    input_mf = fuzz.trimf(universe, [6, 7, 8])
    expected = 0
    tmf = TriangularMF(universe, [0, 1, 2])
    actual = tmf.nonsingleton_interp_mem(input_mf, "centroid")
    assert expected == actual


def test_singleton_interp_mem_trap():
    universe = np.arange(0, 11, 1)
    mf = fuzz.trapmf(universe, [0, 2, 8, 10])
    x = 3
    expected = fuzz.interp_membership(universe, mf, x)
    tmf = TrapezoidalMF(universe, [0, 2, 8, 10])
    actual = tmf.singleton_interp_mem(x)
    assert expected == actual


def test_singleton_interp_mem_trap_outside_universe():
    universe = np.arange(0, 11, 1)
    tmf = TrapezoidalMF(universe, [0, 2, 8, 10])
    x = 11
    with pytest.raises(ValueError):
        tmf.singleton_interp_mem(x)


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
