from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import skfuzzy as fuzz


class MembershipFunction(ABC):
    """An abstract class for a any MembershipFunction

    Contains all the common functionality for any membership function.

    Attributes:
        _universe: a numpy array representing the universe
        _mf: a numpy array (wrapped with the skfuzzy library)
            representing the membership function
    """

    @property
    @abstractmethod
    def universe(self) -> NDArray:
        """The universe"""
        pass

    @property
    @abstractmethod
    def mf(self) -> NDArray:
        """The membership function"""
        pass

    def __eq__(self, other):
        """Compares the MembershipFunction with another
        to determine equality

        Args:
            other: Another MembershipFunction

        Returns:
            A boolean, where True represents the two inputs being
            equal
        """
        if isinstance(other, MembershipFunction):
            return np.array_equal(self.universe, other.universe) and \
                np.array_equal(self.mf, other.mf)
        return False

    def singleton_interp_mem(self, x) -> float:
        """Computes the membership value of a singleton fuzzy set x with
        this class

        Args:
            x: The value of the singleton fuzzy set.

        Returns:
            The membership value of x in this class.

        """
        if x > max(self.universe) or x < min(self.universe):
            raise ValueError("%f is outside range of universe" % (x))
        return fuzz.interp_membership(self.universe, self.mf, x)

    def nonsingleton_interp_mem(self, input_mf: NDArray,
                                defuzz: str) -> float:
        """Computes the membership value of a non-singleton fuzzy set with this
        class.

        Args:
            input_mf (np.ndarray): The input fuzzy set to be compared to this
                class.
            defuzz (string): The defuzzification method to use. valid values
                are "centroid" and "similarity".

        Returns:
            float: The membership value of input_mf in mf.
        """
        if defuzz == "centroid":
            agg = np.fmin(input_mf, self.mf)
            try:
                x = fuzz.defuzz(self.universe, agg, defuzz)
                return fuzz.interp_membership(self.universe, self.mf, x)
            except AssertionError:
                return 0
        elif defuzz == "similarity":
            num = np.sum(np.fmin(input_mf, self.mf))
            den = np.sum(np.fmax(input_mf, self.mf))
            return num / den

        return -1


class TriangularMF(MembershipFunction):
    """A Triangular Membership Function"""

    def __init__(self, universe: NDArray, params: list):
        """Initializes a triangular membership function.

        Args:
            universe: The universe of discourse for the fuzzy set.
            params: The parameters of the triangular membership function
                [a, b, c]
        """
        self._universe = universe
        self._mf = fuzz.trimf(universe, params)

    @property
    def universe(self):
        return self._universe

    @property
    def mf(self):
        return self._mf

    def singleton_interp_mem(self, x: float):
        return super().singleton_interp_mem(x)

    def nonsingleton_interp_mem(self, input_mf: np.ndarray,
                                defuzz: str):
        return super().nonsingleton_interp_mem(input_mf, defuzz)


class TrapezoidalMF(MembershipFunction):
    """A Trapezoidal Membership Function"""

    def __init__(self, universe: np.ndarray, params: list):
        """Initializes a trapezoidal membership function.

        Args:
            universe (np.ndarray): The universe of discourse for the fuzzy set.
            params (list): The parameters of the trapezoidal membership
                function [a, b, c, d]
        """
        self._universe = universe
        self._mf = fuzz.trapmf(universe, params)

    @property
    def universe(self):
        return self._universe

    @property
    def mf(self):
        return self._mf

    def singleton_interp_mem(self, x: float):
        return super().singleton_interp_mem(x)

    def nonsingleton_interp_mem(self, input_mf: np.ndarray,
                                defuzz: str):
        return super().nonsingleton_interp_mem(input_mf, defuzz)


class GauAngleMF(MembershipFunction):
    """A GauAngle Membership Function

    This is a essentially a Gaussian Membership function,
    but with its values set to 0 outside of some interval
    """

    def __init__(self, universe: NDArray, params: list,
                 start: float, end: float):
        """Initializes a GauAngle membership function.

        Args:
            universe: The universe of discourse for the fuzzy set.
            params: The parameters of the GauAngle membership function
                [mean, sigma]
            start: The first value at which the membership
                function is non-zero.
            end: The last value at which the membership
                function is non-zero.
        """
        self._universe = universe
        self._mf = fuzz.gaussmf(universe, params[0], params[1])
        # Calculate the indices of the start / end points in the NDArray
        step = universe[1] - universe[0]
        start_idx = max(int(start * (1 / step)) + 1, 0)
        self._mf[:start_idx] = [0] * start_idx
        if end != -1:
            end_idx = int(end * (1 / step))
            self._mf[end_idx:] = [0] * (len(universe) - end_idx)

    @property
    def universe(self):
        return self._universe

    @property
    def mf(self):
        return self._mf

    def singleton_interp_mem(self, x: float):
        return super().singleton_interp_mem(x)

    def nonsingleton_interp_mem(self, input_mf: np.ndarray,
                                defuzz: str):
        return super().nonsingleton_interp_mem(input_mf, defuzz)
