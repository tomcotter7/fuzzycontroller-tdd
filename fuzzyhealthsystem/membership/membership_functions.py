from abc import ABC, abstractmethod
import numpy as np
import skfuzzy as fuzz


class MembershipFunction(ABC):
    @property
    @abstractmethod
    def universe(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def mf(self) -> np.ndarray:
        pass

    def __eq__(self, other):
        if isinstance(other, MembershipFunction):
            return np.array_equal(self.universe, other.universe) and \
                np.array_equal(self.mf, other.mf)
        return False

    def singleton_interp_mem(self, x) -> float:
        """
        Compute the membership value of a singleton fuzzy set x with the
        membership function mf (an attribute of this class).

        Args:
            x (float): The value of the singleton fuzzy set.

        Returns:
            float: The membership value of x in mf.

        """
        if x not in self.universe:
            raise ValueError("X is outside range of universe")
        return fuzz.interp_membership(self.universe, self.mf, x)

    def nonsingleton_interp_mem(self, input_mf: np.ndarray,
                                defuzz: str) -> float:
        """
        Compute the membership value of a non-singleton fuzzy set with the
        membership function mf (an attribute of this class).

        Args:
            input_mf (np.ndarray): The input fuzzy set to be compared to mf.
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

    def __init__(self, universe: np.ndarray, params: list):
        """
        Create a triangular membership function.

        Args:
            universe (np.ndarray): The universe of discourse for the fuzzy set.
            params (list): The parameters of the triangular membership function
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

    def __init__(self, universe: np.ndarray, params: list):
        """
        Create a trapezoidal membership function.

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

    def __init__(self, universe: np.ndarray, params: list,
                 start: int, end: int):
        """
        Create a GauAngle membership function.

        Args:
            universe (np.ndarray): The universe of discourse for the fuzzy set.
            params (list): The parameters of the GauAngle membership function
                [mean, sigma]
            start (int): The first value at which the membership
                function is non-zero.
            end (int): The last value at which the membership
                function is non-zero.
        """
        self._universe = universe
        self._mf = fuzz.gaussmf(universe, params[0], params[1])
        step = universe[1] - universe[0]
        start_idx = int(start * (1 / step)) + 1
        end_idx = int(end * (1 / step))
        self._mf[:start_idx] = [0] * start_idx
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
