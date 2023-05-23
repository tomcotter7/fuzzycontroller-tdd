from fuzzycontroller.membership.membership_functions import TriangularMF, \
        TrapezoidalMF, GauAngleMF, MembershipFunction
import numpy as np


class LinguisticTerm:

    def __init__(self, universe: np.ndarray, term: dict) -> None:
        self.name = term['name']
        self.universe = universe
        self.mf = self._load_mf(term['mf'])

    def __eq__(self, other) -> bool:
        if isinstance(other, LinguisticTerm):
            return self.name == other.name and \
                    np.array_equal(self.universe, other.universe) and \
                    self.mf == other.mf
        return False

    def _load_mf(self, mf: dict) -> MembershipFunction:
        """
        Load the membership function from a dictionary.
        Should be of the form {'type': 'trimf', 'params': [a, b, c]}

        Args:
            mf: dictionary containing the membership function info.
        """
        if mf['type'] == "trimf":
            return TriangularMF(self.universe, mf['params'])
        elif mf['type'] == "trapmf":
            return TrapezoidalMF(self.universe, mf['params'])
        return GauAngleMF(self.universe, mf['params'],
                          float(mf['start']), float(mf['end']))

    def compute_membership(self, crisp_input: float or np.ndarray,
                           input_type: str) -> float:
        """
        Compute the membership value of the given crisp input.

        Args:
            crisp_input: crisp input value.
            input_type: type of the input, either 'singleton' or
                'non-singleton'
        Returns:
            membership value of the given crisp input.
        """
        if input_type == "singleton":
            return self.mf.singleton_interp_mem(crisp_input)

        return 0.0

    def apply_fmax_or_fmin(self, f, other_input):
        """
        Apply the fmax or fmin function to the membership function of this
        linguistic term and another input np.ndarray. Used for applying OR
        and AND, as well as for applying the implication function.

        Args:
            f: either np.fmax or np.fmin
            other_input: np.ndarray of the other input

        Returns:
            np.ndarray: the result of applying f to the membership function of
                this linguistic term and other_input.
        """

        return f(self.mf.mf, other_input)
