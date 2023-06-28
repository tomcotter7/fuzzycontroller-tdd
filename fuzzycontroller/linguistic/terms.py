from numpy._typing import NDArray
from ..membership.membership_functions import TriangularMF, \
        TrapezoidalMF, GauAngleMF, MembershipFunction
import numpy as np


class LinguisticTerm:
    """A linguistic term.

    An example of a linguistic term is 'hot', which is a term that would come
    under the linguistic variable 'temperature'. The term 'hot' would be
    defined by a membership function.

    Attributes:
        name: A string representation of the name.
        universe: A numpy array representation of the universe of discourse.
        mf: A MembershipFunction object defining the
            associated membership function.
    """

    def __init__(self, universe: NDArray, term: dict) -> None:
        """Initializes the LinguisticTerm based on the universe and a
        dictionary.

        Args:
            universe: the universe of discourse
            term: a dictionary defining the term. should be of the
                form, {'name': str, 'mf': {'type': str, 'params': lst}}
        """
        self.name = term['name']
        self.universe = universe
        self.mf = self._load_mf(term['mf'])

    def __eq__(self, other) -> bool:
        """Tests equality of this LinguisticTerm and another.

        Args:
            other: The linguistic term to test against

        Returns:
            A boolean which is true when the two terms are equal,
            i.e they have the same name, universe and mf.
        """
        if isinstance(other, LinguisticTerm):
            return self.name == other.name and \
                    np.array_equal(self.universe, other.universe) and \
                    self.mf == other.mf
        return False

    def _load_mf(self, mf: dict) -> MembershipFunction:
        """Loads the membership function from a dictionary.
        Example dict: {'type': 'trimf', 'params': [1, 5, 10]}

        Args:
            mf: dictionary containing the membership function info.

        Returns:
            A membership function defined in the dictionary.
        """
        if mf['type'] == "trimf":
            return TriangularMF(self.universe, mf['params'])
        elif mf['type'] == "trapmf":
            return TrapezoidalMF(self.universe, mf['params'])
        return GauAngleMF(self.universe, mf['params'],
                          float(mf['start']), float(mf['end']))

    def compute_membership(self, input_value,
                           input_type: str) -> float:
        """Compute the membership value of the given input.

        Args:
            crisp_input: crisp input value.
            input_type: type of the input, either 'singleton' or
                'non-singleton'
        Returns:
            membership value of the given crisp input.
        """
        if input_type == "singleton" and isinstance(input_value, float):
            return self.mf.singleton_interp_mem(input_value)

        if input_type == "non-singleton" and isinstance(input_value,
                                                        MembershipFunction):
            return self.mf.nonsingleton_interp_mem(input_value.mf,
                                                   defuzz='similarity')
        return 0.0

    def apply_fmax_or_fmin(self, f, other_input) -> NDArray:
        """Apply the fmax or fmin function to the membership function of this
        linguistic term and another input np.ndarray/float. Used for applying
        OR and AND, as well as for applying the implication function.

        Args:
            f: either np.fmax or np.fmin
            other_input: the input np.ndarray/float

        Returns:
            the result of applying f to the membership function of
                this linguistic term and other_input.
        """

        return f(self.mf.mf, other_input)
    
    def graph(self, ax):
        """Plot this linguistic term's membership function."""
        self.mf.graph(ax, self.name)
