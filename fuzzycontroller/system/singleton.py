from __future__ import annotations
from .fis import FIS


class SingletonFIS(FIS):
    """A Singleton Inference System.

    A singleton inference system calculates firing strengths / degrees of
    membership using a singleton fuzzy set / a spike for each crisp input.
    Inherits from :class:`FIS`.
    """

    def __init__(self) -> None:
        """Initializes the singleton inference system."""
        super().__init__()
        self._type = "singleton"

    @property
    def type(self):
        """Returns the type of the inference system."""
        return self._type

    def get_all_firing_strengths(self, crisp_inputs: dict[str, float]) \
            -> dict[str, dict[str, float]]:
        """Computes the firing strength of all the linguistic terms
        from inputs. Inputs should be of the form {'input1': 1.0,
        'input2': 2.0} where the keys are the names of the linguistic
        variables and the values are the crisp inputs.

        Args:
            crisp_inputs: dictionary containing the crisp inputs.

        Returns:
            dictionary containing the firing strengths of all the linguistic
            terms. Will be of the form {'input1': {('term1', 0.5), ...},
            'input2': ...}

        """
        return {key: self.variables[key].compute_memberships(
                crisp_inputs[key], self.type)
                for key in crisp_inputs.keys()}


