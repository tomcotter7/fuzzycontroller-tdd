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
