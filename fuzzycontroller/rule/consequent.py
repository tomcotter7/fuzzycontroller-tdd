class Consequent:

    def __init__(self, name, term):
        self.name = name
        self.term = term

    def to_string(self) -> str:
        """
        e.g. "temperature IS cold"

        Returns:
            string representation of the consequent.
        """
        return f"{self.name} IS {self.term.name}"
