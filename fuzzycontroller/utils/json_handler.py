import json


class JsonHandler():
    """A JSon Handler class"""

    def read(self, path: str):
        """Reads a json file and returns a dictionary.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return data
