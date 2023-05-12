import json


class JsonHandler():

    def read(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
