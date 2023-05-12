from fuzzyhealthsystem.utils.json_handler import JsonHandler


def test_load_file():
    jh = JsonHandler()
    data = jh.read('params.json')
    assert data['inputs']['temperature']['universe']['start'] == '0'
