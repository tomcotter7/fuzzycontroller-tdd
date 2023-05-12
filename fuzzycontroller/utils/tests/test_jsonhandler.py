from fuzzycontroller.utils.json_handler import JsonHandler


def test_load_file():
    jh = JsonHandler()
    data = jh.read('fuzzycontroller/system/tests/data.json')
    assert data['inputs']['temperature']['universe']['start'] == '0'
