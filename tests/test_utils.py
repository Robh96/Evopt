import pytest
from evopt import utils
import numpy as np
import pandas as pd
import os

def test_convert_to_native():
    assert utils.convert_to_native(np.float64(1.23456)) == 1.235
    assert utils.convert_to_native([np.float64(1.23456), 2.345]) == [1.235, 2.345]
    assert utils.convert_to_native({'a': np.float64(1.23456), 'b': 2.345}) == {'a': 1.235, 'b': 2.345}
    assert utils.convert_to_native(None) == 'None'

def test_format_array():
    arr = np.array([1.23456, 2.34567, 3.45678])
    assert utils.format_array(arr, precision=3) == "1.235, 2.346, 3.457"
    assert utils.format_array(arr, precision=2) == "1.23, 2.35, 3.46"

def test_write_to_csv(tmpdir):
    csv_path = os.path.join(tmpdir, "test.csv")
    data = {'a': 1, 'b': 2.345, 'c': 'test'}
    utils.write_to_csv(data, csv_path)
    df = pd.read_csv(csv_path)
    assert df['a'][0] == 1
    assert df['b'][0] == 2.345
    assert df['c'][0] == 'test'
