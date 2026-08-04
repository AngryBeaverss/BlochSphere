import numpy as np

from bloch_orderfinder.core.backend import get_backend
from bloch_orderfinder.core.encoder import ExponentMapping


def test_exponent_mapping_default_row_stride_matches_v3():
    backend = get_backend('cpu')
    m = ExponentMapping(size=5, row_stride=None, origin=0)
    g = backend.to_cpu(m.grid(backend))
    # v3 convention: exp[y,x] = y*size + x
    expected = np.array([[y*5 + x for x in range(5)] for y in range(5)], dtype=np.int64)
    assert np.array_equal(g, expected)


def test_exponent_mapping_custom_row_stride():
    backend = get_backend('cpu')
    m = ExponentMapping(size=4, row_stride=10, origin=7)
    g = backend.to_cpu(m.grid(backend))
    expected = np.array([[7 + y*10 + x for x in range(4)] for y in range(4)], dtype=np.int64)
    assert np.array_equal(g, expected)
