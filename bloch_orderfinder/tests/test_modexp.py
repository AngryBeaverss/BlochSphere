import numpy as np

from bloch_orderfinder.core.backend import get_backend
from bloch_orderfinder.core.modexp import modexp_array_raw


def test_modexp_array_raw_matches_python_pow_cpu():
    backend = get_backend('cpu')
    xp = backend.xp
    N = 91
    a = 3
    exps = xp.arange(0, 128, dtype=xp.int64).reshape(32, 4)
    vals = modexp_array_raw(backend, exps, a=a, modN=N)
    vals_np = backend.to_cpu(vals)
    expected = np.vectorize(lambda e: pow(a, int(e), N))(np.arange(0, 128).reshape(32, 4))
    assert np.array_equal(vals_np, expected.astype(np.int64))
