import numpy as np

from bloch_orderfinder.core.backend import get_backend
from bloch_orderfinder.core.scoring import scan_aligned_difference


def test_aligned_difference_recovers_period_on_repeated_complex_blocks():
    rng = np.random.default_rng(0)
    r = 37
    n = 5000
    block = rng.standard_normal(r) + 1j * rng.standard_normal(r)
    x = np.tile(block, int(np.ceil(n / r)))[:n]
    x = x * np.exp(1j * 0.3) + 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    backend = get_backend('cpu')
    x_xp = backend.to_gpu(x.astype(np.complex64))

    lag, diag = scan_aligned_difference(
        backend,
        x_xp,
        min_lag=2,
        max_lag=200,
        coarse_step=5,
        refine_half_window=20,
        window=2000,
    )
    assert abs(int(lag) - r) <= 1
