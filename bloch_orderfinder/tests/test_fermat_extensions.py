import numpy as np

from bloch_orderfinder.core.backend import get_backend
from bloch_orderfinder.core.encoder import ExponentMapping, init_spins
from bloch_orderfinder.core.modexp import modexp_grid_affine
from bloch_orderfinder.core.dynamics import llg_heun


def test_rectangular_exponent_mapping():
    backend = get_backend("cpu")
    mapping = ExponentMapping(size=6, height=3, row_stride=10, origin=7)
    grid = backend.to_cpu(mapping.grid(backend))
    expected = np.array(
        [[7 + y * 10 + x for x in range(6)] for y in range(3)],
        dtype=np.uint64,
    )
    assert grid.shape == (3, 6)
    assert np.array_equal(grid, expected)


def test_big_integer_modulus_is_projected_exactly():
    backend = get_backend("cpu")
    N = (1 << 130) + 51
    a = 17
    reduce_mod = 8192
    width = 9
    height = 4
    stride = 13
    origin = 5

    projected = modexp_grid_affine(
        backend,
        size=width,
        height=height,
        row_stride=stride,
        origin=origin,
        a=a,
        modN=N,
        reduce_mod=reduce_mod,
    )
    expected = np.array(
        [
            [pow(a, origin + y * stride + x, N) % reduce_mod for x in range(width)]
            for y in range(height)
        ],
        dtype=np.uint64,
    )
    assert projected.shape == (height, width)
    assert np.array_equal(projected, expected)


def test_period_width_produces_exact_repeated_rows():
    backend = get_backend("cpu")
    # ord_91(3) = 6, so a width/row_stride of 6 closes exactly.
    mapping = ExponentMapping(size=6, height=4, row_stride=6)
    S = init_spins(
        backend,
        mapping,
        pattern="modphase",
        mod_N=91,
        mod_a=3,
        reduce_mod=64,
    )
    S = backend.to_cpu(S)
    assert S.shape == (4, 6, 3)
    assert np.allclose(S[0], S[1], atol=1e-7)
    assert np.allclose(S[1], S[2], atol=1e-7)
    assert np.allclose(S[2], S[3], atol=1e-7)

    evolved = llg_heun(
        backend,
        backend.to_gpu(S),
        steps=2,
        substeps=2,
    )
    evolved = backend.to_cpu(evolved)
    assert np.allclose(evolved[0], evolved[1], atol=1e-6)
    assert np.allclose(evolved[1], evolved[2], atol=1e-6)
    assert np.allclose(evolved[2], evolved[3], atol=1e-6)
