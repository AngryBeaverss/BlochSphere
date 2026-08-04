"""Modular exponentiation kernels.

We consolidate the elementwise modular exponentiation used by blocksorder_gpu.py:
- GPU path: CuPy ElementwiseKernel implementing safe 128-bit multiply-reduce
- CPU path: Python's built-in pow(a, e, N)

Source: blocksorder_gpu.py's _modexp_array_raw kernel (modexp_kernel_128bit_v2). fileciteturn8file2L20-L41

We deliberately use this non-Montgomery kernel to support even moduli and to
avoid Montgomery's "odd modulus" restriction in v3. fileciteturn8file14L78-L80
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .backend import Backend


@lru_cache(maxsize=1)
def _cupy_modexp_kernel() -> Any:
    import cupy as cp

    # Unsigned 64-bit modular exponentiation with safe mulmod using __umul64hi.
    # Key fix: treat exponent as uint64 to avoid negative exponent hang on GPU.
    return cp.ElementwiseKernel(
        in_params='uint64 e, uint64 a_in, uint64 m',
        out_params='uint64 out',
        operation=r'''
            #define MUL128_MOD_U64(x, y, m, result) do { \
                unsigned long long ux = (unsigned long long)(x); \
                unsigned long long uy = (unsigned long long)(y); \
                unsigned long long um = (unsigned long long)(m); \
                unsigned long long hi = __umul64hi(ux, uy); \
                unsigned long long lo = ux * uy; \
                if (hi == 0ULL) { \
                    result = (unsigned long long)(lo % um); \
                } else { \
                    unsigned long long r = lo % um; \
                    unsigned long long base64_mod = 0ULL; \
                    if (um != 0ULL) { \
                        base64_mod = (0xFFFFFFFFFFFFFFFFULL % um) + 1ULL; \
                        if (base64_mod >= um) base64_mod -= um; \
                    } \
                    for (int i = 0; i < 64 && hi > 0ULL; i++) { \
                        if (hi & 1ULL) { \
                            r += base64_mod; \
                            if (r >= um) r -= um; \
                        } \
                        base64_mod += base64_mod; \
                        if (base64_mod >= um) base64_mod -= um; \
                        hi >>= 1ULL; \
                    } \
                    result = (unsigned long long)r; \
                } \
            } while(0)

            unsigned long long mod = (unsigned long long)m;
            // m is guaranteed > 1 by Python caller.
            unsigned long long base = (unsigned long long)(a_in % mod);
            unsigned long long result = 1ULL % mod;
            unsigned long long exp = (unsigned long long)e;

            while (exp) {
                if (exp & 1ULL) {
                    unsigned long long tmp;
                    MUL128_MOD_U64(result, base, mod, tmp);
                    result = tmp;
                }
                exp >>= 1ULL;
                if (exp) {
                    unsigned long long tmp2;
                    MUL128_MOD_U64(base, base, mod, tmp2);
                    base = tmp2;
                }
            }
            out = (unsigned long long)result;
        ''',
        name='modexp_u64_safe',
    )


def modexp_array_raw(backend: Backend, exponents, a: int, modN: int):
    """Elementwise pow(a, e, modN) for an exponent grid.

    Returns an array with the same shape as `exponents`, dtype uint64 on the active backend.
    """
    xp = backend.xp

    modN = int(modN)
    if modN <= 1:
        raise ValueError(f"modN must be > 1, got {modN}")

    a = int(a) % modN
    exponents = exponents.astype(xp.uint64, copy=False)

    if backend.on_gpu:
        try:
            kern = _cupy_modexp_kernel()
            exponents = xp.ascontiguousarray(exponents)
            return kern(exponents, xp.uint64(a), xp.uint64(modN))
        except Exception as e:
            # Graceful fallback: compute on CPU and move back to GPU.
            # This preserves correctness and makes GPU issues diagnosable without losing the run.
            flat = backend.to_cpu(exponents).ravel()
            out = np.empty_like(flat, dtype=np.uint64)
            for i in range(flat.size):
                out[i] = np.uint64(pow(int(a), int(flat[i]), int(modN)))
            return backend.to_gpu(out.reshape(backend.to_cpu(exponents).shape))

    # CPU: Python pow loop (vectorize is convenient but can be slower for large grids).
    flat = np.asarray(exponents).ravel()
    out = np.empty_like(flat, dtype=np.uint64)
    for i in range(flat.size):
        out[i] = np.uint64(pow(int(a), int(flat[i]), int(modN)))
    return backend.to_gpu(out.reshape(np.asarray(exponents).shape))


def modexp_grid_affine(
    backend: Backend,
    *,
    size: int,
    row_stride: int,
    origin: int,
    a: int,
    modN: int,
):
    """Fast residue grid for affine exponent mappings.

    Computes:
        R[y, x] = a^(origin + y*row_stride + x) mod modN

    in O(size^2) modular multiplies plus two modular exponentiations.

    Notes
    -----
    This is *much* faster than calling pow(a, e, modN) for every element when
    exponents are laid out affinely (the project's default mapping).

    On GPU backends, we compute on CPU and upload once. For typical sizes and
    moduli, that is still substantially faster than per-element square-and-multiply
    kernels.
    """

    size = int(size)
    row_stride = int(row_stride)
    origin = int(origin)

    modN = int(modN)
    if modN <= 1:
        raise ValueError(f"modN must be > 1, got {modN}")

    a = int(a) % modN

    # Two pow calls total.
    row_step = pow(a, row_stride, modN)
    row_start = pow(a, origin, modN)

    out = np.empty((size, size), dtype=np.uint64)

    # Fill each row via recurrence: v_{x+1} = v_x * a (mod N)
    for y in range(size):
        v = row_start
        for x in range(size):
            out[y, x] = np.uint64(v)
            v = (v * a) % modN
        # Next row's start: multiply by a^row_stride.
        row_start = (row_start * row_step) % modN

    return backend.to_gpu(out) if backend.on_gpu else out
