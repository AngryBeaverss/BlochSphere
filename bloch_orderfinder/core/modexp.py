"""Modular exponentiation kernels.

Features
--------
1) Elementwise pow(a, e, N) for an exponent grid.
2) Fast residue grid for affine exponent mappings:
      e(y,x) = origin + y*row_stride + x
3) Optional residue caching to disk (NPZ) with lightweight metadata checks.

The GPU elementwise kernel uses a safe 128-bit multiply/mod reduction based on
__umul64hi, derived from the prototypes.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np

from .backend import Backend


# -------------------------------
# GPU elementwise kernel (CuPy)
# -------------------------------

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
        except Exception:
            # Graceful fallback: compute on CPU and move back to GPU.
            flat = backend.to_cpu(exponents).ravel()
            out = np.empty_like(flat, dtype=np.uint64)
            for i in range(flat.size):
                out[i] = np.uint64(pow(int(a), int(flat[i]), int(modN)))
            return backend.to_gpu(out.reshape(backend.to_cpu(exponents).shape))

    # CPU: Python pow loop.
    flat = np.asarray(exponents).ravel()
    out = np.empty_like(flat, dtype=np.uint64)
    for i in range(flat.size):
        out[i] = np.uint64(pow(int(a), int(flat[i]), int(modN)))
    return backend.to_gpu(out.reshape(np.asarray(exponents).shape))


def modexp_grid_affine(
    backend: Backend,
    *,
    size: int,
    height: int | None = None,
    row_stride: int,
    origin: int,
    a: int,
    modN: int,
    reduce_mod: int | None = None,
):
    """Fast residue grid for affine exponent mappings.

    Computes the exact big-integer recurrence

        v(y, x) = a^(origin + y*row_stride + x) mod modN

    and stores either ``v`` itself or ``v % reduce_mod``.  Supplying
    ``reduce_mod`` is what permits moduli larger than 64 bits: the modular
    recurrence remains exact in Python integers while the image stores only a
    compact uint64 projection.

    ``size`` is the image width.  ``height`` defaults to ``size`` for backward
    compatibility with the original square lattice.
    """

    width = int(size)
    height = width if height is None else int(height)
    row_stride = int(row_stride)
    origin = int(origin)

    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")

    modN = int(modN)
    if modN <= 1:
        raise ValueError(f"modN must be > 1, got {modN}")

    u64_max = int(np.iinfo(np.uint64).max)
    if reduce_mod is None:
        if modN > u64_max:
            raise ValueError(
                "modN exceeds uint64 storage; supply reduce_mod so the exact "
                "big-integer residues can be projected into the image"
            )
        storage_mod = None
    else:
        storage_mod = int(reduce_mod)
        if storage_mod <= 1:
            raise ValueError(f"reduce_mod must be > 1, got {storage_mod}")
        if storage_mod > u64_max:
            raise ValueError(
                f"reduce_mod must fit uint64 (<= {u64_max}), got {storage_mod}"
            )

    a = int(a) % modN

    # Two pow calls total, both on arbitrary-precision Python integers.
    row_step = pow(a, row_stride, modN)
    row_start = pow(a, origin, modN)

    out = np.empty((height, width), dtype=np.uint64)

    # Fill each row via recurrence: v_{x+1} = v_x * a (mod N).
    for y in range(height):
        v = row_start
        for x in range(width):
            stored = v if storage_mod is None else (v % storage_mod)
            out[y, x] = np.uint64(stored)
            v = (v * a) % modN
        # Next row's start: multiply by a^row_stride.
        row_start = (row_start * row_step) % modN

    return backend.to_gpu(out) if backend.on_gpu else out


# -------------------------------
# Residue caching (NPZ)
# -------------------------------

def build_residue_meta(
    *,
    modN: int,
    a: int,
    size: int,
    height: int | None = None,
    row_stride: int,
    origin: int,
    reduce_mod: int | None = None,
    pattern: str | None = None,
) -> Dict[str, int | str | None]:
    """Construct metadata describing the residue grid.

    Notes
    -----
    When ``reduce_mod`` is supplied, the cached uint64 grid contains the exact
    residues modulo ``modN`` projected through ``value % reduce_mod``.
    """
    return {
        "modN": int(modN),
        "a": int(a),
        "size": int(size),
        "height": int(size if height is None else height),
        "row_stride": int(row_stride),
        "origin": int(origin),
        "reduce_mod": (int(reduce_mod) if reduce_mod is not None else None),
        "pattern": pattern,
        "format": "u64_grid_v2_projected",
    }


def _sha1_u64_grid(arr_u64: np.ndarray) -> str:
    view = np.ascontiguousarray(arr_u64).view(np.uint8)
    h = hashlib.sha1()
    h.update(view)
    return h.hexdigest()


def save_residues_npz(
    path: str,
    residues_u64,
    *,
    meta: Dict[str, int | str | None],
    include_sha1: bool = False,
    overwrite: bool = False,
) -> str:
    """Save a residue grid to NPZ.

    Parameters
    ----------
    residues_u64:
        Array-like residue grid. Will be saved as uint64 on disk.
    meta:
        Metadata dict (use build_residue_meta).
    include_sha1:
        If True, compute SHA1 of the raw u64 bytes and store as "sha1".
        This is O(size^2) in bytes; keep it off for very large grids if you care.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing residue cache: {path}")

    arr = np.asarray(residues_u64, dtype=np.uint64)
    meta2 = dict(meta)
    if include_sha1:
        meta2["sha1"] = _sha1_u64_grid(arr)

    meta_json = json.dumps(meta2, sort_keys=True)
    np.savez_compressed(path, residues=arr, meta=np.string_(meta_json))
    return path


def load_residues_npz(
    path: str,
    *,
    expected_meta: Dict[str, int | str | None] | None = None,
    strict: bool = True,
    verify_sha1: bool = False,
) -> Tuple[np.ndarray, Dict[str, int | str | None]]:
    """Load a residue grid from NPZ.

    If expected_meta is provided, we compare overlapping keys.
    - strict=True: raise on mismatch.
    - strict=False: return the data and meta anyway.

    If verify_sha1=True and the file contains a sha1 in meta, it is verified.
    """
    with np.load(path, allow_pickle=False) as z:
        residues = z["residues"].astype(np.uint64, copy=False)
        meta_json = str(z["meta"].item())

    meta = json.loads(meta_json)

    if expected_meta is not None:
        mismatches = []
        for k, v in expected_meta.items():
            if k in meta and meta[k] != v:
                mismatches.append((k, meta[k], v))
        if mismatches and strict:
            msg = "; ".join([f"{k}: file={a} expected={b}" for k, a, b in mismatches])
            raise ValueError(f"Residue cache metadata mismatch for {path}: {msg}")

    if verify_sha1 and ("sha1" in meta):
        actual = _sha1_u64_grid(residues)
        if actual != meta["sha1"]:
            raise ValueError(f"Residue cache sha1 mismatch for {path}: file={meta['sha1']} actual={actual}")

    return residues, meta
