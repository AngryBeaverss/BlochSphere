"""Encoder: exponent grid → residues → Bloch initialization.

Required patterns (preserved from prototypes):
- modval/modbin/modphase from blocksorder_gpu.py
- modmask from blocksorder_gpu_v3_coherent_gui.py

Robustness changes:
- add `row_stride` for exponent mapping: exponent(y,x)=origin + y*row_stride + x
  Default row_stride == size (backward compatible with v3's exp_arr=y*N+x).

New (non-order-finding) functionality:
- optional residue-grid caching to disk (NPZ): load residues if present, else compute
  and optionally save. This supports "modexp now, analyze later" workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import sys
sys.set_int_max_str_digits(0)

import os
import numpy as np

from .backend import Backend, rand_normal_like
from .modexp import (
    modexp_grid_affine,
    build_residue_meta,
    save_residues_npz,
    load_residues_npz,
)
from .math_utils import normalize


EncodingMode = Literal["random", "up", "hsv", "modval", "modbin", "modphase", "modmask"]


@dataclass(frozen=True)
class ExponentMapping:
    """Maps image coordinates to modular-exponentiation exponents.

    ``size`` remains the width for backward compatibility.  ``height`` defaults
    to the same value, preserving the original square lattice.
    """

    size: int
    height: int | None = None
    row_stride: int | None = None
    origin: int = 0

    @property
    def width(self) -> int:
        return int(self.size)

    @property
    def rows(self) -> int:
        return int(self.size if self.height is None else self.height)

    def grid(self, backend: Backend):
        """Return a uint64 exponent grid shaped ``(height, width)``."""
        xp = backend.xp
        W = self.width
        H = self.rows
        rs = int(self.row_stride) if self.row_stride is not None else W
        y, x = xp.meshgrid(
            xp.arange(H, dtype=xp.uint64),
            xp.arange(W, dtype=xp.uint64),
            indexing="ij",
        )
        return (self.origin + y * rs + x).astype(xp.uint64)


def _maybe_load_or_compute_residues(
    backend: Backend,
    *,
    cache_path: str | None,
    cache_load: bool,
    cache_save: bool,
    cache_overwrite: bool,
    cache_strict_meta: bool,
    cache_verify_sha1: bool,
    cache_include_sha1: bool,
    size: int,
    height: int,
    row_stride: int,
    origin: int,
    a: int,
    modN: int,
    reduce_mod: int | None,
    pattern: str | None,
):
    expected = build_residue_meta(
        modN=modN,
        a=a,
        size=size,
        height=height,
        row_stride=row_stride,
        origin=origin,
        reduce_mod=reduce_mod,
        pattern=pattern,
    )

    if cache_path and cache_load and os.path.exists(cache_path):
        residues_cpu, _meta = load_residues_npz(
            cache_path,
            expected_meta=expected,
            strict=cache_strict_meta,
            verify_sha1=cache_verify_sha1,
        )
        return backend.to_gpu(residues_cpu) if backend.on_gpu else residues_cpu

    # Compute fresh.
    residues = modexp_grid_affine(
        backend,
        size=int(size),
        height=int(height),
        row_stride=int(row_stride),
        origin=int(origin),
        a=int(a),
        modN=int(modN),
        reduce_mod=(int(reduce_mod) if reduce_mod is not None else None),
    )

    if cache_path and cache_save:
        # Always save CPU bytes (portable) even if backend is GPU.
        residues_cpu = backend.to_cpu(residues)
        save_residues_npz(
            cache_path,
            residues_cpu,
            meta=expected,
            include_sha1=cache_include_sha1,
            overwrite=cache_overwrite,
        )

    return residues


def init_spins(
    backend: Backend,
    mapping: ExponentMapping,
    *,
    pattern: EncodingMode = "random",
    seed: int = 42,
    mod_N: int | None = None,
    mod_a: int | None = None,
    reduce_mod: int | None = None,
    modmask: str | None = None,
    dtype=None,
    # Residue cache controls (non-order-finding infrastructure)
    residues_cache_path: str | None = None,
    residues_cache_load: bool = True,
    residues_cache_save: bool = False,
    residues_cache_overwrite: bool = False,
    residues_cache_strict_meta: bool = True,
    residues_cache_verify_sha1: bool = False,
    residues_cache_include_sha1: bool = False,
):
    """Initialize spin lattice ``S[height,width,3]`` on the Bloch sphere.

    For modular patterns, residues are computed via the affine recurrence.
    If residues_cache_path is provided, we can load/save residues to disk.

    Notes
    -----
    - With ``reduce_mod``, the exact big-integer recurrence is projected while the
      grid is generated, so the stored image remains uint64 even when ``mod_N``
      is thousands of bits long.
    """

    xp = backend.xp
    W = mapping.width
    H = mapping.rows
    dtype = xp.float32 if dtype is None else dtype

    S = xp.zeros((H, W, 3), dtype=dtype)

    if pattern == "random":
        rng = np.random.default_rng(seed)
        phi = backend.to_gpu(rng.random((H, W)).astype(np.float32)) * (2 * xp.pi)
        cost = backend.to_gpu(np.random.default_rng(seed + 1).random((H, W)).astype(np.float32)) * 2 - 1
        sint = xp.sqrt(xp.maximum(1 - cost * cost, 0))
        S[..., 0] = sint * xp.cos(phi)
        S[..., 1] = sint * xp.sin(phi)
        S[..., 2] = cost

    elif pattern == "up":
        S[..., 2] = 1.0

    elif pattern == "hsv":
        y, x = xp.meshgrid(
            xp.arange(H, dtype=dtype), xp.arange(W, dtype=dtype), indexing="ij"
        )
        phi = 2 * xp.pi * x / W
        theta = xp.pi * y / max(H, 1)
        S[..., 0] = xp.sin(theta) * xp.cos(phi)
        S[..., 1] = xp.sin(theta) * xp.sin(phi)
        S[..., 2] = xp.cos(theta)

    elif pattern in ("modval", "modbin", "modphase", "modmask"):
        if mod_N is None or mod_a is None:
            raise ValueError(f"Pattern '{pattern}' requires mod_N and mod_a")

        rs = int(mapping.row_stride) if mapping.row_stride is not None else int(W)

        # Compute/load residues once.
        vals_raw = _maybe_load_or_compute_residues(
            backend,
            cache_path=residues_cache_path,
            cache_load=residues_cache_load,
            cache_save=residues_cache_save,
            cache_overwrite=residues_cache_overwrite,
            cache_strict_meta=residues_cache_strict_meta,
            cache_verify_sha1=residues_cache_verify_sha1,
            cache_include_sha1=residues_cache_include_sha1,
            size=int(W),
            height=int(H),
            row_stride=rs,
            origin=int(mapping.origin),
            a=int(mod_a),
            modN=int(mod_N),
            reduce_mod=reduce_mod,
            pattern=pattern,
        )

        if pattern in ("modval", "modbin", "modphase"):
            if reduce_mod is not None:
                vals_eff = vals_raw % int(reduce_mod)
                effective_mod = int(reduce_mod)
            else:
                vals_eff = vals_raw
                effective_mod = int(mod_N)

            vals_float = vals_eff.astype(dtype)

            if pattern == "modval":
                Sz = 2.0 * (vals_float / effective_mod) - 1.0
                S[..., 2] = Sz
                S[..., 0] = rand_normal_like(backend, (H, W), dtype=dtype, seed=seed) * 0.1
                S[..., 1] = rand_normal_like(backend, (H, W), dtype=dtype, seed=(seed + 1 if seed is not None else None)) * 0.1

            elif pattern == "modbin":
                S[..., 2] = xp.where(vals_eff == 1, 1.0, -1.0).astype(dtype)
                S[..., 0] = rand_normal_like(backend, (H, W), dtype=dtype, seed=seed) * 0.1
                S[..., 1] = rand_normal_like(backend, (H, W), dtype=dtype, seed=(seed + 1 if seed is not None else None)) * 0.1

            elif pattern == "modphase":
                theta = xp.pi * (vals_float / effective_mod)
                phi = 2 * xp.pi * (vals_float % 1000) / 1000.0
                S[..., 0] = xp.sin(theta) * xp.cos(phi)
                S[..., 1] = xp.sin(theta) * xp.sin(phi)
                S[..., 2] = xp.cos(theta)

        elif pattern == "modmask":
            # Membership mask encoding.
            if modmask:
                residues = {int(r.strip()) for r in modmask.split(",") if r.strip()}
            else:
                residues = set()
            mask_np = np.isin(backend.to_cpu(vals_raw), list(residues))
            mask = backend.to_gpu(mask_np.astype(np.float32))
            S[..., 2] = xp.where(mask > 0.5, 1.0, -1.0).astype(dtype)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return normalize(xp, S)
