"""Encoder: exponent grid → residues → Bloch initialization.

Required patterns (preserved from prototypes):
- modval/modbin/modphase from blocksorder_gpu.py fileciteturn8file1L32-L58
- modmask from blocksorder_gpu_v3_coherent_gui.py fileciteturn8file8L30-L60

Key robustness change:
- add `row_stride` for exponent mapping: exponent(y,x)=origin + y*row_stride + x
  Default row_stride == size (backward compatible with v3's exp_arr=y*N+x). fileciteturn8file8L33-L35
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .backend import Backend, rand_normal_like
from .modexp import modexp_grid_affine
from .math_utils import normalize


EncodingMode = Literal["random", "up", "hsv", "modval", "modbin", "modphase", "modmask"]


@dataclass(frozen=True)
class ExponentMapping:
    """Maps (row, col) -> exponent for modular exponentiation sampling."""
    size: int
    row_stride: int | None = None
    origin: int = 0

    def grid(self, backend: Backend):
        """Return uint64 exponent grid shaped (size,size) on the active backend."""
        xp = backend.xp
        N = int(self.size)
        rs = int(self.row_stride) if self.row_stride is not None else N
        y, x = xp.meshgrid(xp.arange(N, dtype=xp.uint64), xp.arange(N, dtype=xp.uint64), indexing="ij")
        return (self.origin + y * rs + x).astype(xp.uint64)


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
):
    """Initialize spin lattice S[size,size,3] on Bloch sphere.

    For modular patterns, we compute residues efficiently by exploiting the
    affine exponent mapping. On GPU backends, residues are computed on CPU and
    uploaded once (often faster than per-element GPU modular exponentiation).
    """
    xp = backend.xp
    N = mapping.size
    dtype = xp.float32 if dtype is None else dtype

    S = xp.zeros((N, N, 3), dtype=dtype)

    if pattern == "random":
        # Sphere sampling via phi + cos(theta), as in v3. fileciteturn8file8L11-L17
        rng = np.random.default_rng(seed)
        phi = backend.to_gpu(rng.random((N, N)).astype(np.float32)) * (2 * xp.pi)
        cost = backend.to_gpu(np.random.default_rng(seed + 1).random((N, N)).astype(np.float32)) * 2 - 1
        sint = xp.sqrt(xp.maximum(1 - cost * cost, 0))
        S[..., 0] = sint * xp.cos(phi)
        S[..., 1] = sint * xp.sin(phi)
        S[..., 2] = cost

    elif pattern == "up":
        S[..., 2] = 1.0

    elif pattern == "hsv":
        # Smooth test field, from v3. fileciteturn8file8L22-L28
        y, x = xp.meshgrid(xp.arange(N, dtype=dtype), xp.arange(N, dtype=dtype), indexing="ij")
        phi = 2 * xp.pi * x / N
        theta = xp.pi * y / N
        S[..., 0] = xp.sin(theta) * xp.cos(phi)
        S[..., 1] = xp.sin(theta) * xp.sin(phi)
        S[..., 2] = xp.cos(theta)

    elif pattern in ("modval", "modbin", "modphase", "modmask"):
        if mod_N is None or mod_a is None:
            raise ValueError(f"Pattern '{pattern}' requires mod_N and mod_a")

        # Exponent mapping is affine: e(y,x)=origin + y*row_stride + x.
        # We can exploit that to avoid per-element pow().
        rs = int(mapping.row_stride) if mapping.row_stride is not None else int(N)

        if pattern in ("modval", "modbin", "modphase"):
            vals_raw = modexp_grid_affine(
                backend,
                size=int(N),
                row_stride=rs,
                origin=int(mapping.origin),
                a=int(mod_a),
                modN=int(mod_N),
            )
            if reduce_mod is not None:
                vals_raw = vals_raw % int(reduce_mod)
                effective_mod = int(reduce_mod)
            else:
                effective_mod = int(mod_N)
            vals_float = vals_raw.astype(dtype)

            if pattern == "modval":
                # Map residue [0, effective_mod) → Sz [-1, 1]. fileciteturn8file1L32-L38
                Sz = 2.0 * (vals_float / effective_mod) - 1.0
                S[..., 2] = Sz
                # Small random xy for dynamics. fileciteturn8file1L36-L39
                S[..., 0] = rand_normal_like(backend, (N, N), dtype=dtype, seed=seed) * 0.1
                S[..., 1] = rand_normal_like(backend, (N, N), dtype=dtype, seed=(seed + 1 if seed is not None else None)) * 0.1

            elif pattern == "modbin":
                # +1 where a^x ≡ 1 (mod N) after optional reduce_mod; else -1. fileciteturn8file1L40-L48
                S[..., 2] = xp.where(vals_raw == 1, 1.0, -1.0).astype(dtype)
                S[..., 0] = rand_normal_like(backend, (N, N), dtype=dtype, seed=seed) * 0.1
                S[..., 1] = rand_normal_like(backend, (N, N), dtype=dtype, seed=(seed + 1 if seed is not None else None)) * 0.1

            elif pattern == "modphase":
                # Full 3D Bloch encoding. fileciteturn8file1L50-L58
                theta = xp.pi * (vals_float / effective_mod)
                phi = 2 * xp.pi * (vals_float % 1000) / 1000.0
                S[..., 0] = xp.sin(theta) * xp.cos(phi)
                S[..., 1] = xp.sin(theta) * xp.sin(phi)
                S[..., 2] = xp.cos(theta)

        elif pattern == "modmask":
            # Membership mask encoding from v3 (no reduce_mod in the original). fileciteturn8file8L52-L60
            raw = modexp_grid_affine(
                backend,
                size=int(N),
                row_stride=rs,
                origin=int(mapping.origin),
                a=int(mod_a),
                modN=int(mod_N),
            )
            if modmask:
                residues = {int(r.strip()) for r in modmask.split(",") if r.strip()}
            else:
                residues = set()
            mask_np = np.isin(backend.to_cpu(raw), list(residues))
            mask = backend.to_gpu(mask_np.astype(np.float32))
            S[..., 2] = xp.where(mask > 0.5, 1.0, -1.0).astype(dtype)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return normalize(xp, S)