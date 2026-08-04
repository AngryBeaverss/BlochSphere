"""Backend selection (CuPy-first with NumPy fallback).

The existing prototypes use the convention:
- try to import cupy as xp
- else import numpy as xp

We preserve that style (see blocksorder_gpu_v3.py). fileciteturn8file14L18-L28

This module adds:
- an explicit backend override (cpu/gpu/auto)
- small helpers to move arrays between host/device.

Note: "CPU fallback produces identical results" is handled by writing algorithms
that use the same math and dtypes on both backends for the small-N regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as _np
import sys
sys.set_int_max_str_digits(0)


BackendName = Literal["auto", "cpu", "gpu"]


@dataclass(frozen=True)
class Backend:
    xp: Any
    on_gpu: bool

    def to_cpu(self, a):
        if self.on_gpu:
            return self.xp.asnumpy(a)
        return a

    def to_gpu(self, a):
        if self.on_gpu:
            return self.xp.asarray(a)
        return a


def get_backend(request: BackendName = "auto") -> Backend:
    """Return an (xp, on_gpu) pair.

    request:
      - auto: use CuPy if available else NumPy
      - gpu : require CuPy
      - cpu : force NumPy
    """
    request = str(request).lower()

    if request == "cpu":
        import numpy as xp  # noqa: F401
        return Backend(xp=xp, on_gpu=False)

    try:
        if request in ("auto", "gpu"):
            import cupy as xp  # type: ignore
            return Backend(xp=xp, on_gpu=True)
    except Exception:
        if request == "gpu":
            raise RuntimeError("backend=gpu requested but CuPy is not available")

    import numpy as xp  # noqa: F401
    return Backend(xp=xp, on_gpu=False)


def ensure_float_dtype(xp, float64: bool):
    return xp.float64 if float64 else xp.float32


def rand_like(backend: Backend, shape, *, dtype, seed: int | None):
    """Reproducible uniform randoms on CPU and GPU.

    Mirrors blocksorder_gpu_v3's intent (RandomState on GPU, Generator on CPU). fileciteturn8file14L42-L51
    """
    xp = backend.xp
    if seed is None:
        return xp.random.random(shape, dtype=dtype) if backend.on_gpu else backend.to_gpu(_np.random.rand(*shape).astype(dtype))

    if backend.on_gpu:
        rs = xp.random.RandomState(int(seed))
        return rs.random_sample(shape, dtype=dtype)

    rng = _np.random.default_rng(int(seed))
    return backend.to_gpu(rng.random(shape, dtype=_np.float32).astype(dtype))


def rand_normal_like(backend: Backend, shape, *, dtype, seed: int | None):
    """Reproducible normal randoms on CPU and GPU.

    Mirrors blocksorder_gpu_v3's intent (RandomState.normal on GPU, Generator.normal on CPU). fileciteturn8file14L54-L66
    """
    xp = backend.xp
    if seed is None:
        return xp.random.normal(size=shape).astype(dtype) if backend.on_gpu else backend.to_gpu(_np.random.normal(size=shape).astype(dtype))

    if backend.on_gpu:
        rs = xp.random.RandomState(int(seed))
        return rs.normal(size=shape).astype(dtype)

    rng = _np.random.default_rng(int(seed))
    return backend.to_gpu(rng.normal(size=shape).astype(dtype))
