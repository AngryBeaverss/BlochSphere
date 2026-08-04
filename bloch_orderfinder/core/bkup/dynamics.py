"""Dynamics: optional LLG evolution with coherent neighbor overlap.

We preserve the coherent effective field construction:
- psi emitter = spin_to_psi1(S)
- neighbor complex overlap psi_n = neighbor_sum_complex(psi)
- transverse field from real/imag parts scaled by (J * psi_scale)
- optional longitudinal coupling Jz on Sz
as implemented in blocksorder_gpu_v3.py and blocksorder_gpu_v3_coherent_gui.py. fileciteturn9file3L15-L46 fileciteturn9file7L37-L60

LLG integration: Heun (predictor-corrector) with coherent flag plumbed through,
matching blocksorder_gpu_v3_coherent_gui.py. fileciteturn9file1L10-L24
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .backend import Backend
from .math_utils import cross3, normalize, neighbor_sum_4


def spin_to_psi1(backend: Backend, S, eps=1e-12):
    """Complex amplitude-like emitter (|1> amplitude) used by coherent field.

    Source: blocksorder_gpu_v3.py / v3_coherent_gui. fileciteturn9file3L48-L54
    """
    xp = backend.xp
    Sx = S[..., 0]
    Sy = S[..., 1]
    Sz = xp.clip(S[..., 2], -1.0, 1.0)
    phi = xp.arctan2(Sy, Sx)
    amp = xp.sqrt(xp.maximum(0.5 * (1.0 - Sz), 0.0))
    return amp.astype(xp.complex64) * xp.exp(1j * phi).astype(xp.complex64)


def neighbor_sum_complex(backend: Backend, Z):
    xp = backend.xp
    return xp.roll(Z, 1, axis=0) + xp.roll(Z, -1, axis=0) + xp.roll(Z, 1, axis=1) + xp.roll(Z, -1, axis=1)


def effective_field(
    backend: Backend,
    S,
    *,
    J: float = 1.0,
    Bext: Tuple[float, float, float] = (0.0, 0.0, 0.1),
    coherent: bool = False,
    psi_scale: float = 1.0,
    Jz: float | None = None,
):
    xp = backend.xp
    if not coherent:
        H = J * neighbor_sum_4(xp, S)
    else:
        if Jz is None:
            Jz = J

        psi = spin_to_psi1(backend, S)
        psi_n = neighbor_sum_complex(backend, psi)

        H = xp.zeros_like(S)
        H[..., 0] = (J * psi_scale) * xp.real(psi_n)
        H[..., 1] = (J * psi_scale) * xp.imag(psi_n)

        Sz = S[..., 2:3]
        Sz_n = xp.roll(Sz, 1, axis=0) + xp.roll(Sz, -1, axis=0) + xp.roll(Sz, 1, axis=1) + xp.roll(Sz, -1, axis=1)
        H[..., 2] = (Jz * Sz_n[..., 0])

    H[..., 0] += Bext[0]
    H[..., 1] += Bext[1]
    H[..., 2] += Bext[2]
    return H


def llg_rhs(
    backend: Backend,
    S,
    *,
    J: float = 1.0,
    Bext: Tuple[float, float, float] = (0.0, 0.0, 0.1),
    alpha: float = 0.05,
    gamma: float = 1.0,
    coherent: bool = False,
    psi_scale: float = 1.0,
    Jz: float | None = None,
):
    xp = backend.xp
    H = effective_field(backend, S, J=J, Bext=Bext, coherent=coherent, psi_scale=psi_scale, Jz=Jz)
    SxH = cross3(xp, S, H)
    SxSxH = cross3(xp, S, SxH)
    return -gamma * SxH + alpha * gamma * SxSxH


def llg_heun(
    backend: Backend,
    S,
    *,
    steps: int = 1,
    J: float = 1.0,
    Bext: Tuple[float, float, float] = (0.0, 0.0, 0.1),
    alpha: float = 0.05,
    gamma: float = 1.0,
    dt: float = 0.01,
    substeps: int = 10,
    coherent: bool = False,
    psi_scale: float = 1.0,
    Jz: float | None = None,
):
    """Heun (predictor-corrector) integrator for LLG.

    Coherent flag is threaded through, per v3_coherent_gui. fileciteturn9file1L10-L24
    """
    xp = backend.xp
    for _ in range(int(steps)):
        for _ in range(int(substeps)):
            k1 = llg_rhs(backend, S, J=J, Bext=Bext, alpha=alpha, gamma=gamma, coherent=coherent, psi_scale=psi_scale, Jz=Jz)
            S1 = normalize(xp, S + dt * k1)
            k2 = llg_rhs(backend, S1, J=J, Bext=Bext, alpha=alpha, gamma=gamma, coherent=coherent, psi_scale=psi_scale, Jz=Jz)
            S = normalize(xp, S + 0.5 * dt * (k1 + k2))
    return S


@dataclass(frozen=True)
class Diagnostics:
    Q: float
    E: float
    alignment: float
    Mx: float
    My: float
    Mz: float


def compute_diagnostics(backend: Backend, S, *, J: float = 1.0) -> Diagnostics:
    """A minimal subset of the v3 diagnostics for logging/debugging."""
    xp = backend.xp
    Sn = neighbor_sum_4(xp, S)
    dots = xp.sum(S * Sn, axis=-1)
    alignment = float(backend.to_cpu(xp.mean(dots) / 4.0))
    E = float(backend.to_cpu(-0.5 * J * xp.mean(xp.sum(S * Sn, axis=-1))))
    M = backend.to_cpu(xp.mean(S, axis=(0, 1)))
    # Skyrmion charge is optional and expensive; keep 0 here unless you need it.
    return Diagnostics(Q=0.0, E=E, alignment=alignment, Mx=float(M[0]), My=float(M[1]), Mz=float(M[2]))
