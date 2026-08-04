"""Shared math primitives used across encoder/dynamics/scoring."""

from __future__ import annotations


def normalize(xp, S, eps=1e-12):
    """Normalize last axis to unit length."""
    nrm = xp.sqrt(xp.maximum(xp.sum(S * S, axis=-1, keepdims=True), eps))
    return S / nrm


def cross3(xp, A, B):
    """Vector cross product along last axis."""
    cx = A[..., 1] * B[..., 2] - A[..., 2] * B[..., 1]
    cy = A[..., 2] * B[..., 0] - A[..., 0] * B[..., 2]
    cz = A[..., 0] * B[..., 1] - A[..., 1] * B[..., 0]
    return xp.stack([cx, cy, cz], axis=-1)


def dot3(xp, A, B):
    """Dot product along last axis."""
    return xp.sum(A * B, axis=-1)


def neighbor_sum_4(xp, S):
    """4-neighbor sum with periodic boundary conditions."""
    return (
        xp.roll(S, 1, axis=0)
        + xp.roll(S, -1, axis=0)
        + xp.roll(S, 1, axis=1)
        + xp.roll(S, -1, axis=1)
    )
