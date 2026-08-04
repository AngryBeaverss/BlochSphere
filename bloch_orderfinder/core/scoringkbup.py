"""Scoring: quantitative period score.

Primary score:
- normalized correlation on a 1D signature extracted from the Bloch lattice
- harmonic-aware post-processing (checks nearby multiples/divisors and prefers a smaller representative lag)

Fallback:
- alignment-invariant lag scan using min_c mean(|x[n] - c x[n+L]|^2),
  which is robust to complex phase rotation and amplitude scaling.
  Based on coprime_jacobi_2adic_fixed_periodviz.py. fileciteturn9file12L1-L6 fileciteturn9file9L11-L14
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .backend import Backend
import sys
sys.set_int_max_str_digits(0)


SeqMode = Literal["sz", "sx", "sy", "psi1"]


@dataclass(frozen=True)
class LagScore:
    lag: int
    score: float
    method: str
    detail: dict


@dataclass(frozen=True)
class ShiftScore:
    """2D torus-translation symmetry score.

    dx, dy are signed shifts in pixel coordinates (wrap-around torus).
    delta_x is the implied exponent shift under the mapping:
        exponent(y,x) = origin + y*row_stride + x
    so:
        delta_x = dx + dy*row_stride
    """

    dx: int
    dy: int
    score: float
    delta_x: int
    method: str
    detail: dict


def extract_sequence(backend: Backend, S, *, mode: SeqMode = "sz"):
    """Flatten the lattice row-major; exponent-contiguous only when row_stride == width."""
    xp = backend.xp
    if mode == "sz":
        x = S[..., 2]
    elif mode == "sx":
        x = S[..., 0]
    elif mode == "sy":
        x = S[..., 1]
    elif mode == "psi1":
        from .dynamics import spin_to_psi1
        x = spin_to_psi1(backend, S)
    else:
        raise ValueError(f"Unknown sequence mode: {mode}")
    return x.reshape(-1)


def extract_field2d(backend: Backend, S, *, mode: SeqMode = "psi1"):
    """Return a 2D field (size,size) suitable for image-native scoring.

    For psi1 we use the complex emitter used by coherent dynamics.
    For sx/sy/sz we cast the real channel to complex.
    """
    xp = backend.xp
    if mode == "sz":
        Z = S[..., 2]
        return Z.astype(xp.complex64)
    if mode == "sx":
        Z = S[..., 0]
        return Z.astype(xp.complex64)
    if mode == "sy":
        Z = S[..., 1]
        return Z.astype(xp.complex64)
    if mode == "psi1":
        from .dynamics import spin_to_psi1
        return spin_to_psi1(backend, S)
    raise ValueError(f"Unknown sequence mode: {mode}")


def scan_2d_torus_shifts_fft(
    backend: Backend,
    Z2d,
    *,
    row_stride: int,
    dx_max: int = -1,
    dy_max: int = -1,
    topk: int = 10,
    exclude_origin: bool = True,
    exclude_zero_delta: bool = True,
    min_abs_delta_x: int = 1,
    max_abs_delta_x: int | None = None,
):
    """Find strong 2D translation symmetries using FFT autocorrelation.

    We compute cyclic (torus) autocorrelation:
        C = ifft2(fft2(Z) * conj(fft2(Z)))
    and score shifts by normalized magnitude:
        score(dy,dx) = |C[dy,dx]| / sum(|Z|^2)

    IMPORTANT: when `row_stride != width`, the exponent mapping has a built-in
    exact symmetry for any shift with `delta_x = dx + dy*row_stride == 0`
    (it moves along iso-exponent diagonals). Those are not "order" and are
    excluded by default via `exclude_zero_delta=True`.

    Filtering:
      - if dx_max/dy_max >= 0, we only consider |dx|<=dx_max and |dy|<=dy_max
      - we can also bound |delta_x| via [min_abs_delta_x, max_abs_delta_x]

    Returns: top-k ShiftScore candidates.
    """
    xp = backend.xp
    H, W = int(Z2d.shape[0]), int(Z2d.shape[1])

    dx_max = int(dx_max)
    dy_max = int(dy_max)
    topk = int(max(1, topk))
    row_stride = int(row_stride)
    min_abs_delta_x = int(max(0, min_abs_delta_x))
    max_abs_delta_x = int(max_abs_delta_x) if max_abs_delta_x is not None else None

    # Energy normalization
    E = xp.sum(xp.abs(Z2d) ** 2).real + 1e-12

    # FFT autocorrelation (cyclic/torus)
    F = xp.fft.fft2(Z2d)
    C = xp.fft.ifft2(F * xp.conj(F))
    S = xp.abs(C) / E  # shape (H,W), real

    flat = S.reshape(-1)
    nflat = int(flat.shape[0])

    # Grab a generous pool of top candidates, then filter down.
    pool = max(topk * 300, 5000)
    pool = min(pool, nflat)

    cand: list[ShiftScore] = []
    used_delta = set()

    while True:
        if pool >= nflat:
            idx = xp.argsort(flat)[::-1]
        else:
            part = xp.argpartition(flat, nflat - pool)[nflat - pool :]
            idx = part[xp.argsort(flat[part])[::-1]]

        idx_cpu = backend.to_cpu(idx)

        for j in idx_cpu:
            j = int(j)
            iy = j // W
            ix = j - iy * W

            # signed minimal shift on the torus
            dy = iy if iy <= H // 2 else iy - H
            dx = ix if ix <= W // 2 else ix - W

            if exclude_origin and dx == 0 and dy == 0:
                continue
            if dx_max >= 0 and abs(dx) > dx_max:
                continue
            if dy_max >= 0 and abs(dy) > dy_max:
                continue

            delta_x = int(dx + dy * row_stride)

            # Critical: remove iso-exponent diagonal symmetries (delta_x==0)
            if exclude_zero_delta and delta_x == 0:
                continue
            if min_abs_delta_x and abs(delta_x) < min_abs_delta_x:
                continue
            if max_abs_delta_x is not None and abs(delta_x) > max_abs_delta_x:
                continue

            if delta_x in used_delta:
                continue
            used_delta.add(delta_x)

            score = float(backend.to_cpu(flat[j]))
            cand.append(
                ShiftScore(
                    dx=int(dx),
                    dy=int(dy),
                    score=float(score),
                    delta_x=int(delta_x),
                    method="fft2_autocorr",
                    detail={
                        "H": H,
                        "W": W,
                        "row_stride": row_stride,
                        "dx_max": dx_max,
                        "dy_max": dy_max,
                        "min_abs_delta_x": min_abs_delta_x,
                        "max_abs_delta_x": max_abs_delta_x,
                        "pool": int(pool),
                    },
                )
            )
            if len(cand) >= topk:
                break

        if len(cand) >= topk or pool >= nflat:
            break
        pool = min(nflat, pool * 2)

    cand.sort(key=lambda z: (-z.score, abs(z.delta_x), abs(z.dy), abs(z.dx)))
    return cand[:topk]


def scan_2d_staggered_shifts_fft(
    backend: Backend,
    Z2d,
    *,
    row_stride: int,
    dx_max: int = -1,
    dy_max: int = -1,
    topk: int = 10,
    exclude_zero_delta: bool = True,
    min_abs_delta_x: int = 1,
    max_abs_delta_x: int | None = None,
):
    """Score nonwrapped row-pair shifts with one uniform exponent displacement.

    This is the staggered companion to :func:`scan_2d_torus_shifts_fft`.
    It compares only overlapping, nonwrapped regions, so every compared pair for
    a candidate ``(dx, dy)`` has exactly the same exponent displacement::

        delta_x = dx + dy * row_stride

    No new controls are required.  The existing horizontal lag bounds are reused:
    ``min_abs_delta_x`` and ``max_abs_delta_x`` bound ``abs(dx)`` here, rather
    than the potentially enormous total displacement ``abs(delta_x)``.  ``dy``
    is limited by the existing ``dy_max`` setting (or all row separations when
    ``dy_max < 0``).

    The calculation uses zero-padded 1D FFT cross-correlation across each row
    pair, avoiding the horizontal and vertical wrap classes mixed by torus
    autocorrelation.
    """
    xp = backend.xp
    H, W = int(Z2d.shape[0]), int(Z2d.shape[1])
    if H < 2 or W < 2:
        return []

    row_stride = int(row_stride)
    topk = int(max(1, topk))

    max_dy = H - 1 if int(dy_max) < 0 else min(abs(int(dy_max)), H - 1)
    if max_dy < 1:
        return []

    # Require at least half-row horizontal overlap.  Without this, offsets
    # near +/- (W-1) are ranked from only one or two samples and normalized
    # correlation becomes trivially ~1 rather than statistically meaningful.
    horizontal_cap = W // 2
    if int(dx_max) >= 0:
        horizontal_cap = min(horizontal_cap, abs(int(dx_max)))
    if max_abs_delta_x is not None:
        horizontal_cap = min(horizontal_cap, abs(int(max_abs_delta_x)))

    min_abs_dx = max(0, int(min_abs_delta_x))
    if horizontal_cap < min_abs_dx:
        return []

    # Zero padding makes the FFT correlation linear rather than cyclic.
    nfft = 1 << max(1, (2 * W - 1).bit_length())
    candidates: list[ShiftScore] = []
    used_delta: set[int] = set()

    for dy in range(1, max_dy + 1):
        A = Z2d[: H - dy, :]
        B = Z2d[dy:, :]

        FA = xp.fft.fft(A, n=nfft, axis=1)
        FB = xp.fft.fft(B, n=nfft, axis=1)
        corr = xp.sum(xp.fft.ifft(FB * xp.conj(FA), axis=1), axis=0)

        ea_col = xp.sum(xp.abs(A) ** 2, axis=0).real
        eb_col = xp.sum(xp.abs(B) ** 2, axis=0).real
        ea_prefix = xp.cumsum(ea_col)
        eb_prefix = xp.cumsum(eb_col)
        ea_total = ea_prefix[-1]
        eb_total = eb_prefix[-1]

        dx_values = list(range(-horizontal_cap, horizontal_cap + 1))
        if min_abs_dx:
            dx_values = [dx for dx in dx_values if abs(dx) >= min_abs_dx]

        scores = []
        for dx in dx_values:
            if dx >= 0:
                overlap = W - dx
                if overlap <= 0:
                    continue
                num = corr[dx]
                ea = ea_prefix[overlap - 1]
                eb = eb_total if dx == 0 else (eb_total - eb_prefix[dx - 1])
            else:
                d = -dx
                overlap = W - d
                if overlap <= 0:
                    continue
                num = corr[nfft - d]
                ea = ea_total - ea_prefix[d - 1]
                eb = eb_prefix[overlap - 1]

            den = xp.sqrt(ea * eb) + 1e-12
            # Cauchy-Schwarz bounds this normalized score by 1.  Clamp only
            # tiny FFT/float32 roundoff excursions above that bound.
            score = min(1.0, float(backend.to_cpu(xp.abs(num) / den)))
            scores.append((score, dx, overlap))

        # Retain a small local pool from each row separation before global sort.
        scores.sort(key=lambda item: (-item[0], abs(item[1]), item[1]))
        for score, dx, overlap in scores[: max(topk * 4, topk)]:
            delta_x = int(dx + dy * row_stride)
            if exclude_zero_delta and delta_x == 0:
                continue

            for sx, sy, sd in ((dx, dy, delta_x), (-dx, -dy, -delta_x)):
                if sd in used_delta:
                    continue
                used_delta.add(sd)
                candidates.append(
                    ShiftScore(
                        dx=int(sx),
                        dy=int(sy),
                        score=float(score),
                        delta_x=int(sd),
                        method="fft_rowpair_nonwrapped",
                        detail={
                            "H": H,
                            "W": W,
                            "row_stride": row_stride,
                            "row_separation": int(abs(sy)),
                            "horizontal_offset": int(sx),
                            "overlap_rows": int(H - abs(sy)),
                            "overlap_cols": int(overlap),
                            "nfft": int(nfft),
                        },
                    )
                )

    candidates.sort(key=lambda z: (-z.score, abs(z.dy), abs(z.dx), abs(z.delta_x)))
    return candidates[:topk]

def _vdot(backend: Backend, a, b):
    xp = backend.xp
    return xp.vdot(a, b)


def corr_at_lag(backend: Backend, x, L: int, *, window: int | None = None) -> float:
    """Normalized correlation magnitude at lag L over a fixed window."""
    xp = backend.xp
    n = int(x.shape[0])
    L = int(L)
    if L <= 0 or L >= n - 2:
        return 0.0
    M = n - L if window is None else int(min(window, n - L))
    if M < 4:
        return 0.0
    x1 = x[:M]
    x2 = x[L:L + M]
    num = _vdot(backend, x1, x2)
    den = xp.sqrt(_vdot(backend, x1, x1).real * _vdot(backend, x2, x2).real) + 1e-12
    val = xp.abs(num) / den
    return float(backend.to_cpu(val))


def harmonic_scores(
    backend: Backend,
    x,
    L: int,
    *,
    max_harmonic: int = 6,
    window: int | None = None,
):
    """Return correlation scores for {L, L*k, L/k} within range."""
    n = int(x.shape[0])
    L = int(L)
    scores: dict[int, float] = {}
    if L <= 0:
        return scores

    # base + multiples
    for k in range(1, max_harmonic + 1):
        m = L * k
        if 1 < m < n - 1:
            scores[m] = corr_at_lag(backend, x, m, window=window)

    # divisors
    for k in range(2, max_harmonic + 1):
        if L % k == 0:
            d = L // k
            if 1 < d < n - 1:
                scores[d] = corr_at_lag(backend, x, d, window=window)

    return scores


def choose_representative_lag(scores: dict[int, float], *, tol: float = 0.02) -> tuple[int, float]:
    """Pick a smaller lag that is 'almost as good' as the best.

    If a multiple and its divisor are both strong, this biases toward the divisor
    (i.e., the more fundamental period) instead of reporting a giant multiple.
    """
    if not scores:
        return 0, 0.0
    best_lag = max(scores, key=lambda k: scores[k])
    best_score = float(scores[best_lag])
    # all lags within tol of best score
    near = [L for L, s in scores.items() if (best_score - float(s)) <= tol * max(1e-9, best_score)]
    rep = min(near) if near else best_lag
    rep_score = float(scores[rep])
    return int(rep), float(rep_score)


def scan_correlation(
    backend: Backend,
    x,
    *,
    min_lag: int,
    max_lag: int,
    step: int = 1,
    topk: int = 10,
    max_harmonic: int = 6,
    window: int | None = None,
    representative_tol: float = 0.02,
):
    """Scan lags and return top-k harmonic-aware candidates."""
    min_lag = int(min_lag)
    max_lag = int(max_lag)
    step = int(step)
    raw: list[LagScore] = []
    for L in range(min_lag, max_lag + 1, step):
        scores = harmonic_scores(backend, x, L, max_harmonic=max_harmonic, window=window)
        rep, rep_score = choose_representative_lag(scores, tol=representative_tol)
        if rep <= 1:
            continue
        raw.append(
            LagScore(
                lag=rep,
                score=rep_score,
                method="harmonic_corr",
                detail={"base": int(L), "scores": scores, "rep_tol": representative_tol},
            )
        )

    # keep best score per lag
    best_by_lag: dict[int, LagScore] = {}
    for item in raw:
        prev = best_by_lag.get(item.lag)
        if prev is None or item.score > prev.score:
            best_by_lag[item.lag] = item
    items = list(best_by_lag.values())
    items.sort(key=lambda z: (-z.score, z.lag))
    return items[: int(topk)]


def aligned_difference_cost(backend: Backend, x, L: int, *, window: int) -> float:
    """Alignment-invariant least-squares cost for a given lag.

    Uses:
        min_c ||s1 - c*s2||^2 = ||s1||^2 - |<s2,s1>|^2 / ||s2||^2
    from coprime_jacobi_2adic_fixed_periodviz.py. fileciteturn9file9L11-L14
    """
    xp = backend.xp
    n = int(x.shape[0])
    L = int(L)
    if L <= 0 or L >= n - 2:
        return float("inf")
    M = int(min(window, n - L - 1))
    if M < 4:
        return float("inf")
    s1 = x[:M]
    s2 = x[L:L + M]
    E1 = _vdot(backend, s1, s1).real
    E2 = _vdot(backend, s2, s2).real
    if float(backend.to_cpu(E1)) <= 0.0 or float(backend.to_cpu(E2)) <= 0.0:
        return float("inf")
    C = _vdot(backend, s2, s1)
    min_sse = E1 - (xp.abs(C) ** 2) / (E2 + 1e-24)
    return float(backend.to_cpu(min_sse.real / float(M)))


def scan_aligned_difference(
    backend: Backend,
    x,
    *,
    min_lag: int,
    max_lag: int,
    coarse_step: int = 25,
    refine_half_window: int = 1500,
    window: int = 65536,
):
    """Coarse-to-fine aligned-difference scan (no FFT).

    Mirrors the structure of _scan_period_by_aligned_difference (coarse then refine). fileciteturn9file9L16-L28
    """
    n = int(x.shape[0])
    min_lag = int(max(1, min_lag))
    max_lag = int(min(max_lag, (n // 2) - 2))
    if max_lag <= min_lag + 2:
        return None, {"reason": "lag_range_too_small"}

    Ls = list(range(min_lag, max_lag + 1, int(max(1, coarse_step))))
    costs = np.array([aligned_difference_cost(backend, x, L, window=window) for L in Ls], dtype=np.float64)
    i0 = int(np.argmin(costs))
    L0 = int(Ls[i0])

    lo = max(min_lag, L0 - int(refine_half_window))
    hi = min(max_lag, L0 + int(refine_half_window))
    Ls2 = list(range(lo, hi + 1))
    costs2 = np.array([aligned_difference_cost(backend, x, L, window=window) for L in Ls2], dtype=np.float64)
    i1 = int(np.argmin(costs2))
    L1 = int(Ls2[i1])

    diag = {
        "coarse_step": int(coarse_step),
        "coarse": {"lags": Ls, "costs": costs.tolist()},
        "refine": {"lags": Ls2, "costs": costs2.tolist()},
        "best_cost": float(costs2[i1]),
    }
    return L1, diag