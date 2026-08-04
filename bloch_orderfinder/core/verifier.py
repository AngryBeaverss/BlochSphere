"""Verifier: validate and simplify candidate r.

We use the pow(a, r, N) == 1 check and a light minimalization pass (strip small primes),
as implemented in best_cupy_shor_orderfinder_harmonics.py. fileciteturn9file11L3-L10
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt, gcd
from typing import Iterable
import sys
sys.set_int_max_str_digits(0)


_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23)


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    r_in: int
    r_min: int | None
    reason: str
    multiple_of_min: int | None = None


@dataclass(frozen=True)
class FactorLeakResult:
    """GCD-based 'CRT leak' diagnostic for a candidate r.

    If a^r ≡ 1 (mod p) but not (mod q) for N=pq, then gcd(a^r - 1, N) reveals p.
    We also compute gcd(a^r + 1, N) which can reveal a factor when a^r ≡ -1 (mod p).
    """

    r_in: int
    a_pow_r_mod_N: int
    gcd_minus_1: int
    gcd_plus_1: int

    def nontrivial_factors(self, N: int) -> tuple[int, ...]:
        out = []
        if 1 < self.gcd_minus_1 < N:
            out.append(self.gcd_minus_1)
        if 1 < self.gcd_plus_1 < N and self.gcd_plus_1 not in out:
            out.append(self.gcd_plus_1)
        return tuple(out)

def factor_leak(a: int, N: int, r: int) -> FactorLeakResult:
    """Compute gcd(a^r ± 1, N) as a diagnostic for factor-order leakage."""
    rr = int(r)
    x = pow(int(a), rr, int(N))
    gm = gcd(x - 1, int(N))
    gp = gcd(x + 1, int(N))
    return FactorLeakResult(r_in=rr, a_pow_r_mod_N=x, gcd_minus_1=gm, gcd_plus_1=gp)


def is_order_multiple(a: int, N: int, r: int) -> bool:
    return pow(int(a), int(r), int(N)) == 1


def minimize_order(a: int, N: int, r: int, *, thorough_if_small: bool = True, thorough_cap: int = 1_000_000) -> int | None:
    """Reduce a verified r to (likely) minimal order by stripping factors."""
    r = int(r)
    if r <= 1:
        return None
    if pow(a, r, N) != 1:
        return None

    # Strip small primes first (fast and matches prototype). fileciteturn9file11L5-L9
    for p in _SMALL_PRIMES:
        while r % p == 0 and pow(a, r // p, N) == 1:
            r //= p

    # Optional thorough minimization by scanning remaining factors when r is small.
    if thorough_if_small and r <= thorough_cap:
        changed = True
        while changed:
            changed = False
            lim = isqrt(r)
            for d in range(2, lim + 1):
                if r % d != 0:
                    continue
                q = r // d
                # Try smaller factor first
                if pow(a, q, N) == 1:
                    r = q
                    changed = True
                    break
                if pow(a, d, N) == 1:
                    r = d
                    changed = True
                    break

    return int(r)


def verify_candidate(a: int, N: int, r: int) -> VerifyResult:
    r = int(r)
    if r <= 1:
        return VerifyResult(ok=False, r_in=r, r_min=None, reason="r<=1")

    if pow(a, r, N) != 1:
        return VerifyResult(ok=False, r_in=r, r_min=None, reason="pow(a,r,N)!=1")

    r_min = minimize_order(a, N, r)
    if r_min is None:
        return VerifyResult(ok=False, r_in=r, r_min=None, reason="minimize_failed")

    mult = r // r_min if r_min and r % r_min == 0 else None
    if r_min == r:
        return VerifyResult(ok=True, r_in=r, r_min=r_min, reason="r_is_order", multiple_of_min=mult)
    return VerifyResult(ok=True, r_in=r, r_min=r_min, reason="r_is_multiple_of_order", multiple_of_min=mult)
