"""Single CLI entrypoint for Bloch-lattice order finding."""




import argparse
import sys
sys.set_int_max_str_digits(0)
import os
import time
from math import gcd


from ..core.backend import get_backend
from ..core.encoder import ExponentMapping, init_spins
from ..core.dynamics import llg_heun, compute_diagnostics
from ..core.scoring import (
    extract_sequence,
    extract_field2d,
    scan_correlation,
    scan_aligned_difference,
    scan_2d_torus_shifts_fft,
    scan_2d_staggered_shifts_fft,
    LagScore,
)
from ..core.verifier import verify_candidate, factor_leak
from ..vis.visualize import save_npz, save_png


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="bloch_orderfinder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Order finding via Bloch-sphere lattice encodings and Bloch-lattice dynamics (GPU-first, CuPy).",
    )

    # ---------------- Core problem ----------------
    g = ap.add_argument_group("Core algorithm")
    g.add_argument("--N", dest="mod_N", type=int, required=True, help="Modulus N for order finding")
    g.add_argument("--a", dest="mod_a", type=int, required=True, help="Base a (must be coprime to N for an order)")
    g.add_argument("--size", type=int, default=256, help="Grid width (and height unless --height is supplied)")
    g.add_argument(
        "--height",
        type=int,
        default=None,
        help="Grid height; default=size preserves the original square lattice",
    )
    g.add_argument(
        "--pattern",
        type=str,
        default="modphase",
        choices=["random", "up", "hsv", "modval", "modbin", "modphase", "modmask"],
        help="Bloch encoding pattern",
    )
    g.add_argument(
        "--reduce-mod",
        type=int,
        default=None,
        help="Optional uint64 image projection of exact residues (required when N exceeds 64 bits)",
    )
    g.add_argument("--modmask", type=str, default=None, help="Comma-separated residue list used by modmask")

    # Exponent mapping (stability knob)
    g.add_argument(
        "--row-stride",
        type=int,
        default=None,
        help="Exponent increment per row; default=size keeps v3 compatibility (exp=y*size+x)",
    )

    # ---------------- Dynamics (optional) ----------------
    d = ap.add_argument_group("Dynamics (optional)")
    d.add_argument("--steps", type=int, default=0, help="Number of outer LLG steps (0 disables dynamics)")
    d.add_argument("--J", type=float, default=1.0)
    d.add_argument("--alpha", type=float, default=0.05)
    d.add_argument("--gamma", type=float, default=1.0)
    d.add_argument("--dt", type=float, default=0.01)
    d.add_argument("--substeps", type=int, default=10)
    d.add_argument("--Bx", type=float, default=0.0)
    d.add_argument("--By", type=float, default=0.0)
    d.add_argument("--Bz", type=float, default=0.1)
    d.add_argument("--coherent", action="store_true", help="Use coherent neighbor-overlap transverse field")
    d.add_argument("--psi-scale", type=float, default=1.0, help="Scale factor for coherent transverse field")
    d.add_argument("--Jz", type=float, default=None, help="Optional separate Sz coupling (only used with --coherent)")

    # ---------------- Scoring ----------------
    s = ap.add_argument_group("Scoring")
    s.add_argument(
        "--score-mode",
        type=str,
        default="both",
        choices=["both", "flat", "image"],
        help="Scoring domain: 'flat' uses 1D row-major sequence; 'image' uses 2D torus shift symmetry; 'both' prints both",
    )
    s.add_argument(
        "--seq", type=str, default="psi1", choices=["sz", "sx", "sy", "psi1"], help="Sequence extracted for scoring"
    )
    s.add_argument("--score2d-dx-max", type=int, default=-1, help="Max |dx| searched for 2D shift scoring (-1 = no limit)")
    s.add_argument("--score2d-dy-max", type=int, default=-1, help="Max |dy| searched for 2D shift scoring (-1 = no limit)")
    s.add_argument("--score2d-topk", type=int, default=10, help="Top-k shifts to report for 2D shift scoring")
    s.add_argument("--score2d-allow-zero-delta", action="store_true", help="Allow delta_x=0 shift candidates (usually uninformative)")
    s.add_argument(
        "--row-lock-report",
        action="store_true",
        help="Report vertical row-translation correlation before and after LLG",
    )
    s.add_argument(
        "--row-lock-threshold",
        type=float,
        default=0.999999,
        help="Threshold used to label a persistent row lock",
    )

    s.add_argument(
        "--leak-check",
        action="store_true",
        help="Run gcd(a^r ± 1, N) diagnostics on top candidates to detect factor-order 'CRT leakage'",
    )
    s.add_argument("--leak-topk", type=int, default=10, help="How many unique candidate r values to leak-check")
    s.add_argument("--leak-mult-max", type=int, default=5, help="Test multiples 2r,3r,...,mr for leak (default 5)")
    s.add_argument("--leak-no-stop", action="store_true", help="Harvest every distinct leak across the full sweep instead of returning on the first nontrivial factor")
    s.add_argument("--leak-fuzzy-window", type=int, default=0, help="Search ±window around each candidate (0 disables)")
    s.add_argument("--leak-fuzzy-step", type=int, default=100, help="Step size for fuzzy leak search")
    s.add_argument("--wrap-lifts", type=int, default=0, help="Also test lifted candidates r + t*M where M=size*row_stride, for t=1..wrap_lifts (0 disables). Applied to leak-check and verify.")
    s.add_argument("--lag-min", type=int, default=2)
    s.add_argument("--lag-max", type=int, default=5000)
    s.add_argument("--lag-step", type=int, default=1)
    s.add_argument("--topk", type=int, default=10)
    s.add_argument("--max-harmonic", type=int, default=6)
    s.add_argument("--score-window", type=int, default=None, help="Optional fixed window length for correlation scoring")
    s.add_argument("--aligned-fallback", action="store_true", help="Also run aligned-difference lag scan fallback")
    s.add_argument("--aligned-window", type=int, default=65536)
    s.add_argument("--aligned-coarse-step", type=int, default=25)
    s.add_argument("--aligned-refine-half-window", type=int, default=1500)

    # ---------------- Output / reproducibility ----------------
    o = ap.add_argument_group("Output / reproducibility")
    o.add_argument("--outdir", type=str, default="runs", help="Output directory")
    o.add_argument("--run-id", type=str, default=None, help="Run identifier (subdir); default is timestamp")
    o.add_argument("--save-initial", action="store_true", help="Save initial state as npz/png")
    o.add_argument("--save-every", type=int, default=0, help="Save every K dynamics steps (0 disables)")
    o.add_argument("--save-final", action="store_true", help="Save final state as npz (default: off)")
    o.add_argument("--save-png", action="store_true", help="Also save PNGs (requires Pillow)")
    o.add_argument("--viz", type=str, default="hsv", choices=["hsv", "sz", "szcolor"])

    # ---------------- Backend ----------------
    b = ap.add_argument_group("Backend")
    b.add_argument("--backend", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    b.add_argument("--float64", action="store_true", help="Use float64 for dynamics/scoring (slower)")
    b.add_argument("--seed", type=int, default=42)

    return ap


def _mk_run_dir(outdir: str, run_id: str | None):
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(outdir, run_id)
    os.makedirs(path, exist_ok=True)
    return path

def _int_modexp(a, e, N):
    return int(pow(int(a), int(e), int(N)))

def _as_int(x):
    # handles python int, numpy scalar, cupy scalar
    if x is None:
        return None
    try:
        return int(x)
    except TypeError:
        # some scalars need .item()
        return int(x.item())




def _row_lock_score(backend, S) -> float:
    """Mean Bloch-vector dot product under a one-row torus translation."""
    xp = backend.xp
    if int(S.shape[0]) < 2:
        return 1.0
    shifted = xp.roll(S, -1, axis=0)
    dots = xp.sum(S * shifted, axis=-1)
    return float(backend.to_cpu(xp.mean(dots)))

def _harmonic_cluster_deltas(
    shift_candidates,
    base_delta=None,
    tol=0.01,
    min_mult=1,
    max_mult=24,
):
    """Return |delta_x| values close to integer multiples of a base delta."""
    from fractions import Fraction

    if not shift_candidates:
        return []

    if base_delta is None:
        base_delta = abs(int(shift_candidates[0].delta_x))

    base_delta = abs(int(base_delta))
    if base_delta <= 0:
        return []

    # Convert 0.01 into the exact rational 1/100.
    tolerance = Fraction(str(tol))
    tol_num = tolerance.numerator
    tol_den = tolerance.denominator

    kept = []

    for candidate in shift_candidates:
        d = abs(int(candidate.delta_x))
        if d <= 0:
            continue

        # Nearest integer multiple without converting either integer to float.
        m = (d + base_delta // 2) // base_delta

        if m < min_mult or m > max_mult:
            continue

        target = m * base_delta
        error = abs(d - target)

        # Equivalent to:
        # error <= tol * target
        # but uses exact integer arithmetic.
        if error * tol_den <= target * tol_num:
            kept.append(d)

    return kept



def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    backend = get_backend(args.backend)
    xp = backend.xp
    dtype = xp.float64 if args.float64 else xp.float32

    run_dir = _mk_run_dir(args.outdir, args.run_id)
    print(f"[run] backend={'cupy' if backend.on_gpu else 'numpy'} dtype={dtype} out={run_dir}")

    mapping = ExponentMapping(size=args.size, height=args.height, row_stride=args.row_stride)
    width = mapping.width
    height = mapping.rows
    row_stride = mapping.row_stride if mapping.row_stride is not None else width
    M = int(row_stride) * int(height)
    print(f"[grid] shape=({height},{width}) row_stride={row_stride}")
    wrap_lifts = max(0, int(getattr(args, 'wrap_lifts', 0)))

    # ---------------- Encoder ----------------
    S = init_spins(
        backend,
        mapping,
        pattern=args.pattern,
        seed=args.seed,
        mod_N=args.mod_N,
        mod_a=args.mod_a,
        reduce_mod=args.reduce_mod,
        modmask=args.modmask,
        dtype=dtype,
    )

    initial_row_lock = _row_lock_score(backend, S) if args.row_lock_report else None
    if args.row_lock_report:
        print(f"[row-lock] initial={initial_row_lock:.9f}")

    if args.save_initial:
        npz = save_npz(backend, run_dir, "S_init", 0, S)
        print(f"[save] {npz}")
        if args.save_png:
            png = save_png(backend, run_dir, "S_init", 0, S, viz_mode=args.viz)
            if png:
                print(f"[save] {png}")

    # ---------------- Dynamics (optional) ----------------
    if args.steps > 0:
        Bext = (args.Bx, args.By, args.Bz)
        for step in range(1, int(args.steps) + 1):
            S = llg_heun(
                backend,
                S,
                steps=1,
                J=args.J,
                Bext=Bext,
                alpha=args.alpha,
                gamma=args.gamma,
                dt=args.dt,
                substeps=args.substeps,
                coherent=args.coherent,
                psi_scale=args.psi_scale,
                Jz=args.Jz,
            )
            if args.save_every and (step % int(args.save_every) == 0):
                diag = compute_diagnostics(backend, S, J=args.J)
                npz = save_npz(backend, run_dir, "S", step, S, diagnostics=diag)
                print(f"[save] {npz}  E={diag.E:.4f} align={diag.alignment:.4f}")
                if args.save_png:
                    png = save_png(backend, run_dir, "S", step, S, viz_mode=args.viz)
                    if png:
                        print(f"[save] {png}")

    final_row_lock = _row_lock_score(backend, S) if args.row_lock_report else None
    if args.row_lock_report:
        persistent = (
            initial_row_lock is not None
            and initial_row_lock >= float(args.row_lock_threshold)
            and final_row_lock >= float(args.row_lock_threshold)
        )
        print(
            f"[row-lock] final={final_row_lock:.9f} "
            f"persistent={'YES' if persistent else 'no'} "
            f"threshold={float(args.row_lock_threshold):.9f}"
        )

    # Save final state only if requested
    diag = compute_diagnostics(backend, S, J=args.J)
    if getattr(args, "save_final", False):
        npz = save_npz(backend, run_dir, "S_final", int(args.steps), S, diagnostics=diag)
        print(f"[save] {npz}  E={diag.E:.4f} align={diag.alignment:.4f}")
        if args.save_png:
            png = save_png(backend, run_dir, "S_final", int(args.steps), S, viz_mode=args.viz)
            if png:
                print(f"[save] {png}")
    else:
        print(f"[diag] E={diag.E:.4f} align={diag.alignment:.4f}")

    # ---------------- Scoring ----------------
    eff_stride = args.row_stride if args.row_stride is not None else args.size

    shift_candidates = []
    stagger_candidates = []
    inferred_image_r = None
    if args.score_mode in ("both", "image"):
        Z2d = extract_field2d(backend, S, mode=args.seq)
        H, W = int(Z2d.shape[0]), int(Z2d.shape[1])
        print(
            f"[score2d] field={args.seq} shape=({H},{W}) scanning shifts (dx_max={args.score2d_dx_max}, dy_max={args.score2d_dy_max}) with |delta_x| in [{args.lag_min},{args.lag_max}]"
        )

        shift_candidates = scan_2d_torus_shifts_fft(
            backend,
            Z2d,
            row_stride=int(eff_stride),
            dx_max=int(args.score2d_dx_max),
            dy_max=int(args.score2d_dy_max),
            topk=int(args.score2d_topk),
            exclude_zero_delta=(not bool(args.score2d_allow_zero_delta)),
            min_abs_delta_x=int(args.lag_min),
            max_abs_delta_x=int(args.lag_max),
        )
        if shift_candidates:
            print("\n[score2d] top torus-shift symmetry candidates:")
            for i, c in enumerate(shift_candidates, 1):
                print(
                    f"  {i:2d}. shift=(dx={c.dx:4d}, dy={c.dy:4d}) score={c.score:.6f} delta_x={c.delta_x}"
                )

            # Diagnostics: compare best score to a crude noise floor ~ 1/sqrt(H*W)

            import math

            noise_floor = 1.0 / math.sqrt(max(1, H * W))

            best_score = float(shift_candidates[0].score)

            snr = (best_score / noise_floor) if noise_floor > 0 else 0.0

            print(f"[score2d] best_score={best_score:.6g} noise~{noise_floor:.6g} snr~{snr:.2f}")


            # Robust image-native 'fundamental' estimate:

            #  - if the score surface is near-noise, do NOT infer a fundamental

            #  - otherwise, keep only deltas consistent with a harmonic family and gcd within that cluster

            inferred_image_r = None

            if snr >= 4.0:

                base = abs(int(shift_candidates[0].delta_x))

                cluster = _harmonic_cluster_deltas(shift_candidates, base_delta=base, tol=0.01, max_mult=24)

                if cluster:

                    g = 0

                    for d in cluster:

                        g = math.gcd(g, d)

                    if g > 0:

                        print(f"[score2d] inferred fundamental delta_x (harmonic-cluster gcd) = {g}")

                        inferred_image_r = g

                else:

                    print("[score2d] skipped fundamental inference (no harmonic-consistent cluster)")

            else:

                print("[score2d] skipped fundamental inference (scores near noise floor); still using candidates for leak/verify")
        else:
            print("[score2d] no shift candidates")

        # When row_stride differs from the image width, score the already-present
        # row staggering without torus wrap.  Existing 2D/lag limits are reused.
        if H > 1 and int(eff_stride) != W:
            stagger_candidates = scan_2d_staggered_shifts_fft(
                backend,
                Z2d,
                row_stride=int(eff_stride),
                dx_max=int(args.score2d_dx_max),
                dy_max=int(args.score2d_dy_max),
                topk=int(args.score2d_topk),
                exclude_zero_delta=(not bool(args.score2d_allow_zero_delta)),
                min_abs_delta_x=int(args.lag_min),
                max_abs_delta_x=int(args.lag_max),
            )
            if stagger_candidates:
                print("\n[score2d-stagger] top nonwrapped row-pair candidates:")
                for i, c in enumerate(stagger_candidates, 1):
                    print(
                        f"  {i:2d}. shift=(dx={c.dx:4d}, dy={c.dy:4d}) "
                        f"score={c.score:.6f} delta_x={c.delta_x}"
                    )
            else:
                print("[score2d-stagger] no row-pair candidates")

    candidates = []
    x = None
    n = None
    max_lag = None
    if args.score_mode in ("both", "flat"):
        x = extract_sequence(backend, S, mode=args.seq)
        n = int(x.shape[0])
        max_lag = min(int(args.lag_max), (n // 2) - 2)
        print(f"[score] sequence={args.seq} length={n} scanning lags {args.lag_min}..{max_lag} step={args.lag_step}")

        candidates = scan_correlation(
            backend,
            x,
            min_lag=int(args.lag_min),
            max_lag=int(max_lag),
            step=int(args.lag_step),
            topk=int(args.topk),
            max_harmonic=int(args.max_harmonic),
            window=args.score_window,
        )

        if candidates:
            print("\n[score] top harmonic-aware correlation candidates:")
            for i, c in enumerate(candidates, 1):
                print(f"  {i:2d}. lag={c.lag:8d} score={c.score:.6f}")
        else:
            print("[score] no correlation candidates")

    aligned_best = None
    if args.aligned_fallback:
        if x is None or max_lag is None:
            print("\n[score] aligned-difference fallback requested but score_mode=image; skipping (no 1D sequence).")
        else:
            print("\n[score] running aligned-difference fallback scan ...")
            L1, diag2 = scan_aligned_difference(
                backend,
                x,
                min_lag=int(args.lag_min),
                max_lag=int(max_lag),
                coarse_step=int(args.aligned_coarse_step),
                refine_half_window=int(args.aligned_refine_half_window),
                window=int(args.aligned_window),
            )
            if L1 is not None:
                aligned_best = LagScore(
                    lag=int(L1),
                    score=float(-diag2.get("best_cost", 0.0)),
                    method="aligned_diff",
                    detail=diag2,
                )
                print(f"  aligned best lag={L1} best_cost={diag2.get('best_cost')}")
            else:
                print(f"  aligned scan failed: {diag2}")

    # ---------------- Verifier ----------------
    # Lags live on the sampled torus; one vertical circuit advances height*row_stride.
    L_torus = int(eff_stride) * int(height)

    def _add_with_complement(out, r, front=False):
        """Add r and its torus-complement (L_torus-r) while preserving priority order."""
        rr = int(r)
        if rr <= 0:
            return
        rc = L_torus - rr
        pair = [rr]
        if 0 < rc != rr:
            pair.append(rc)
        if front:
            out[0:0] = pair
        else:
            out.extend(pair)

    to_test = []
    if aligned_best is not None:
        _add_with_complement(to_test, aligned_best.lag, front=True)

    # Interleave ordinary torus shifts with nonwrapped row-stagger shifts so the
    # existing leak-topk budget samples both geometries without another knob.
    image_topk = max(1, int(args.score2d_topk))
    for i in range(image_topk):
        if i < len(shift_candidates):
            _add_with_complement(to_test, abs(int(shift_candidates[i].delta_x)))
        if i < len(stagger_candidates):
            _add_with_complement(to_test, abs(int(stagger_candidates[i].delta_x)))

    for c in candidates:
        _add_with_complement(to_test, c.lag)

    seen = set()
    lags = []
    for r in to_test:
        rr = int(r)
        if rr not in seen:
            seen.add(rr)
            lags.append(rr)

    if getattr(args, "leak_check", False):
        print("\n[leak]")
        found_factors = set()

        fuzzy_window = int(getattr(args, "leak_fuzzy_window", 0) or 0)
        fuzzy_step = max(1, int(getattr(args, "leak_fuzzy_step", 100) or 100))

        a = int(args.mod_a)
        N = int(args.mod_N)

        leak_mult_max = max(1, int(getattr(args, "leak_mult_max", 5) or 5))
        leak_no_stop = bool(getattr(args, "leak_no_stop", False))

        # Torus modulus for lifts
        M = int(M)  # already computed earlier; ensure int
        aM = pow(a, M, N) if wrap_lifts > 0 else None

        # For fuzzy stepping: multiply by a^(fuzzy_step)
        mul_fwd = pow(a, fuzzy_step, N) if fuzzy_window > 0 else None

        for r in lags[: max(1, int(args.leak_topk))]:
            r = int(r)
            if r <= 0:
                continue

            # base a^r
            x_lift = pow(a, r, N)

            for t in range(0, wrap_lifts + 1):
                test_r0 = r + t * M

                # --- (A) test m * test_r0 using repeated multiplication by a^(test_r0)
                xm = 1
                for m in range(1, leak_mult_max + 1):
                    xm = (xm * x_lift) % N  # xm == a^(m*test_r0) mod N

                    # (optional parity optimization: if (m*test_r0) odd and M even, gm can be hopeless a lot of the time)
                    gm = gcd(xm - 1, N)
                    gp = gcd(xm + 1, N)

                    for g, which in ((gm, "-1"), (gp, "+1")):
                        if 1 < g < N and g not in found_factors:
                            found_factors.add(g)
                            if m == 1:
                                if t == 0:
                                    print(f"  r={test_r0} -> gcd(a^r{which},N)={g}  (factor leak)")
                                else:
                                    print(f"  r={test_r0} -> gcd(a^r{which},N)={g}  (factor leak) (lift t={t}, +{t}*M)")
                            else:
                                if t == 0:
                                    print(
                                        f"  r={m * test_r0} -> gcd(a^(m*r){which},N)={g}  (factor leak) (multiple {m}× of candidate {test_r0})")
                                else:
                                    print(
                                        f"  r={m * test_r0} -> gcd(a^(m*r){which},N)={g}  (factor leak) (multiple {m}×, lift t={t}, +{t}*M)")

                    if found_factors and not leak_no_stop:
                        break

                if found_factors and not leak_no_stop:
                    break

                # --- (B) fuzzy walk around test_r0: test_r0 + k*fuzzy_step
                if fuzzy_window > 0:
                    # start from a^(test_r0) (guaranteed non-None)
                    x = int(x_lift)
                    max_k = max(0, fuzzy_window // fuzzy_step)
                    rr = test_r0

                    for k in range(1, max_k + 1):
                        rr += fuzzy_step
                        x = (x * mul_fwd) % N  # now x == a^(rr) mod N

                        gm = gcd(x - 1, N)
                        gp = gcd(x + 1, N)
                        for g, which in ((gm, "-1"), (gp, "+1")):
                            if 1 < g < N and g not in found_factors:
                                found_factors.add(g)
                                print(
                                    f"  r={rr} -> gcd(a^r{which},N)={g}  (factor leak) (fuzzy +{k}*{fuzzy_step}, lift t={t})")
                        if found_factors and not leak_no_stop:
                            break

                if found_factors and not leak_no_stop:
                    break

                # IMPORTANT: update x_lift for next t (this was missing)
                if aM is not None and t < wrap_lifts:
                    x_lift = (x_lift * aM) % N

            if found_factors and not leak_no_stop:
                # default behavior: print cofactors and stop on first nontrivial factor
                for g in sorted(found_factors):
                    other = N // g
                    if g * other == N:
                        print(f"  factors: {g} × {other} = {N}")
                print("[leak] stopping early due to nontrivial factor")
                return 0

        # end of full candidate sweep
        if found_factors:
            print(f"[leak] harvested {len(found_factors)} distinct factor(s) across full sweep")
            for g in sorted(found_factors):
                other = N // g
                if g * other == N:
                    print(f"  factors: {g} × {other} = {N}")
        else:
            print("  no nontrivial gcd leaks in tested candidates")


    print("\n[verify]")
    any_ok = False
    seen_v = set()
    for r0 in lags[: max(1, int(args.topk) + 2)]:
        r0 = int(r0)
        if r0 <= 0:
            continue
        for t in range(0, wrap_lifts + 1):
            r = r0 + t * M
            if r in seen_v:
                continue
            seen_v.add(r)
            vr = verify_candidate(args.mod_a, args.mod_N, r)
            if vr.ok:
                any_ok = True
                if vr.reason == "r_is_order":
                    if t == 0:
                        print(f"  r={vr.r_in} VERIFIED (minimal order)")
                    else:
                        print(f"  r={vr.r_in} VERIFIED (minimal order) (lift t={t}, +{t}*M)")
                else:
                    if t == 0:
                        print(f"  r={vr.r_in} VERIFIED (multiple); minimal r={vr.r_min} (x{vr.multiple_of_min})")
                    else:
                        print(f"  r={vr.r_in} VERIFIED (multiple); minimal r={vr.r_min} (x{vr.multiple_of_min}) (lift t={t}, +{t}*M)")
            else:
                if wrap_lifts > 0 and t > 0:
                    print(f"  r={vr.r_in} reject: {vr.reason} (lift t={t})")
                else:
                    print(f"  r={vr.r_in} reject: {vr.reason}")
    if not any_ok:
        print("  no verified r in tested candidates")

    # Save a small text summary
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"N={args.mod_N} a={args.mod_a}\n")
        eff_stride = args.row_stride if args.row_stride is not None else args.size
        eff_height = args.height if args.height is not None else args.size
        f.write(
            f"pattern={args.pattern} width={args.size} height={eff_height} "
            f"row_stride={args.row_stride} (effective={eff_stride})\n"
        )
        if args.row_lock_report:
            f.write(
                f"row_lock_initial={initial_row_lock} row_lock_final={final_row_lock} "
                f"threshold={args.row_lock_threshold}\n"
            )
        f.write(f"coherent={args.coherent} steps={args.steps} J={args.J} alpha={args.alpha} dt={args.dt} substeps={args.substeps}\n")
        lag_max_written = int(max_lag) if max_lag is not None else int(args.lag_max)
        f.write(f"score_mode={args.score_mode}\n")
        f.write(f"seq={args.seq} lag_min={args.lag_min} lag_max={lag_max_written} lag_step={args.lag_step} max_harmonic={args.max_harmonic}\n")
        f.write(f"score2d_dx_max={args.score2d_dx_max} score2d_dy_max={args.score2d_dy_max} score2d_topk={args.score2d_topk} score2d_allow_zero_delta={bool(args.score2d_allow_zero_delta)}\n")
        f.write(f"inferred_image_r={inferred_image_r}\n")
        if shift_candidates:
            f.write("top_shift_candidates:\n")
            for c in shift_candidates:
                f.write(f"  dx={c.dx} dy={c.dy} score={c.score} delta_x={c.delta_x}\n")
        if stagger_candidates:
            f.write("top_stagger_candidates:\n")
            for c in stagger_candidates:
                f.write(f"  dx={c.dx} dy={c.dy} score={c.score} delta_x={c.delta_x}\n")
        if candidates:
            f.write("top_candidates:\n")
            for c in candidates:
                f.write(f"  lag={c.lag} score={c.score}\n")
        if aligned_best is not None:
            f.write(f"aligned_best: lag={aligned_best.lag} best_cost={aligned_best.detail.get('best_cost')}\n")
    print(f"\n[run] wrote {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
