"""Single CLI entrypoint for Bloch-lattice order finding."""

from __future__ import annotations

import argparse
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
    g.add_argument("--size", type=int, default=256, help="Grid size (size x size samples)")
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
        help="Optional modulus reduction applied after residues (modval/modbin/modphase)",
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
        "--leak-check",
        action="store_true",
        help="Run gcd(a^r ± 1, N) diagnostics on top candidates to detect factor-order 'CRT leakage'",
    )
    s.add_argument("--leak-topk", type=int, default=10, help="How many unique candidate r values to leak-check")
    s.add_argument("--leak-mult-max", type=int, default=5, help="Test multiples 2r,3r,...,mr for leak (default 5)")
    s.add_argument("--leak-fuzzy-window", type=int, default=0, help="Search ±window around each candidate (0 disables)")
    s.add_argument("--leak-fuzzy-step", type=int, default=100, help="Step size for fuzzy leak search")
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


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    backend = get_backend(args.backend)
    xp = backend.xp
    dtype = xp.float64 if args.float64 else xp.float32

    run_dir = _mk_run_dir(args.outdir, args.run_id)
    print(f"[run] backend={'cupy' if backend.on_gpu else 'numpy'} dtype={dtype} out={run_dir}")

    mapping = ExponentMapping(size=args.size, row_stride=args.row_stride)

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
            # Quick image-native 'fundamental' estimate: gcd of strong delta_x peaks
            import math
            g = 0
            for c in shift_candidates:
                g = math.gcd(g, abs(int(c.delta_x)))
            if g:
                print(f"[score2d] inferred fundamental delta_x (gcd of top peaks) = {g}")
                inferred_image_r = g
            else:
                inferred_image_r = None
        else:
            print("[score2d] no shift candidates")

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
    # Lags live on an SxS torus: treat candidates modulo L_torus and always test complements.
    L_torus = int(eff_stride) * int(args.size)

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

    # 2D shift candidates imply exponent shifts delta_x; include them next (positive magnitude).
    for sc in shift_candidates[: max(1, int(args.score2d_topk))]:
        _add_with_complement(to_test, abs(int(sc.delta_x)))

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
        fuzzy_window = getattr(args, "leak_fuzzy_window", 0)
        fuzzy_step = max(1, getattr(args, "leak_fuzzy_step", 100))

        a = int(args.mod_a)
        N = int(args.mod_N)

        for r in lags[: max(1, int(args.leak_topk))]:
            if r <= 0:
                continue

            # Test exact value first
            x = pow(a, r, N)

            # Multiplicative sweep: test m*r using x^m without extra pow()
            # m=1 tests r itself, m=2,3,... test multiples
            leak_mult_max = max(1, int(getattr(args, "leak_mult_max", 5)))
            xm = 1
            for m in range(1, leak_mult_max + 1):
                xm = (xm * x) % N  # now xm == a^(m*r) mod N
                gm = gcd(xm - 1, N)
                gp = gcd(xm + 1, N)
                for g, which in [(gm, "-1"), (gp, "+1")]:
                    if 1 < g < N and g not in found_factors:
                        found_factors.add(g)
                        if m == 1:
                            print(f"  r={r} -> gcd(a^r{which},N)={g}  (factor leak)")
                        else:
                            print(f"  r={m * r} -> gcd(a^r{which},N)={g}  (factor leak) (multiple {m}× of candidate {r})")
                if found_factors:
                    break


            # Fast fuzzy scan using multiplicative update
            if fuzzy_window > 0 and not found_factors:
                # Precompute multiplier for step size
                mul_fwd = pow(a, fuzzy_step, N)
                mul_bwd = pow(mul_fwd, -1, N)  # modular inverse for backward scan

                # Scan forward: r+step, r+2*step, ...
                x_fwd = (x * mul_fwd) % N
                for k in range(fuzzy_step, fuzzy_window + 1, fuzzy_step):
                    test_r = r + k
                    gm = gcd(x_fwd - 1, N)
                    gp = gcd(x_fwd + 1, N)
                    for g, which in [(gm, "-1"), (gp, "+1")]:
                        if 1 < g < N and g not in found_factors:
                            found_factors.add(g)
                            print(f"  r={test_r} -> gcd(a^r{which},N)={g}  (factor leak) (candidate {r} + {k})")
                    x_fwd = (x_fwd * mul_fwd) % N

                # Scan backward: r-step, r-2*step, ...
                x_bwd = (x * mul_bwd) % N
                for k in range(fuzzy_step, fuzzy_window + 1, fuzzy_step):
                    test_r = r - k
                    if test_r <= 0:
                        break
                    gm = gcd(x_bwd - 1, N)
                    gp = gcd(x_bwd + 1, N)
                    for g, which in [(gm, "-1"), (gp, "+1")]:
                        if 1 < g < N and g not in found_factors:
                            found_factors.add(g)
                            print(f"  r={test_r} -> gcd(a^r{which},N)={g}  (factor leak) (candidate {r} - {k})")
                    x_bwd = (x_bwd * mul_bwd) % N

        if not found_factors:
            print("  no nontrivial gcd leaks in tested candidates")
    print("\n[verify]")
    any_ok = False
    for r in lags[: max(1, int(args.topk) + 2)]:
        vr = verify_candidate(args.mod_a, args.mod_N, r)
        if vr.ok:
            any_ok = True
            if vr.reason == "r_is_order":
                print(f"  r={vr.r_in} VERIFIED (minimal order)")
            else:
                print(f"  r={vr.r_in} VERIFIED (multiple); minimal r={vr.r_min} (x{vr.multiple_of_min})")
        else:
            print(f"  r={vr.r_in} reject: {vr.reason}")

    if not any_ok:
        print("  no verified r in tested candidates")

    # Save a small text summary
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"N={args.mod_N} a={args.mod_a}\n")
        eff_stride = args.row_stride if args.row_stride is not None else args.size
        f.write(f"pattern={args.pattern} size={args.size} row_stride={args.row_stride} (effective={eff_stride})\n")
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
