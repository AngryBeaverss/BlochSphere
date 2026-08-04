# Bloch Orderfinder (Bloch-lattice period recovery)

Primary objective: recover the multiplicative order **r** such that:

`pow(a, r, N) == 1`

Factoring is optional (only for validation). The core method is:

1. **Encode** residues `a^e mod N` into a Bloch-sphere lattice (modphase / modbin / modval / modmask).
2. Optionally evolve that lattice via **LLG** (Landau–Lifshitz–Gilbert) dynamics.
3. Score periodicity using either:
   - **image-native 2D torus shift symmetry** (recommended): detects translational invariances directly in the lattice
   - 1D flattened signature (legacy): harmonic-aware normalized correlation + aligned-difference fallback
4. **Verify** candidates with `pow(a, r, N) == 1` and minimize if the candidate is a multiple.

## Install / run (local)

From repo root:

```bash
python -m pip install -r requirements.txt  # optional
python -m bloch_orderfinder --help
```

CuPy is optional. If present, the program uses GPU automatically (or force with `--backend gpu`).

## Quick start: period sweep

Pick N, a, and a grid size large enough to contain several periods:

```bash
python -m bloch_orderfinder \
  --N 91 --a 3 \
  --size 256 --pattern modphase \
  --steps 0 \
  --seq psi1 \
  --lag-min 2 --lag-max 2000 --lag-step 1 --topk 10 --max-harmonic 6 \
  --aligned-fallback \
  --outdir runs --save-initial --save-png
```

The program prints ranked candidates.

### Image-native scoring (recommended)

This mode keeps the image central and avoids row-major flattening artifacts by scoring 2D
translation symmetry on the torus using FFT autocorrelation.

```bash
python -m bloch_orderfinder \
  --N 91 --a 3 \
  --size 256 --pattern modphase \
  --steps 0 \
  --seq psi1 \
  --score-mode image \
  --score2d-dx-max 64 --score2d-dy-max 64 --score2d-topk 10
```

Each candidate prints as `shift=(dx,dy)` with an implied exponent shift `delta_x = dx + dy*row_stride`.

## Robustness knob: row stride

The exponent mapping is:

`e(y, x) = origin + y*row_stride + x`

Default `row_stride == size` reproduces the v3 mapping `e = y*size + x`. You can make the display width independent from the exponent step by changing `--row-stride`.

Example:

```bash
python -m bloch_orderfinder --N 91 --a 3 --size 256 --row-stride 1024 ...
```

## Output

A run directory is created under `--outdir` (default: `runs/`). It contains:

- `S_init_*.npz` (optional) and `S_final_*.npz` (always)
- optional PNG frames (requires Pillow)
- `summary.txt` with key parameters and top candidates

Each `.npz` stores `Sx,Sy,Sz` plus step and optional diagnostics.

## Repo layout

- `bloch_orderfinder/core/encoder.py` — encoding patterns and exponent mapping
- `bloch_orderfinder/core/modexp.py` — GPU/CPU modular exponentiation
- `bloch_orderfinder/core/dynamics.py` — LLG + coherent neighbor overlap
- `bloch_orderfinder/core/scoring.py` — harmonic-aware scoring + aligned fallback
- `bloch_orderfinder/core/verifier.py` — candidate verification/minimization
- `bloch_orderfinder/vis/visualize.py` — rasterize + save npz/png
- `bloch_orderfinder/cli/main.py` — single CLI entrypoint
- `bloch_orderfinder/tests/` — pytest tests

## Fermat / large-modulus mode

This patched version keeps the original square-grid behavior, but adds two
backward-compatible options needed for large Fermat experiments:

- `--height H`: use an `H x size` rectangular lattice.
- exact arbitrary-precision residue generation with immediate `--reduce-mod`
  projection, so `N` is no longer limited to 64 bits on the CPU path.

The exponent map remains

`e(y, x) = origin + y*row_stride + x`.

A period lock depends on `row_stride`, not on storing a square whose two sides
both equal the period.  Thus a small rectangular image can test a large period.
For the base-2 Fermat-12 control (`ord(2) = 8192`):

```bash
N=$(python -c 'print((1 << 4096) + 1)')
python -m bloch_orderfinder \
  --N "$N" --a 2 \
  --size 256 --height 4 --row-stride 8192 \
  --pattern modphase --reduce-mod 8192 \
  --steps 1 --substeps 1 \
  --score-mode image \
  --score2d-dx-max 0 --score2d-dy-max 1 \
  --lag-min 2 --lag-max 10000 \
  --row-lock-report --backend cpu
```

The expected control output contains:

```text
[row-lock] initial=1.000000000
[row-lock] final=1.000000000 persistent=YES
shift=(dx=0, dy=1) ... delta_x=8192
r=8192 VERIFIED (minimal order)
```

For a modulus larger than 64 bits, `--reduce-mod` is required.  The code still
computes every recurrence step exactly modulo `N` with Python big integers;
only the value stored in the image is reduced.
