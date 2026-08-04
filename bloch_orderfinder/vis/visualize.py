"""Visualization and reproducibility helpers.

Features:
- rasterize spin field to RGB (hsv view or Sz heatmap)
- save frames as .npz (Sx,Sy,Sz, step + optional diagnostics)
- optionally save PNGs if Pillow is installed

Based on blocksorder_gpu_v3_coherent_gui.py save/rasterize approach. fileciteturn9file6L1-L27
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Literal, Optional

import numpy as np

from ..core.backend import Backend
from ..core.dynamics import Diagnostics


VizMode = Literal["hsv", "sz", "szcolor"]


def hsv_to_rgb_fast(xp, h, s, v):
    # From blocksorder_gpu_v3.py. fileciteturn9file13L57-L66
    h6 = h * 6.0
    i = xp.floor(h6).astype(xp.int32) % 6
    f = h6 - xp.floor(h6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r = xp.where(i == 0, v, xp.where(i == 1, q, xp.where(i == 2, p, xp.where(i == 3, p, xp.where(i == 4, t, v)))))
    g = xp.where(i == 0, t, xp.where(i == 1, v, xp.where(i == 2, v, xp.where(i == 3, q, xp.where(i == 4, p, p)))))
    b = xp.where(i == 0, p, xp.where(i == 1, p, xp.where(i == 2, t, xp.where(i == 3, v, xp.where(i == 4, v, q)))))
    rgb = xp.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(xp.uint8)


def rasterize(backend: Backend, S, mode: VizMode = "hsv"):
    xp = backend.xp
    if mode == "hsv":
        Sx = S[..., 0]
        Sy = S[..., 1]
        Sz = S[..., 2]
        h = (xp.arctan2(Sy, Sx) + xp.pi) / (2 * xp.pi)
        s = xp.ones_like(h)
        v = (Sz + 1) / 2
        return hsv_to_rgb_fast(xp, h, s, v)
    if mode in ("sz", "szcolor"):
        Sz = S[..., 2]
        v = ((Sz + 1) / 2).clip(0, 1)
        if mode == "sz":
            g = (v * 255).astype(xp.uint8)
            return xp.stack([g, g, g], axis=-1)
        # blue-white-red-ish without specifying exact palette: map to HSV with fixed hue is a palette.
        h = xp.where(Sz >= 0, 0.0, 2.0 / 3.0)  # red vs blue
        s = xp.abs(Sz).clip(0, 1)
        return hsv_to_rgb_fast(xp, h, s, v)
    raise ValueError(f"Unknown viz mode: {mode}")


def save_npz(backend: Backend, out_dir: str, prefix: str, step: int, S, diagnostics: Optional[Diagnostics] = None):
    os.makedirs(out_dir, exist_ok=True)
    arr = backend.to_cpu(S)
    save_dict = {"Sx": arr[..., 0], "Sy": arr[..., 1], "Sz": arr[..., 2], "step": int(step)}
    if diagnostics is not None:
        save_dict.update(asdict(diagnostics))
    path = os.path.join(out_dir, f"{prefix}_{step:08d}.npz")
    np.savez_compressed(path, **save_dict)
    return path


def save_png(backend: Backend, out_dir: str, prefix: str, step: int, S, *, viz_mode: VizMode = "hsv"):
    os.makedirs(out_dir, exist_ok=True)
    try:
        from PIL import Image
    except Exception:
        return None
    rgb = rasterize(backend, S, mode=viz_mode)
    img = Image.fromarray(backend.to_cpu(rgb))
    path = os.path.join(out_dir, f"{prefix}_{step:08d}.png")
    img.save(path)
    return path


def _preview_slice_factor(h: int, w: int, preview_max: int) -> int:
    preview_max = int(preview_max)
    if preview_max <= 0:
        return 1
    return max(1, int(np.ceil(max(h, w) / preview_max)))


def save_preview_png(
    backend: Backend,
    out_dir: str,
    prefix: str,
    step: int,
    S,
    *,
    viz_mode: VizMode = "hsv",
    preview_max: int = 512,
):
    """Save a downsampled PNG for quick scanning.

    This is intended for large lattices (e.g., 2048–6144), where saving full-res
    PNGs every checkpoint is expensive and visually redundant.

    Downsampling uses simple nearest-neighbor slicing (S[::k,::k]) so it is
    cheap on both CPU and GPU.
    """
    xp = backend.xp
    h, w = int(S.shape[0]), int(S.shape[1])
    k = _preview_slice_factor(h, w, int(preview_max))
    if k > 1:
        S_small = S[::k, ::k, :]
    else:
        S_small = S
    return save_png(backend, out_dir, prefix, step, S_small, viz_mode=viz_mode)


def make_mp4_from_pngs(
    out_dir: str,
    prefix: str,
    *,
    out_path: str | None = None,
    fps: int = 30,
    glob_pattern: str | None = None,
) -> str:
    """Create an MP4 from PNG frames.

    Requires imageio + an ffmpeg backend.

    Parameters
    ----------
    out_dir, prefix:
        Used to find frames like f"{prefix}_*.png" in out_dir.
    out_path:
        Defaults to f"{prefix}.mp4" in out_dir.
    fps:
        Frames per second.
    glob_pattern:
        Override the default glob if needed.

    Returns
    -------
    Path to the written MP4.
    """
    import glob

    import imageio.v2 as imageio  # v2 writer API is stable

    if glob_pattern is None:
        glob_pattern = os.path.join(out_dir, f"{prefix}_*.png")

    frames = sorted(glob.glob(glob_pattern))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found for pattern: {glob_pattern}")

    if out_path is None:
        out_path = os.path.join(out_dir, f"{prefix}.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    # Stream frames; avoids holding everything in RAM.
    try:
        with imageio.get_writer(out_path, fps=int(fps)) as writer:
            for fn in frames:
                writer.append_data(imageio.imread(fn))
    except Exception as e:
        raise RuntimeError(
            "Failed to write MP4 via imageio. Ensure an ffmpeg backend is available. "
            f"Underlying error: {e}"
        )

    return out_path
