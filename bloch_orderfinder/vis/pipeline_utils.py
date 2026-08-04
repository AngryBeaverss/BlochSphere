"""Convenience pipeline helpers.

This file intentionally stays out of the order-finding logic.
It wires together:
- dynamics.llg_heun_checkpointed
- visualize.save_npz / save_preview_png
- visualize.make_mp4_from_pngs

So you can run large experiments and get:
- deterministic-ish checkpoint artifacts
- cheap previews for fast visual triage
- an optional MP4 summary
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from ..core.backend import Backend
from ..core.dynamics import Diagnostics, llg_heun_checkpointed
from .visualize import save_npz, save_preview_png, make_mp4_from_pngs, VizMode


def run_dynamics_with_artifacts(
    backend: Backend,
    S,
    *,
    out_dir: str,
    prefix: str = "S",
    steps: int,
    save_every: int = 0,
    save_npz_frames: bool = True,
    save_preview_frames: bool = True,
    preview_max: int = 512,
    viz_mode: VizMode = "hsv",
    make_mp4: bool = False,
    mp4_fps: int = 30,
    llg_diag_J: float = 1.0,
    **llg_kwargs,
):
    """Run LLG evolution and persist artifacts at checkpoints.

    Parameters
    ----------
    S:
        Initial spin field on the active backend.
    save_every:
        Checkpoint cadence in LLG outer steps.
        - 0 disables checkpointing (single run, no intermediate artifacts).
    save_npz_frames:
        Save full-resolution NPZ at each checkpoint.
    save_preview_frames:
        Save downsampled PNG previews at each checkpoint.
    make_mp4:
        If True, writes an MP4 from the preview PNGs.

    Returns
    -------
    Final (S, last_diagnostics, mp4_path_or_None)
    """

    last_diag: Optional[Diagnostics] = None

    def _cb(step: int, S_step, diag: Diagnostics):
        nonlocal last_diag
        last_diag = diag
        if save_npz_frames:
            save_npz(backend, out_dir, prefix, step, S_step, diagnostics=diag)
        if save_preview_frames:
            save_preview_png(backend, out_dir, prefix, step, S_step, viz_mode=viz_mode, preview_max=preview_max)

    S_final = llg_heun_checkpointed(
        backend,
        S,
        steps=int(steps),
        save_every=int(save_every),
        callback=_cb if (save_npz_frames or save_preview_frames) else None,
        include_initial=True,
        diag_J=float(llg_diag_J),
        **llg_kwargs,
    )

    mp4_path = None
    if make_mp4 and save_preview_frames:
        mp4_path = make_mp4_from_pngs(out_dir, prefix, fps=int(mp4_fps))

    return S_final, last_diag, mp4_path
