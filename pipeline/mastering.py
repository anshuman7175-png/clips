"""Loudness mastering + hard QC gate (PLAN.md Layer 7).

Two-pass FFmpeg loudnorm to -14 LUFS integrated / -1 dBTP true peak,
then a measurement pass that FAILS the run if the master is out of spec.
Requires ffmpeg on PATH.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:]}")
    return proc.stderr  # loudnorm prints its JSON to stderr


def _measure(path: Path, i: float, tp: float) -> dict:
    stderr = _run([
        "ffmpeg", "-hide_banner", "-i", str(path),
        "-af", f"loudnorm=I={i}:TP={tp}:LRA=11:print_format=json",
        "-f", "null", "-",
    ])
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.DOTALL)
    if not match:
        raise RuntimeError("could not parse loudnorm measurement")
    return json.loads(match.group(0))


def master(in_path: Path, out_path: Path, cfg) -> dict:
    i, tp = cfg.target_lufs, cfg.target_true_peak_db
    m = _measure(in_path, i, tp)  # pass 1: measure
    _run([  # pass 2: linear-mode normalization using measured values
        "ffmpeg", "-hide_banner", "-y", "-i", str(in_path),
        "-af",
        (
            f"loudnorm=I={i}:TP={tp}:LRA=11:linear=true"
            f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
            f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
        ),
        "-ar", "48000", str(out_path),
    ])
    # QC gate: re-measure the master.
    check = _measure(out_path, i, tp)
    out_i, out_tp = float(check["input_i"]), float(check["input_tp"])
    passed = abs(out_i - i) <= cfg.lufs_tolerance and out_tp <= tp + 0.3
    result = {
        "path": str(out_path),
        "integrated_lufs": out_i,
        "true_peak_db": out_tp,
        "target_lufs": i,
        "qc_passed": passed,
    }
    if not passed:
        raise RuntimeError(f"MASTERING QC FAILED: {result}")
    return result
