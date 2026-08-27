"""Technical QC gate (PLAN.md Layer 8: VQAThinker + cheap ffmpeg checks).

Two tiers, so the gate is never skipped just because the GPU box is busy:

1. ffmpeg tier (CPU, anywhere): blackdetect / freezedetect / duration drift.
   Catches dead renders, stuck stills, and assembly timing bugs for free.
2. VQAThinker tier (GPU box): no-reference VQA on a per-event basis - bad
   upscales, artifacts, jarring generated shots. Lazy import, injectable.

Both tiers emit defects as intervals; map_defects_to_events() converts them
into the Layer 8 contract: EDL event id + timestamp + failing attribute.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

DURATION_DRIFT_S = 1.0     # assembled runtime may drift this much from the EDL
MIN_BLACK_S = 0.4          # black shorter than this can be an intentional dip
MIN_FREEZE_S = 3.0         # stills legitimately hold; only flag long freezes
                           # on events that are supposed to be FOOTAGE
VQA_MIN = 0.35             # VQAThinker per-event floor

_BLACK_RE = re.compile(r"black_start:(?P<s>[\d.]+).*?black_end:(?P<e>[\d.]+)")
_FREEZE_START_RE = re.compile(r"freeze_start: (?P<s>[\d.]+)")
_FREEZE_END_RE = re.compile(r"freeze_end: (?P<e>[\d.]+)")


def run_technical_checks(video_path: Path, edl: dict) -> dict:
    """CPU tier. Runs ffmpeg detectors and maps defects to EDL events."""
    defects = []
    duration = _ffprobe_duration(video_path)
    if duration is not None and abs(duration - edl["duration_s"]) > DURATION_DRIFT_S:
        defects.append({"type": "duration_drift", "start": 0.0,
                        "end": edl["duration_s"],
                        "detail": f"assembled={duration:.2f}s edl={edl['duration_s']:.2f}s"})
    stderr = _run_detectors(video_path)
    defects += parse_black_intervals(stderr)
    defects += parse_freeze_intervals(stderr)
    mapped = map_defects_to_events(defects, edl)
    return {"passed": not mapped, "tier": "ffmpeg", "failures": mapped}


def parse_black_intervals(stderr: str) -> list[dict]:
    return [{"type": "black", "start": float(m["s"]), "end": float(m["e"])}
            for m in _BLACK_RE.finditer(stderr)
            if float(m["e"]) - float(m["s"]) >= MIN_BLACK_S]


def parse_freeze_intervals(stderr: str) -> list[dict]:
    starts = [float(m["s"]) for m in _FREEZE_START_RE.finditer(stderr)]
    ends = [float(m["e"]) for m in _FREEZE_END_RE.finditer(stderr)]
    return [{"type": "freeze", "start": s, "end": e}
            for s, e in zip(starts, ends) if e - s >= MIN_FREEZE_S]


def map_defects_to_events(defects: list[dict], edl: dict) -> list[dict]:
    """Layer 8 contract: every defect interval becomes per-event failures.
    Freezes are only failures on FOOTAGE events (stills hold by design)."""
    failures = []
    for d in defects:
        hits = [e for e in edl["events"]
                if e["video_in"] < d["end"] and e["video_out"] > d["start"]]
        if d["type"] == "freeze":
            hits = [e for e in hits if e["kind"] == "footage"]
        if not hits and d["type"] == "duration_drift":
            hits = edl["events"][-1:]  # drift shows up at the tail
        for e in hits:
            failures.append({
                "event_id": e["event_id"],
                "t": round(max(d["start"], e["video_in"]), 3),
                "attribute": d["type"],
                "detail": d.get("detail",
                                f"{d['type']} {d['start']:.2f}-{d['end']:.2f}s"),
            })
    return failures


def run_vqa_checks(video_path: Path, edl: dict,
                   scorer: Callable[[Path, float, float], float] | None = None,
                   vqa_min: float = VQA_MIN) -> dict:
    """GPU tier. `scorer(video, t0, t1) -> quality 0..1` per event span.
    Default lazy-loads VQAThinker; offline tests inject a scorer."""
    scorer = scorer or _vqa_thinker_scorer()
    failures = []
    for e in edl["events"]:
        q = scorer(video_path, e["video_in"], e["video_out"])
        if q < vqa_min:
            failures.append({"event_id": e["event_id"], "t": e["video_in"],
                             "attribute": "vqa_quality", "score": round(q, 4)})
    return {"passed": not failures, "tier": "vqa_thinker", "failures": failures}


def _run_detectors(video_path: Path) -> str:
    proc = subprocess.run(
        ["ffmpeg", "-i", str(video_path),
         "-vf", f"blackdetect=d={MIN_BLACK_S}:pix_th=0.10,"
                f"freezedetect=n=-60dB:d={MIN_FREEZE_S}",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True)
    return proc.stderr


def _ffprobe_duration(video_path: Path) -> float | None:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _vqa_thinker_scorer() -> Callable[[Path, float, float], float]:
    try:
        import torch  # noqa: F401
        from vqa_thinker import VQAThinker  # type: ignore
    except ImportError as err:
        raise RuntimeError(
            "VQA tier needs the GPU box: pip install torch vqa-thinker "
            "(or inject a scorer callable / rely on the ffmpeg tier)"
        ) from err
    model = VQAThinker.from_pretrained()

    def scorer(video_path: Path, t0: float, t1: float) -> float:
        return float(model.score_clip(str(video_path), start=t0, end=t1))
    return scorer
