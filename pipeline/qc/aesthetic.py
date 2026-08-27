"""Peak-End aesthetic QC gate (PLAN.md Layer 8: Peak-End-Net).

The Peak-End rule: perceived quality is dominated by the PEAK moment and the
ENDING, not the average. That is exactly what Act 3 and the midpoint reversal
control, so this gate scores probes non-uniformly:

1. build_probe_plan()     - CPU, pure: pick probe timestamps from the EDL.
                            Every event gets one probe; events that ARE the
                            peak (held reveals) and the final 10% are tagged
                            so the aggregate can weight them.
2. score_probes()         - GPU box: extract frames (ffmpeg) and score each
                            with Peak-End-Net (lazy import). Any callable
                            frame-scorer can be injected for offline tests.
3. evaluate_aesthetics()  - CPU, pure: Peak-End aggregate + per-attribute
                            thresholds. Failures map to event ids/timestamps.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

# Peak-End-Net's aesthetic attributes (10, plus the overall score).
ATTRIBUTES = ("composition", "lighting", "color_harmony", "depth_of_field",
              "motion", "sharpness", "exposure", "noise", "framing",
              "visual_interest")

# Peak-End aggregate weights: peak and ending dominate (the rule itself).
PEAK_WEIGHT, END_WEIGHT, MEAN_WEIGHT = 0.40, 0.40, 0.20
END_FRACTION = 0.10          # last 10% of runtime is "the ending"
DEFAULT_OVERALL_MIN = 0.55   # gate threshold on the aggregate
ATTRIBUTE_MIN = 0.30         # any single attribute below this fails its event


def build_probe_plan(edl: dict, timeline: dict) -> dict:
    """One probe per EDL event, at the event's visual midpoint.
    Tags: "peak" (held reveal events), "end" (final END_FRACTION), "body"."""
    duration = edl["duration_s"]
    end_start = duration * (1.0 - END_FRACTION)
    reveal_beats = {b["beat_id"] for b in timeline["beats"]
                    if any(s["is_reveal"] for s in b["sentences"])}
    probes = []
    for e in edl["events"]:
        t = round((e["video_in"] + e["video_out"]) / 2.0, 3)
        if e["hold"] and e["beat_id"] in reveal_beats:
            segment = "peak"
        elif t >= end_start:
            segment = "end"
        else:
            segment = "body"
        probes.append({"event_id": e["event_id"], "t": t, "segment": segment,
                       "kind": e["kind"], "source": e["source"]})
    return {"duration_s": duration, "probes": probes}


def score_probes(video_path: Path, plan: dict,
                 scorer: Callable[[Path], dict] | None = None,
                 frames_dir: Path | None = None) -> list[dict]:
    """Extract one frame per probe with ffmpeg and score it.
    `scorer(frame_path) -> {"overall": float, <attribute>: float, ...}`.
    Default scorer lazy-loads Peak-End-Net (GPU box)."""
    scorer = scorer or _peak_end_net_scorer()
    frames_dir = frames_dir or video_path.parent / "qc_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for probe in plan["probes"]:
        frame = frames_dir / f"{probe['event_id']}.png"
        if not frame.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(probe["t"]), "-i", str(video_path),
                 "-frames:v", "1", "-loglevel", "error", str(frame)],
                check=True)
        scored.append({**probe, "scores": scorer(frame)})
    return scored


def evaluate_aesthetics(plan: dict, scored_probes: list[dict],
                        overall_min: float = DEFAULT_OVERALL_MIN) -> dict:
    """Pure Peak-End aggregation + failure mapping. `scored_probes` items
    carry {"event_id", "t", "segment", "scores": {"overall", <attrs>}}."""
    if not scored_probes:
        return {"passed": False, "aggregate": 0.0, "failures": [],
                "reason": "no probes scored"}

    by_segment: dict[str, list[float]] = {"peak": [], "end": [], "body": []}
    for p in scored_probes:
        by_segment[p["segment"]].append(p["scores"]["overall"])
    all_scores = [p["scores"]["overall"] for p in scored_probes]

    # Peak = best moment anywhere (peak-tagged probes preferred, else global max).
    peak = max(by_segment["peak"]) if by_segment["peak"] else max(all_scores)
    end = (sum(by_segment["end"]) / len(by_segment["end"])
           if by_segment["end"] else all_scores[-1])
    mean = sum(all_scores) / len(all_scores)
    aggregate = round(PEAK_WEIGHT * peak + END_WEIGHT * end + MEAN_WEIGHT * mean, 4)

    # Per-event attribute failures -> EDL timestamps (Layer 8 contract).
    failures = []
    for p in scored_probes:
        for attr, val in p["scores"].items():
            if attr == "overall":
                continue
            if val < ATTRIBUTE_MIN:
                failures.append({"event_id": p["event_id"], "t": p["t"],
                                 "attribute": attr, "score": round(val, 4)})

    return {
        "passed": aggregate >= overall_min and not failures,
        "aggregate": aggregate,
        "peak": round(peak, 4), "end": round(end, 4), "mean": round(mean, 4),
        "overall_min": overall_min,
        "failures": failures,
    }


def _peak_end_net_scorer() -> Callable[[Path], dict]:
    """Lazy-load Peak-End-Net (MIT). GPU box only - offline runs inject a scorer."""
    try:
        import torch  # noqa: F401
        from peakend_net import PeakEndNet  # type: ignore
    except ImportError as err:
        raise RuntimeError(
            "aesthetic scoring needs the GPU box: pip install torch peakend-net "
            "(or inject a scorer callable / use run_milestone4 --skip-gpu)"
        ) from err
    model = PeakEndNet.from_pretrained()

    def scorer(frame_path: Path) -> dict:
        out = model.score(str(frame_path))
        return {"overall": float(out["overall"]),
                **{a: float(out.get(a, out["overall"])) for a in ATTRIBUTES}}
    return scorer
