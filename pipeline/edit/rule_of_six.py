"""Rule-of-Six EDL scorer (PLAN.md Layer 6).

Walter Murch's hierarchy, with his canonical weights:
    emotion 51% > story 23% > rhythm 10% > eye-trace 7% > 2D plane 5% > 3D space 4%

The builder produces N candidate EDLs from different jitter seeds; this module
scores each candidate and the best one ships. That is also the anti-templating
mechanism: the winning seed differs per video, so the cutting pattern never
repeats. M5's retention post-mortems adjust these weights over time.

Each component returns 0..1:
- emotion  : do shot durations track the emotion vectors? (tension -> shorter,
             gravity/reveals -> longer/held) + reveals actually held.
- story    : narration coverage - every second of a beat's audio is covered by
             a shot planned FOR that beat; coverage gaps are story failures.
- rhythm   : cut intervals must breathe - metronomic cutting (low variance)
             and flash-cutting are both penalized.
- eye_trace: consecutive hard cuts between same-kind shots (doc -> doc) make
             the eye jump with no motivation; J/L-cuts absorb this.
- plane_2d : variety of shot kinds across the video (all-documents reads flat).
- space_3d : source continuity inside a beat - whiplash between archival and
             generated within one beat breaks spatial belief.
"""

from __future__ import annotations

import statistics

WEIGHTS = {"emotion": 0.51, "story": 0.23, "rhythm": 0.10,
           "eye_trace": 0.07, "plane_2d": 0.05, "space_3d": 0.04}


def score_edl(edl: dict, timeline: dict, weights: dict | None = None) -> dict:
    """Score with Murch's canonical weights, or channel-adjusted weights
    produced by the M5 feedback loop (feedback.aggregate.adjust_weights)."""
    w = weights or WEIGHTS
    events = edl["events"]
    if not events:
        return {"total": 0.0, **{k: 0.0 for k in w}}
    beats = {b["beat_id"]: b for b in timeline["beats"]}
    components = {
        "emotion": _emotion(events, beats),
        "story": _story(events, timeline),
        "rhythm": _rhythm(events),
        "eye_trace": _eye_trace(events),
        "plane_2d": _plane_2d(events),
        "space_3d": _space_3d(events),
    }
    total = sum(w[k] * v for k, v in components.items())
    return {"total": round(total, 4), **{k: round(v, 4) for k, v in components.items()}}


def best_edl(timeline: dict, shotlist: dict, seeds: list[int],
             similarity=None, builder=None,
             weights: dict | None = None) -> tuple[dict, dict]:
    """Build one candidate per seed, score all, return (best_edl, report)."""
    from .edl import build_edl
    builder = builder or build_edl
    candidates = []
    for seed in seeds:
        edl = builder(timeline, shotlist, seed, similarity=similarity)
        candidates.append((score_edl(edl, timeline, weights), edl))
    candidates.sort(key=lambda c: c[0]["total"], reverse=True)
    best_score, best = candidates[0]
    report = {
        "winner_seed": best["seed"],
        "winner_score": best_score,
        "weights": weights or WEIGHTS,
        "candidates": [{"seed": e["seed"], "total": s["total"]}
                       for s, e in candidates],
    }
    return best, report


# --- components -------------------------------------------------------------

def _emotion(events: list[dict], beats: dict) -> float:
    """Duration must track the emotion vector; reveals must be held."""
    scores = []
    for e in events:
        beat = beats.get(e["beat_id"])
        if beat is None:
            continue
        dur = e["video_out"] - e["video_in"]
        emo = beat["emotion"]
        # Target duration: 6s neutral, compressed by tension, stretched by gravity.
        target = 6.0 * (1.0 - 0.35 * emo["tension"] + 0.30 * emo["gravity"])
        scores.append(max(0.0, 1.0 - abs(dur - target) / (2.0 * target)))
    held_reveals, reveals = 0, 0
    for beat in beats.values():
        for sent in beat["sentences"]:
            if sent["is_reveal"]:
                reveals += 1
                if any(e["hold"] for e in events
                       if e["beat_id"] == beat["beat_id"]
                       and e["video_in"] < sent["end"] + sent["pause_after_s"]
                       and e["video_out"] > sent["end"]):
                    held_reveals += 1
    duration_fit = statistics.mean(scores) if scores else 0.0
    reveal_fit = held_reveals / reveals if reveals else 1.0
    return 0.6 * duration_fit + 0.4 * reveal_fit


def _story(events: list[dict], timeline: dict) -> float:
    """Fraction of narration time covered by shots planned for that beat,
    with coverage gaps penalized."""
    covered, total = 0.0, 0.0
    for beat in timeline["beats"]:
        span = beat["end"] - beat["start"]
        total += span
        own = [e for e in events
               if e["beat_id"] == beat["beat_id"] and not e["coverage_gap"]]
        covered += sum(min(e["video_out"], beat["end"]) - max(e["video_in"], beat["start"])
                       for e in own if e["video_out"] > beat["start"]
                       and e["video_in"] < beat["end"])
    gap_penalty = 0.1 * sum(e["coverage_gap"] for e in events)
    return max(0.0, min(1.0, covered / total if total else 0.0) - gap_penalty)


def _rhythm(events: list[dict]) -> float:
    """Cut intervals should vary like breathing: too regular = templated,
    too wild or too fast = chaos."""
    durs = [e["video_out"] - e["video_in"] for e in events]
    if len(durs) < 3:
        return 0.5
    mean = statistics.mean(durs)
    cv = statistics.pstdev(durs) / mean if mean else 0.0  # coefficient of variation
    # Sweet spot around cv ~ 0.45 (varied but not chaotic).
    variation = max(0.0, 1.0 - abs(cv - 0.45) / 0.45)
    flash = sum(d < 1.5 for d in durs) / len(durs)
    return variation * (1.0 - flash)


def _eye_trace(events: list[dict]) -> float:
    if len(events) < 2:
        return 1.0
    jarring = sum(1 for a, b in zip(events, events[1:])
                  if b["cut_style"] == "hard" and a["kind"] == b["kind"]
                  and a["beat_id"] != b["beat_id"])
    return 1.0 - jarring / (len(events) - 1)


def _plane_2d(events: list[dict]) -> float:
    kinds = {e["kind"] for e in events}
    return len(kinds) / 3.0  # document / still / footage


def _space_3d(events: list[dict]) -> float:
    if len(events) < 2:
        return 1.0
    whiplash = sum(1 for a, b in zip(events, events[1:])
                   if a["beat_id"] == b["beat_id"]
                   and (a["source"] == "generated") != (b["source"] == "generated"))
    return 1.0 - whiplash / (len(events) - 1)
