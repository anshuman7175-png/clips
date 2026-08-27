"""Retention post-mortem (PLAN.md Layer 10).

The pipeline generated the edit, so every retention timestamp maps to a
script beat, shot type, and audio state. This module:

1. find_dips()   : locates where viewers leave FASTER than the video's own
                   baseline decay (median per-bucket delta). A dip is excess
                   drop, not absolute drop - long videos always decay.
2. find_spikes() : rewatch bumps (what worked; feeds thumbnails/cold opens).
3. map_dip()     : dip ratio -> seconds -> covering EDL events -> beat,
                   shot kind, source, cut style, audio state.
4. diagnose()    : attributes each mapped dip to Rule-of-Six components so
                   the aggregate layer can adjust the scorer's weights.
5. build_postmortem(): one JSON per video - dips, spikes, exposure stats
                   (runtime share per attribute, needed for lift computation).
"""

from __future__ import annotations

import statistics

# A bucket must drop this much beyond baseline to open/extend a dip region.
MIN_EXCESS_PER_BUCKET = 0.004
# Total excess drop for the region to count as a real dip (audience share).
MIN_DIP_DEPTH = 0.01
# Emotion-modulated target duration, mirrored from rule_of_six._emotion.
_TARGET_BASE_S = 6.0


def find_dips(curve: list[dict], min_depth: float = MIN_DIP_DEPTH) -> list[dict]:
    """Regions where the audience leaves faster than this video's baseline.

    Computed on the running-minimum envelope, and dip regions that begin in
    the recovery window right after a rewatch spike are discarded: both are
    viewers returning to trend, not viewers leaving.
    """
    spike_windows = [(s["ratio"], s["ratio"] + SPIKE_RECOVERY_RATIO)
                     for s in find_spikes(curve)]
    watch, low = [], float("inf")
    for p in curve:
        low = min(low, p["watch"])
        watch.append(low)
    deltas = [b - a for a, b in zip(watch, watch[1:])]
    if not deltas:
        return []
    baseline = statistics.median(deltas)  # the video's own decay rate
    excess = [max(0.0, baseline - d) for d in deltas]  # drop beyond baseline

    dips, i = [], 0
    while i < len(excess):
        if excess[i] <= MIN_EXCESS_PER_BUCKET:
            i += 1
            continue
        j = i
        while j < len(excess) and excess[j] > MIN_EXCESS_PER_BUCKET / 2:
            j += 1
        depth = sum(excess[i:j])
        if depth >= min_depth:
            dips.append({
                "start_ratio": curve[i]["x"],
                "end_ratio": curve[min(j, len(curve) - 1)]["x"],
                "depth": round(depth, 4),
            })
        i = j
    dips.sort(key=lambda d: d["depth"], reverse=True)
    return dips


def find_spikes(curve: list[dict]) -> list[dict]:
    """Buckets where watch ratio RISES against decay = rewatched moments."""
    watch = [p["watch"] for p in curve]
    deltas = [b - a for a, b in zip(watch, watch[1:])]
    if not deltas:
        return []
    baseline = statistics.median(deltas)
    return [{"ratio": curve[i]["x"], "height": round(d - baseline, 4)}
            for i, d in enumerate(deltas)
            if d - baseline > 3 * MIN_EXCESS_PER_BUCKET]


def map_dip(dip: dict, edl: dict, timeline: dict) -> dict:
    """Dip ratios -> seconds -> the EDL events and beats under them."""
    dur = edl["duration_s"]
    t0, t1 = dip["start_ratio"] * dur, dip["end_ratio"] * dur
    t1 = max(t1, t0 + 0.5)  # single-bucket dips still get a window
    events = [e for e in edl["events"]
              if e["video_in"] < t1 and e["video_out"] > t0]
    beats = {b["beat_id"]: b for b in timeline["beats"]}
    beat_ids = sorted({e["beat_id"] for e in events})
    return {
        **dip,
        "t_start": round(t0, 2),
        "t_end": round(t1, 2),
        "event_ids": [e["event_id"] for e in events],
        "beat_ids": beat_ids,
        "attributes": _attributes(events),
        "components": diagnose(events, beats, (t0, t1)),
    }


def diagnose(events: list[dict], beats: dict, window: tuple[float, float]) -> list[str]:
    """Attribute a dip to Rule-of-Six components (why did they leave HERE?)."""
    components: set[str] = set()
    durs = []
    for e in events:
        dur = e["video_out"] - e["video_in"]
        durs.append(dur)
        if e["coverage_gap"]:
            components.add("story")       # narration with no planned picture
        if e["source"] == "generated":
            components.add("space_3d")    # generated shot broke belief
        beat = beats.get(e["beat_id"])
        if beat:
            emo = beat["emotion"]
            target = _TARGET_BASE_S * (1.0 - 0.35 * emo["tension"]
                                       + 0.30 * emo["gravity"])
            if abs(dur - target) > target:   # duration fights the emotion
                components.add("emotion")
    # Jarring hard cut between same-kind shots inside the window.
    for a, b in zip(events, events[1:]):
        if (b["cut_style"] == "hard" and a["kind"] == b["kind"]
                and a["beat_id"] != b["beat_id"]):
            components.add("eye_trace")
    if len(durs) >= 3:
        mean = statistics.mean(durs)
        cv = statistics.pstdev(durs) / mean if mean else 0.0
        if cv < 0.15 or mean < 2.0:      # metronomic or flash-cutting
            components.add("rhythm")
        if len({e["kind"] for e in events}) == 1:
            components.add("plane_2d")   # visually flat stretch
    return sorted(components) or ["unattributed"]


def build_postmortem(video_id: str, curve: list[dict],
                     edl: dict, timeline: dict) -> dict:
    """One post-mortem JSON per video: everything the aggregate layer needs."""
    dips = [map_dip(d, edl, timeline) for d in find_dips(curve)]
    return {
        "video_id": video_id,
        "duration_s": edl["duration_s"],
        "dips": dips,
        "spikes": find_spikes(curve),
        "exposure": exposure_stats(edl),
        "final_watch_ratio": curve[-1]["watch"],
    }


def exposure_stats(edl: dict) -> dict:
    """Runtime seconds per attribute - the denominator for dip lift.

    Without exposure normalization, "documents dip most" is meaningless when
    documents are 70% of the runtime.
    """
    stats: dict[str, float] = {}
    total = 0.0
    for e in edl["events"]:
        dur = e["video_out"] - e["video_in"]
        total += dur
        for key in _attributes([e]):
            stats[key] = stats.get(key, 0.0) + dur
    return {"total_s": round(total, 2),
            "by_attribute": {k: round(v, 2) for k, v in sorted(stats.items())}}


def _attributes(events: list[dict]) -> list[str]:
    """The attribute tags carried by a set of events (kind/source/flags)."""
    attrs: set[str] = set()
    for e in events:
        attrs.add(f"kind:{e['kind']}")
        attrs.add(f"source:{'generated' if e['source'] == 'generated' else 'archival'}")
        if e["coverage_gap"]:
            attrs.add("flag:coverage_gap")
        if e["label"]:
            attrs.add("flag:dramatization")
    return sorted(attrs)
