"""Cross-video aggregation -> scorer adjustments (PLAN.md Layer 10).

Aggregate findings across videos, then:
- adjust_weights()   : nudge the Rule-of-Six weights toward the components
                       that keep failing. Bounded per cycle (LEARNING_RATE),
                       floored (no component can be silenced), renormalized
                       to sum 1 - the scorer contract never breaks.
- footage_findings() : exposure-normalized dip lift per attribute -> concrete
                       recommendations for the footage-priority logic
                       (e.g. generated shots over-index on dips -> raise the
                       match threshold pain before falling back to Wan).
- channel state      : feedback/channel_state.json persists weights + history;
                       run_milestone3 loads it and every later video is cut
                       with the adjusted scorer. Delete the file to reset.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from ..edit.rule_of_six import WEIGHTS as BASE_WEIGHTS

LEARNING_RATE = 0.10   # max relative weight shift per aggregation cycle
WEIGHT_FLOOR = 0.02    # no Murch component ever drops out entirely
LIFT_THRESHOLD = 1.3   # dip share must exceed exposure share by 30%
MIN_VIDEOS_FOR_LIFT = 2  # one video is an anecdote, not a pattern


def aggregate_postmortems(postmortems: list[dict]) -> dict:
    """Fold N per-video post-mortems into channel-level evidence."""
    component_pressure: Counter = Counter()
    attr_dip_s: Counter = Counter()
    attr_exposure_s: Counter = Counter()
    total_dip_depth = 0.0
    for pm in postmortems:
        for dip in pm["dips"]:
            total_dip_depth += dip["depth"]
            span = max(dip["t_end"] - dip["t_start"], 0.5)
            for comp in dip["components"]:
                component_pressure[comp] += dip["depth"]
            for attr in dip["attributes"]:
                attr_dip_s[attr] += span
        for attr, secs in pm["exposure"]["by_attribute"].items():
            attr_exposure_s[attr] += secs

    total_dip_s = sum(attr_dip_s.values()) or 1.0
    total_exp_s = sum(attr_exposure_s.values()) or 1.0
    lift = {}
    for attr, exp_s in attr_exposure_s.items():
        exp_share = exp_s / total_exp_s
        dip_share = attr_dip_s.get(attr, 0.0) / total_dip_s
        if exp_share > 0:
            lift[attr] = round(dip_share / exp_share, 3)

    return {
        "videos": [pm["video_id"] for pm in postmortems],
        "n_videos": len(postmortems),
        "total_dip_depth": round(total_dip_depth, 4),
        "component_pressure": {k: round(v, 4)
                               for k, v in component_pressure.most_common()},
        "attribute_lift": dict(sorted(lift.items(),
                                      key=lambda kv: kv[1], reverse=True)),
    }


def adjust_weights(current: dict, aggregate: dict,
                   learning_rate: float = LEARNING_RATE) -> dict:
    """Shift weight TOWARD the components implicated in dips.

    A component that keeps failing needs a larger share of the scorer's
    attention so candidate EDLs weak on it stop winning. Unattributed
    pressure adjusts nothing.
    """
    pressure = {k: v for k, v in aggregate["component_pressure"].items()
                if k in current}
    total_p = sum(pressure.values())
    if not total_p:
        return {k: round(v, 4) for k, v in current.items()}
    adjusted = {k: w * (1.0 + learning_rate * pressure.get(k, 0.0) / total_p)
                for k, w in current.items()}
    # Renormalize with a floor: sums to 1, nothing silenced.
    s = sum(adjusted.values())
    adjusted = {k: max(WEIGHT_FLOOR, v / s) for k, v in adjusted.items()}
    s = sum(adjusted.values())
    return {k: round(v / s, 4) for k, v in adjusted.items()}


def footage_findings(aggregate: dict) -> list[dict]:
    """Exposure-normalized lift -> footage-priority recommendations."""
    findings = []
    if aggregate["n_videos"] < MIN_VIDEOS_FOR_LIFT:
        return [{"finding": f"only {aggregate['n_videos']} video(s) analyzed",
                 "action": "collect more post-mortems before changing "
                           "footage-priority logic"}]
    lift = aggregate["attribute_lift"]
    if lift.get("source:generated", 0.0) > LIFT_THRESHOLD:
        findings.append({
            "finding": f"generated shots over-index on dips "
                       f"(lift {lift['source:generated']}x)",
            "action": "prefer rewrite_line over generate at the match gate; "
                      "lower thresholds only with real archival growth"})
    if lift.get("flag:coverage_gap", 0.0) > LIFT_THRESHOLD:
        findings.append({
            "finding": f"coverage gaps over-index on dips "
                       f"(lift {lift['flag:coverage_gap']}x)",
            "action": "treat coverage gaps as harvest failures - widen "
                      "harvest queries before the edit, not after"})
    for attr, value in lift.items():
        if attr.startswith("kind:") and value > LIFT_THRESHOLD * 1.2:
            findings.append({
                "finding": f"{attr} shots over-index on dips (lift {value}x)",
                "action": f"cap consecutive {attr.split(':')[1]} runtime; "
                          "interleave other shot kinds"})
    return findings


# --- channel state persistence -----------------------------------------------

def write_channel_state(path: Path, aggregate: dict, weights: dict,
                        findings: list[dict]) -> Path:
    """Persist weights + findings; append this cycle to the history."""
    prior = _read(path)
    history = prior.get("history", [])
    history.append({"ts": int(time.time()), "n_videos": aggregate["n_videos"],
                    "weights": weights})
    state = {
        "base_weights": BASE_WEIGHTS,
        "weights": weights,
        "aggregate": aggregate,
        "findings": findings,
        "history": history[-50:],  # bounded
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_channel_weights(path: Path) -> dict | None:
    """Adjusted Rule-of-Six weights, or None if no feedback has landed yet."""
    state = _read(path)
    weights = state.get("weights")
    if weights and set(weights) == set(BASE_WEIGHTS):
        return weights
    return None


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
