"""Narration timeline (PLAN.md Layer 6).

Fuses the script (beats + emotion vectors) with the alignment (word/line
timestamps) into the single time-indexed structure the EDL builder cuts
against. Inter-sentence silences above the editorial threshold are promoted
to first-class PAUSE BEATS: the edit must hold the shot and swell ambience
across them instead of cutting through them.
"""

from __future__ import annotations

# A pause this long is an editorial beat (hold + ambience swell), not a gap.
EDITORIAL_PAUSE_S = 0.9

# Roles whose final sentence is a revelation: the edit HOLDS the shot there
# (Layer 6: "revelations -> held shot").
REVEAL_ROLES = {"midpoint_reversal", "destabilize"}


def build_timeline(script: dict, alignment: dict) -> dict:
    lines_by_beat: dict[str, list[dict]] = {}
    for line in alignment["lines"]:
        lines_by_beat.setdefault(line["beat_id"], []).append(line)

    beats = []
    for beat in script["beats"]:
        beat_lines = lines_by_beat.get(beat["id"], [])
        if not beat_lines:
            continue
        emo = beat.get("emotion", {})
        sentences = []
        for i, line in enumerate(beat_lines):
            is_last = i == len(beat_lines) - 1
            sentences.append({
                "line_id": line["line_id"],
                "start": line["start"],
                "end": line["end"],
                "pause_after_s": line["pause_after_s"],
                "is_reveal": is_last and beat["role"] in REVEAL_ROLES,
                "editorial_pause": line["pause_after_s"] >= EDITORIAL_PAUSE_S,
            })
        beats.append({
            "beat_id": beat["id"],
            "role": beat["role"],
            "emotion": {
                "tension": float(emo.get("tension", 0.4)),
                "gravity": float(emo.get("gravity", 0.4)),
                "pace": float(emo.get("pace", 0.5)),
            },
            "start": beat_lines[0]["start"],
            # A beat owns its trailing pause: the next beat's J-cut audio
            # lead eats into it, which is exactly where a J-cut lives.
            "end": round(beat_lines[-1]["end"] + beat_lines[-1]["pause_after_s"], 3),
            "sentences": sentences,
        })

    word_starts = sorted({w["start"] for w in alignment["words"]}
                         | {w["end"] for w in alignment["words"]})
    return {
        "duration_s": alignment["duration_s"],
        "beats": beats,
        "word_boundaries": word_starts,  # snap targets for word-precise cuts
        "pause_beats": [s["line_id"] for b in beats for s in b["sentences"]
                        if s["editorial_pause"]],
    }


def snap(t: float, boundaries: list[float], max_drift_s: float = 0.35) -> float:
    """Snap a cut time to the nearest word boundary (word-level cut precision,
    Layer 6). Never drift more than max_drift_s - rhythm beats precision."""
    if not boundaries:
        return t
    nearest = min(boundaries, key=lambda b: abs(b - t))
    return nearest if abs(nearest - t) <= max_drift_s else t
