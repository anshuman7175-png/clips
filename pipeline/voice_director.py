"""Voice Director (PLAN.md Layer 3).

Converts per-beat emotion vectors into per-sentence render parameters for the
TTS engine, plus pause lengths between sentences. This is where "sounds AI"
is defeated: no two sentences render with identical parameters.

Output line schema:
{
  "line_id": "b03-s02", "beat_id": "b03", "text": str,
  "exaggeration": float,   # Chatterbox emotion exaggeration (0.25-1.0)
  "cfg_weight": float,     # Chatterbox pace proxy (lower = slower, 0.3-0.7)
  "pause_after_s": float,  # silence spliced after this sentence
}
"""

from __future__ import annotations

import random
import re


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def direct(script: dict, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    lines: list[dict] = []
    for beat in script["beats"]:
        emo = beat.get("emotion", {})
        tension = float(emo.get("tension", 0.4))
        gravity = float(emo.get("gravity", 0.4))
        pace = float(emo.get("pace", 0.5))
        sentences = split_sentences(beat["text"])
        for i, sent in enumerate(sentences):
            is_reveal = beat["role"] in ("midpoint_reversal", "destabilize") and i == len(sentences) - 1
            # Emotion exaggeration: calm baseline 0.35, tension pushes up, jitter defeats templating.
            exaggeration = min(1.0, 0.35 + 0.45 * tension + rng.uniform(-0.05, 0.05))
            # cfg_weight ~ pacing: gravity and reveals slow down; jitter per sentence.
            cfg_weight = max(0.30, min(0.70, 0.55 - 0.15 * gravity + 0.10 * pace + rng.uniform(-0.04, 0.04)))
            if is_reveal:
                cfg_weight = max(0.30, cfg_weight - 0.10)  # slow into the reveal
            # Pauses: longer after reveals and beat ends; breath-scale jitter everywhere.
            if is_reveal:
                pause = rng.uniform(1.1, 1.6)
            elif i == len(sentences) - 1:
                pause = rng.uniform(0.7, 1.1)
            else:
                pause = rng.uniform(0.25, 0.55) + 0.3 * gravity
            lines.append({
                "line_id": f"{beat['id']}-s{i + 1:02d}",
                "beat_id": beat["id"],
                "text": sent,
                "exaggeration": round(exaggeration, 3),
                "cfg_weight": round(cfg_weight, 3),
                "pause_after_s": round(pause, 2),
            })
    return lines
