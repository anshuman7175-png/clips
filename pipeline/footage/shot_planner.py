"""Shot planner: script beats -> concrete visual queries (PLAN.md Layer 4).

Local-LLM pass (cheap, high-volume, same tier as claim decomposition).
Each beat yields 1-3 shot requests. A shot request is a CONTRACT the
match gate can enforce:

{
  "shot_id": "b03-s1",
  "beat_id": "b03",
  "kind": "document" | "still" | "footage",
  "query": "typed FBI memo dated March 1954 with redactions",
  "literal_terms": ["memo", "1954"],   # must appear in the matched asset's metadata
  "evidentiary": true,                 # true => generation is FORBIDDEN for this shot
  "duration_s": 6.0
}

`evidentiary` encodes Layer 0.3 + Layer 4: a shot that presents evidence
(a document, a named person, a specific place at a specific time) may only
be satisfied by a real asset - on failure the DECISION is to rewrite the
script line, never to generate. Atmosphere shots may fall through to Wan.
"""

from __future__ import annotations

import json

from ..config import Config
from ..llm import chat_json

PLAN_SYSTEM = """You are a documentary footage researcher. For each narration
beat, plan 1-3 shots. For each shot decide:
- kind: "document" (paper/records/maps), "still" (photo), or "footage" (moving image)
- query: a literal, visual search phrase describing what must be on screen
- literal_terms: 1-3 words that MUST appear in a matching asset's title/description
- evidentiary: true if the shot presents evidence (specific documents, named people,
  real places/events). false only for pure atmosphere (weather, textures, generic
  night roads). Be strict: when unsure, evidentiary=true.
- duration_s: seconds on screen (3-10)
Return JSON: {"shots": [{"beat_id": str, "kind": str, "query": str,
"literal_terms": [str], "evidentiary": bool, "duration_s": float}]}"""

_VALID_KINDS = {"document", "still", "footage"}


def plan_shots(cfg: Config, script: dict) -> list[dict]:
    shots: list[dict] = []
    for beat in script["beats"]:
        out = chat_json(cfg.local, PLAN_SYSTEM, json.dumps(
            {"beat_id": beat["id"], "text": beat["text"], "role": beat["role"]}))
        for i, shot in enumerate(out.get("shots", [])):
            shots.append(_sanitize(shot, beat["id"], i + 1))
    return shots


def _sanitize(shot: dict, beat_id: str, n: int) -> dict:
    """Enforce the shot-request contract regardless of LLM sloppiness."""
    kind = shot.get("kind") if shot.get("kind") in _VALID_KINDS else "still"
    terms = [str(t).lower() for t in (shot.get("literal_terms") or [])][:3]
    return {
        "shot_id": f"{beat_id}-s{n}",
        "beat_id": beat_id,
        "kind": kind,
        "query": str(shot.get("query", ""))[:300],
        "literal_terms": terms,
        # Fail-safe default: unlabeled shots are treated as evidentiary.
        "evidentiary": bool(shot.get("evidentiary", True)),
        "duration_s": max(3.0, min(10.0, float(shot.get("duration_s", 5.0)))),
    }
