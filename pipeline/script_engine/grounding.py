"""Claim-level verification contract (PLAN.md Layer 5).

Local-LLM (cheap, high-volume) passes:
1. decompose_claims()  - split each beat into atomic factual claims.
2. ground_claims()     - match every claim to a source chunk + NLI verdict.

Output: a citation manifest. Any claim with verdict != "entailed" is flagged;
"contradicted" claims fail the run.
"""

from __future__ import annotations

import json

from ..config import Config
from ..llm import chat_json

DECOMPOSE_SYSTEM = """Split narration text into atomic factual claims.
An atomic claim is a single checkable statement of fact (who/what/when/where).
Skip pure rhetoric, questions, and transitions.
Return JSON: {"claims": [{"claim_id": "c01", "beat_id": str, "text": str}]}"""

GROUND_SYSTEM = """You are a fact verifier. For each claim, find the single best
supporting source excerpt and give an NLI verdict:
- "entailed": the excerpt directly supports the claim
- "neutral": related but does not confirm the claim
- "contradicted": the excerpt conflicts with the claim
Return JSON: {"results": [{"claim_id": str, "source_id": str or null,
"quote": "exact supporting sentence(s) or null", "verdict": "entailed"|"neutral"|"contradicted"}]}"""


def decompose_claims(cfg: Config, script: dict) -> list[dict]:
    claims: list[dict] = []
    for beat in script["beats"]:
        out = chat_json(cfg.local, DECOMPOSE_SYSTEM,
                        json.dumps({"beat_id": beat["id"], "text": beat["text"]}))
        for i, c in enumerate(out.get("claims", [])):
            c["claim_id"] = f"{beat['id']}-c{i + 1:02d}"
            c["beat_id"] = beat["id"]
            claims.append(c)
    return claims


def ground_claims(cfg: Config, claims: list[dict], sources: list[dict]) -> dict:
    src_block = "\n\n".join(f"[{s['id']}] ({s['citation']})\n{s['text']}" for s in sources)
    manifest = {"claims": [], "flagged": [], "contradicted": []}
    # Batch in groups of 10 to keep local-model context small.
    for i in range(0, len(claims), 10):
        batch = claims[i : i + 10]
        out = chat_json(
            cfg.local, GROUND_SYSTEM,
            "SOURCES:\n" + src_block + "\n\nCLAIMS:\n"
            + json.dumps([{"claim_id": c["claim_id"], "text": c["text"]} for c in batch]),
        )
        by_id = {c["claim_id"]: c for c in batch}
        for r in out.get("results", []):
            entry = {**by_id.get(r["claim_id"], {}), **r}
            manifest["claims"].append(entry)
            if r["verdict"] == "contradicted":
                manifest["contradicted"].append(r["claim_id"])
            elif r["verdict"] != "entailed" or not r.get("source_id"):
                manifest["flagged"].append(r["claim_id"])
    return manifest
