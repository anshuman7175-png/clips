"""Script draft + adversarial rewrite (frontier LLM, PLAN.md Layer 2).

Two quality-critical frontier calls per video:
1. draft_script()       - writes the script FROM primary source excerpts.
2. adversarial_rewrite() - attacks the draft for cliches, AI-isms, padding.
"""

from __future__ import annotations

import json

from ..config import Config
from ..llm import chat_json
from .structure import BAN_LIST

BEAT_SCHEMA = """
Return JSON: {"title": str, "beats": [{
  "id": "b01", "act": 1|2|3,
  "role": "cold_open"|"establish"|"destabilize"|"investigate"|"midpoint_reversal"|"synthesis"|"honest_ending",
  "text": "narration text (2-6 sentences)",
  "loop": {"action": "plant"|"payoff", "loop_id": "L1"} or null,
  "emotion": {"tension": 0-1, "wonder": 0-1, "gravity": 0-1, "pace": 0-1}
}]}
"""

DRAFT_SYSTEM = f"""You are a documentary scriptwriter for a long-form mystery channel.
You write ONLY from the primary source excerpts provided. Never invent facts.
Structural contract (violations are auto-rejected by a validator):
- First beat: role=cold_open. Its FIRST SENTENCE is the most arresting verified fact.
  No greetings, no "today we'll explore", no rhetorical warm-up.
- Act 1 (~15-20% of runtime): cold open -> establish person/place -> destabilizing event.
- Act 2 (~55-60%): investigation. Exactly ONE beat with role=midpoint_reversal where the
  obvious explanation breaks, built from a real contradiction in the sources.
- Act 3 (~20-25%): synthesis, then role=honest_ending: state plainly what the documents
  can and cannot say. NO fake resolution.
- Open loops: plant/payoff a loop at least every 3 minutes of narration. Every planted
  loop MUST be paid off. At most ONE deliberate misdirect, and it must be resolved.
- Banned phrases (never use): {", ".join(BAN_LIST)}.
- Tone: restrained, precise, concrete. Let facts carry tension. Short sentences on reveals.
{BEAT_SCHEMA}"""

REWRITE_SYSTEM = f"""You are a ruthless script editor. Attack this documentary script for:
- AI-isms and cliches (banned: {", ".join(BAN_LIST)})
- padding, repetition, unearned claims, vague sentences
- weak cold open (first sentence must be a concrete verified fact)
- limp midpoint reversal
Rewrite it beat-by-beat. Keep the same beat structure, ids, roles, loops, and facts.
Only improve the prose. Preserve every factual claim exactly.
{BEAT_SCHEMA}"""


def draft_script(cfg: Config, case: dict) -> dict:
    target_words = int(cfg.target_minutes * cfg.words_per_minute)
    user = (
        f"CASE: {case['title']}\n"
        f"TARGET LENGTH: ~{target_words} words of narration total.\n\n"
        f"KNOWN CONTRADICTION IN SOURCES (use for midpoint_reversal):\n{case.get('contradiction', 'none provided - find one in the sources')}\n\n"
        "PRIMARY SOURCE EXCERPTS:\n"
        + "\n\n".join(f"[{s['id']}] ({s['citation']})\n{s['text']}" for s in case["sources"])
    )
    return chat_json(cfg.frontier, DRAFT_SYSTEM, user, temperature=0.8)


def adversarial_rewrite(cfg: Config, script: dict) -> dict:
    return chat_json(cfg.frontier, REWRITE_SYSTEM, json.dumps(script, ensure_ascii=False), temperature=0.6)
