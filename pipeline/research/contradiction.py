"""Contradiction detection (PLAN.md Layer 1) - frontier LLM, FEW calls.

Finds where primary sources disagree. The sharpest contradiction becomes the
case file's `contradiction` field, which fuels both the midpoint reversal
(script engine) and the novelty gate (a documented disagreement is exactly
the kind of substantive point existing coverage tends to miss).

Frontier calls are scarce, so pairing is budgeted: documents are compared
pairwise up to MAX_PAIRS, most-substantial documents first.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

from ..config import Config
from ..llm import chat_json

Judge = Callable[[dict, dict], dict]

MAX_PAIRS = 6  # frontier-call budget per case

_JUDGE_SYSTEM = """You compare two primary-source documents about the same case \
for a documentary research pipeline. Decide whether they materially DISAGREE on \
a fact (dates, causes, actors, official findings) - not mere emphasis. Return \
JSON: {"contradicts": bool, "point": "one sentence naming the exact disagreement, \
citing what each source claims", "severity": 0.0-1.0}."""


def llm_judge(cfg: Config) -> Judge:
    """Default judge: frontier endpoint (quality-critical, low-volume)."""
    def judge(a: dict, b: dict) -> dict:
        user = (f"SOURCE A ({a['citation']}):\n{a['text'][:4000]}\n\n"
                f"SOURCE B ({b['citation']}):\n{b['text'][:4000]}")
        return chat_json(cfg.frontier, _JUDGE_SYSTEM, user, temperature=0.2)
    return judge


def find_contradictions(docs: list[dict], judge: Judge,
                        max_pairs: int = MAX_PAIRS) -> list[dict]:
    """Budgeted pairwise comparison; returns disagreements, sharpest first."""
    # Longest documents first: substance up front when the budget truncates.
    ordered = sorted(docs, key=lambda d: -len(d["text"]))
    out = []
    for a, b in list(combinations(ordered, 2))[:max_pairs]:
        verdict = judge(a, b) or {}
        if verdict.get("contradicts"):
            out.append({
                "source_a": a["id"], "source_b": b["id"],
                "point": (verdict.get("point") or "").strip(),
                "severity": max(0.0, min(1.0, float(verdict.get("severity", 0.5)))),
            })
    return sorted(out, key=lambda c: -c["severity"])


def select_reversal(contradictions: list[dict]) -> dict | None:
    """The sharpest documented disagreement = the midpoint reversal fuel."""
    return contradictions[0] if contradictions else None
