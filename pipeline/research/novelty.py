"""Novelty gate (PLAN.md Layer 1).

A case only enters production if it can BEAT existing coverage on at least
MIN_NOVEL_POINTS substantive points. "Substantive points" are the extracted
facts: dated timeline events plus documented source contradictions. Each is
checked against transcripts of existing top videos on the topic; a fact
already told by an existing video is covered, everything else is novel.

Coverage is a pure lexical-entailment heuristic (content-word overlap) so
the gate runs offline and deterministically; an NLI callable can be injected
when the GPU box is available.
"""

from __future__ import annotations

import re
from typing import Callable

MIN_NOVEL_POINTS = 3     # the plan's "at least 3 substantive points"
COVERED_OVERLAP = 0.60   # fraction of a fact's content words found in a transcript

_STOPWORDS = frozenset(
    "the a an and or but of to in on at by for with from into during was were "
    "is are be been being that this these those it its his her their our as "
    "had has have not no never after before about over under between which "
    "who whom whose when where while than then them they he she we you i".split())

Coverage = Callable[[str, str], bool]


def extract_facts(graph: dict, contradictions: list[dict]) -> list[dict]:
    """Substantive points: dated events first, then documented disagreements."""
    facts, seen = [], set()

    def add(text: str, kind: str, ref: str) -> None:
        key = " ".join(_content_words(text))
        if key and key not in seen:
            seen.add(key)
            facts.append({"fact": text, "kind": kind, "ref": ref})

    for ev in graph["timeline"]:
        if ev["date"]:  # undated events are too soft to count as "points"
            add(f"{ev['date']}: {ev['description']}", "timeline", ev["doc_id"])
    for c in contradictions:
        add(c["point"], "contradiction", f"{c['source_a']}|{c['source_b']}")
    return facts


def fact_covered(fact: str, transcript: str) -> bool:
    """Default coverage heuristic: most of the fact's content words appear."""
    words = set(_content_words(fact))
    if not words:
        return True
    hits = words & set(_content_words(transcript))
    return len(hits) / len(words) >= COVERED_OVERLAP


def novelty_gate(facts: list[dict], transcripts: list[str],
                 min_points: int = MIN_NOVEL_POINTS,
                 covered: Coverage = fact_covered) -> dict:
    """Pass/fail + the evidence for the decision. No transcripts = untouched
    topic, everything is novel by definition."""
    novel, already = [], []
    for f in facts:
        if any(covered(f["fact"], t) for t in transcripts):
            already.append(f)
        else:
            novel.append(f)
    return {
        "passed": len(novel) >= min_points,
        "required": min_points,
        "novel_points": novel,
        "covered_points": already,
        "existing_videos": len(transcripts),
    }


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower())
            if len(w) > 2 and w not in _STOPWORDS]
