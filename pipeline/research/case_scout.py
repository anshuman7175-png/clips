"""Case scout: entity/timeline graphs from harvested sources (PLAN.md Layer 1).

Turns a pile of primary-source documents into structure the rest of the
pipeline can reason about:

- entities : who/what appears, merged across documents (ranked by spread)
- timeline : dated events in chronological order
- edges    : entity <-> event co-mentions

Extraction is a local-LLM call per document (mechanical, high-volume ->
LOCAL_* endpoint), injectable for offline tests. A deterministic date parser
backstops the LLM so the timeline never silently loses dated facts.
"""

from __future__ import annotations

import re
from typing import Callable

from ..config import Config
from ..llm import chat_json

Extractor = Callable[[str], dict]

_MONTHS = {m: i + 1 for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"))}
_MONTH_RE = "|".join(m.title() for m in _MONTHS)

_DATE_PATTERNS = (
    # "24 October 1980"
    (re.compile(rf"\b(\d{{1,2}}) ({_MONTH_RE}) (\d{{4}})\b"),
     lambda m: (int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))),
    # "October 24, 1980"
    (re.compile(rf"\b({_MONTH_RE}) (\d{{1,2}}), (\d{{4}})\b"),
     lambda m: (int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))),
    # ISO "1980-10-24"
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
     lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3)))),
)

_EXTRACT_SYSTEM = """You extract structured facts from primary-source documents \
for a documentary research pipeline. Return JSON: \
{"entities": [{"name": str, "type": "person|organization|place|vessel|document|other"}], \
"events": [{"date": "YYYY-MM-DD or empty", "description": one-sentence factual event}]}. \
Only facts stated in the document. No speculation."""


def parse_dates(text: str) -> list[dict]:
    """Deterministic dated-sentence extraction: [{"date", "context"}]."""
    hits = []
    for pattern, to_ymd in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            y, mo, d = to_ymd(m)
            hits.append({"date": f"{y:04d}-{mo:02d}-{d:02d}",
                         "context": _sentence_around(text, m.start())})
    return hits


def llm_extractor(cfg: Config) -> Extractor:
    """Default extractor: local LLM (high-volume, mechanical)."""
    def extract(text: str) -> dict:
        return chat_json(cfg.local, _EXTRACT_SYSTEM, text[:6000], temperature=0.1)
    return extract


def build_graph(docs: list[dict], extractor: Extractor) -> dict:
    """Entity/timeline graph across all documents."""
    entities: dict[str, dict] = {}
    events: list[dict] = []
    for doc in docs:
        ext = extractor(doc["text"]) or {}
        for ent in ext.get("entities", []):
            name = (ent.get("name") or "").strip()
            if not name:
                continue
            e = entities.setdefault(name.lower(), {
                "name": name, "type": ent.get("type", "other"), "doc_ids": []})
            if doc["id"] not in e["doc_ids"]:
                e["doc_ids"].append(doc["id"])
        seen_dates: set[str] = set()
        for ev in ext.get("events", []):
            desc = (ev.get("description") or "").strip()
            if not desc:
                continue
            date = (ev.get("date") or "").strip()
            if date:
                seen_dates.add(date)
            events.append({"date": date, "description": desc, "doc_id": doc["id"]})
        # Deterministic backstop: dated sentences the LLM missed still make
        # the timeline (dates anchor the three-act spine and the reversal).
        for hit in parse_dates(doc["text"]):
            if hit["date"] not in seen_dates:
                events.append({"date": hit["date"], "description": hit["context"],
                               "doc_id": doc["id"]})
                seen_dates.add(hit["date"])

    dated = sorted((e for e in events if e["date"]), key=lambda e: e["date"])
    undated = [e for e in events if not e["date"]]
    timeline = dated + undated

    ranked = sorted(entities.values(),
                    key=lambda e: (-len(e["doc_ids"]), e["name"].lower()))
    edges = [{"entity": ent["name"], "event_index": i}
             for ent in ranked
             for i, ev in enumerate(timeline)
             if ent["name"].lower() in ev["description"].lower()]
    return {"entities": ranked, "timeline": timeline, "edges": edges}


def _sentence_around(text: str, pos: int) -> str:
    start = max(text.rfind(".", 0, pos), text.rfind("!", 0, pos),
                text.rfind("?", 0, pos)) + 1
    end_candidates = [i for i in (text.find(".", pos), text.find("!", pos),
                                  text.find("?", pos)) if i != -1]
    end = min(end_candidates) + 1 if end_candidates else len(text)
    return text[start:end].strip()
