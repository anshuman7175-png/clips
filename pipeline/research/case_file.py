"""Case file assembly (PLAN.md Layer 1 -> Milestone 1 handoff).

Produces cases/<slug>/case.json with EXACTLY the schema the script engine
consumes (title, contradiction, sources[{id, citation, text}]), plus the
research provenance (timeline, tags, novelty verdict) that later layers and
the human reviewer can inspect.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MAX_SOURCES = 6  # the script engine grounds against a handful of deep sources


def build_case_file(title: str, docs: list[dict], reversal: dict | None,
                    graph: dict, novelty: dict) -> dict:
    """Assemble the M1-compatible case dict. Longest documents first - the
    grounding layer entails claims against source TEXT, so depth wins."""
    ordered = sorted(docs, key=lambda d: -len(d["text"]))[:MAX_SOURCES]
    id_map = {d["id"]: f"src{i}" for i, d in enumerate(ordered, 1)}
    case = {
        "title": title,
        "contradiction": (reversal or {}).get("point", ""),
        "sources": [{
            "id": id_map[d["id"]],
            "citation": d["citation"],
            "text": d["text"],
        } for d in ordered],
        # provenance (ignored by M1, used by packaging + human review)
        "contradiction_sources": [
            id_map[sid] for sid in
            ((reversal or {}).get("source_a"), (reversal or {}).get("source_b"))
            if sid in id_map],
        "tags": [e["name"].lower() for e in graph["entities"][:4]],
        "timeline": [{"date": ev["date"], "description": ev["description"]}
                     for ev in graph["timeline"] if ev["date"]],
        "novelty": {
            "passed": novelty["passed"],
            "novel_points": [f["fact"] for f in novelty["novel_points"]],
            "existing_videos": novelty["existing_videos"],
        },
    }
    return case


def write_case_file(case: dict, cases_dir: Path) -> Path:
    out_dir = cases_dir / slug(case["title"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "case.json"
    path.write_text(json.dumps(case, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:60] or "case"
