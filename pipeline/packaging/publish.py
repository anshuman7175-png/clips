"""Disclosure, persona rule, metadata, upload package (PLAN.md Layer 9).

- build_disclosure(): if ANY event on the timeline is generated/dramatized,
  the YouTube "altered or synthetic content" flag is set. Non-negotiable.
- persona_check(): the narrator persona never gives advice - health, legal,
  or financial phrasing in the script is a publish blocker.
- build_metadata(): title, chaptered description with the citation manifest
  and CC-BY attribution baked in, tags.
- build_upload_package(): one JSON with everything an upload needs;
  upload_youtube() actually performs it (lazy googleapiclient import).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TITLE_MAX = 100          # YouTube hard limit
DESCRIPTION_MAX = 5000   # YouTube hard limit
TAGS_MAX_CHARS = 480     # stay under the 500-char aggregate limit

# The persona narrates documents; it never advises. Any of these in the
# narration is a publish blocker, not a warning.
_ADVICE_PATTERNS = re.compile(
    r"\byou should\b|\bwe recommend\b|\bconsult (a|your) (doctor|physician|"
    r"lawyer|attorney|financial advisor)\b|\bmedical advice\b|\blegal advice\b|"
    r"\binvest(ing)? in\b|\bfinancial advice\b|\btalk to your doctor\b",
    re.IGNORECASE)


def build_disclosure(edl: dict) -> dict:
    """Altered-content disclosure decision, derived from the EDL itself."""
    generated = [e["event_id"] for e in edl["events"] if e["source"] == "generated"]
    labeled = [e["event_id"] for e in edl["events"] if e["label"]]
    return {
        "altered_content": bool(generated),
        "generated_events": generated,
        "labeled_events": labeled,
        "statement": ("Contains labeled dramatized reconstructions created "
                      "with generative tools." if generated else None),
    }


def persona_check(script: dict) -> list[dict]:
    """Return advice-giving violations (empty list = pass)."""
    violations = []
    for beat in script["beats"]:
        for m in _ADVICE_PATTERNS.finditer(beat["text"]):
            violations.append({"beat_id": beat["id"], "phrase": m.group(0)})
    return violations


def build_metadata(case: dict, script: dict, timeline: dict, grounding: dict,
                   license_manifest: dict, disclosure: dict) -> dict:
    """Title + chaptered description + citation manifest + attribution."""
    title = (case.get("title") or _hook_sentence(script))[:TITLE_MAX]

    parts: list[str] = [_hook_sentence(script), ""]

    # Chapters: YouTube needs the first at 00:00 and ascending timestamps.
    parts.append("Chapters:")
    parts.append(f"00:00 {_chapter_label(script['beats'][0])}")
    for beat in timeline["beats"][1:]:
        role = next((b for b in script["beats"] if b["id"] == beat["beat_id"]), None)
        if role:
            parts.append(f"{_ts(beat['start'])} {_chapter_label(role)}")
    parts.append("")

    # Citation manifest: only entailed claims carry a source (Layer 5 output).
    sources = _cited_sources(case, grounding)
    if sources:
        parts.append("Sources:")
        parts += [f"- {s}" for s in sources]
        parts.append("")

    attribution = license_manifest.get("attribution", [])
    cc_by = [a for a in attribution if a["license"] == "CC_BY"]
    if attribution:
        parts.append("Archival material:")
        for a in attribution:
            line = f"- {a['title']} ({a['license']}) {a['source_url']}".strip()
            if a["license"] == "CC_BY" and a.get("attribution"):
                line += f" - {a['attribution']}"
            parts.append(line)
        parts.append("")

    if disclosure["altered_content"]:
        parts.append(disclosure["statement"])

    description = "\n".join(parts).strip()[:DESCRIPTION_MAX]

    return {
        "title": title,
        "description": description,
        "tags": _tags(case, script),
        "category_id": "27",  # Education
        "cc_by_count": len(cc_by),
        "chapters": len(timeline["beats"]),
    }


def build_upload_package(run_dir: Path, video_path: Path, metadata: dict,
                         disclosure: dict, thumbnail_jobs: list[dict],
                         qc_summary: dict) -> Path:
    """Everything an upload needs, in one reviewable JSON. Written even when
    QC failed, so the human sees exactly what is blocking."""
    package = {
        "ready_to_upload": qc_summary["passed"],
        "blockers": qc_summary["blockers"],
        "video_path": str(video_path),
        "metadata": metadata,
        "disclosure": disclosure,
        "thumbnails": [{"concept_id": j["concept_id"], "path": j["out_path"],
                        "hook_text": j.get("hook_text")} for j in thumbnail_jobs],
        "youtube": {
            "snippet": {
                "title": metadata["title"],
                "description": metadata["description"],
                "tags": metadata["tags"],
                "categoryId": metadata["category_id"],
            },
            "status": {
                "privacyStatus": "private",   # human flips to public after taste pass
                "selfDeclaredMadeForKids": False,
                "containsSyntheticMedia": disclosure["altered_content"],
            },
        },
    }
    out = run_dir / "upload_package.json"
    out.write_text(json.dumps(package, indent=2), encoding="utf-8")
    return out


def upload_youtube(package_path: Path, client_secrets: Path) -> str:
    """Perform the upload (lazy import - needs google-api-python-client and
    a one-time OAuth browser flow). Returns the video id."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as err:
        raise RuntimeError(
            "upload needs: pip install google-api-python-client google-auth-oauthlib"
        ) from err
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if not package["ready_to_upload"]:
        raise RuntimeError(f"package not ready: blockers={package['blockers']}")
    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_secrets), ["https://www.googleapis.com/auth/youtube.upload"])
    yt = build("youtube", "v3", credentials=flow.run_local_server(port=0))
    request = yt.videos().insert(
        part="snippet,status",
        body={"snippet": package["youtube"]["snippet"],
              "status": package["youtube"]["status"]},
        media_body=MediaFileUpload(package["video_path"], resumable=True))
    response = request.execute()
    return response["id"]


# --- helpers -----------------------------------------------------------------

def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _hook_sentence(script: dict) -> str:
    return re.split(r"(?<=[.!?])\s", script["beats"][0]["text"])[0]


def _chapter_label(beat: dict) -> str:
    return beat["role"].replace("_", " ").title()


def _cited_sources(case: dict, grounding: dict) -> list[str]:
    used_ids = {c["source_id"] for c in grounding["claims"]
                if c.get("source_id") and c["verdict"] == "entailed"}
    by_id = {s["id"]: s for s in case.get("sources", [])}
    out = []
    for sid in sorted(used_ids):
        src = by_id.get(sid)
        out.append(src["citation"] if src else sid)
    return out


def _tags(case: dict, script: dict) -> list[str]:
    tags = ["documentary", "history", "archival footage", "true story"]
    tags += case.get("tags", [])
    # Proper-noun harvest from the script (cheap, deterministic).
    # Mid-sentence capitals only: a capital after ". " is grammar, not a name.
    text = " ".join(b["text"] for b in script["beats"])
    nouns = re.findall(r"(?<=[a-z,;:] )[A-Z][a-z]{3,}\b", text)
    seen = {t.lower(): i for i, t in enumerate(tags)}
    for n in nouns:
        key = n.lower()
        if key in seen:
            tags[seen[key]] = n  # upgrade casing to the proper-noun form
        else:
            seen[key] = len(tags)
            tags.append(n)
    # Enforce the aggregate char budget.
    total, kept = 0, []
    for t in tags:
        total += len(t) + 1
        if total > TAGS_MAX_CHARS:
            break
        kept.append(t)
    return kept
