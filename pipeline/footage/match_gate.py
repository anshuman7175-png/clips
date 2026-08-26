"""Footage matching gate (PLAN.md Layer 4) - the decision engine.

For every shot request:
1. Embed the query in the right space (SigLIP2 for stills/docs, X-CLIP for video).
2. Search ONLY ledger-usable assets (license whitelist enforced here, again -
   defense in depth: even if a banned asset somehow got embedded, it cannot match).
3. Rank candidates by (footage priority class, cosine score). Priority is
   Layer 0.3 and it NEVER reverses: an archival hit at 0.24 beats a stock
   hit at 0.30 as long as both clear the threshold.
4. Literal-match check for evidentiary shots: at least one literal term must
   appear in the asset's title/description. Embedding similarity alone is not
   evidence - a "period-accurate looking memo" is not THE memo.
5. Decision on failure (threshold or literal miss):
     evidentiary shot  -> "rewrite_line"  (script must bend to reality)
     atmosphere shot   -> "generate"      (Wan 2.2 branch, dramatization label)

Output: shotlist manifest - one entry per shot with decision, matched asset,
license class, and attribution. This manifest is the only thing the edit
engine (M3) is allowed to cut from.
"""

from __future__ import annotations

from .ledger import Ledger

# Layer 0.3: real documents > real archival > stock > generated. Lower = better.
_SOURCE_PRIORITY = {"loc": 0, "nara": 0, "archive_org": 1, "pexels": 2, "pixabay": 2,
                    "generated": 3}

_SPACE_BY_KIND = {"document": "still", "still": "still", "footage": "video"}


def run_gate(shots: list[dict], ledger: Ledger, indexes: dict, embedders: dict,
             thresholds: dict[str, float]) -> dict:
    """indexes/embedders: {"still": ..., "video": ...}. Returns the shotlist manifest."""
    shotlist, rewrites, generation_jobs = [], [], []

    # License whitelist snapshot per kind (defense in depth, see module doc).
    usable_by_kind = {
        kind: {r.asset_id: r for r in ledger.usable_records(kind=kind)}
        for kind in ("document", "still", "footage")
    }

    for shot in shots:
        space = _SPACE_BY_KIND[shot["kind"]]
        threshold = thresholds[space]
        usable = usable_by_kind[shot["kind"]]
        query_vec = embedders[space].embed_text(shot["query"])
        candidates = indexes[space].search(query_vec, k=20, allowed_ids=set(usable))

        best = _pick(candidates, usable, shot, threshold)
        entry = {**shot, "decision": None, "asset_id": None, "score": None,
                 "license_class": None, "attribution": None, "label_required": False}

        if best:
            asset_id, score = best
            rec = usable[asset_id]
            entry.update(decision="matched", asset_id=asset_id, score=round(score, 4),
                         license_class=rec.license_class, attribution=rec.attribution,
                         label_required=rec.dramatization)
        elif shot["evidentiary"]:
            # No real asset proves this line -> the LINE is wrong, not the library.
            entry["decision"] = "rewrite_line"
            rewrites.append({"beat_id": shot["beat_id"], "shot_id": shot["shot_id"],
                             "reason": _fail_reason(candidates, threshold)})
        else:
            entry["decision"] = "generate"
            entry["label_required"] = True  # dramatization label, no exceptions
            generation_jobs.append(shot)
        shotlist.append(entry)

    return {
        "shots": shotlist,
        "rewrite_requests": rewrites,
        "generation_jobs": generation_jobs,
        "stats": {
            "matched": sum(s["decision"] == "matched" for s in shotlist),
            "rewrite_line": len(rewrites),
            "generate": len(generation_jobs),
        },
    }


def _pick(candidates: list[tuple[str, float]], usable: dict, shot: dict,
          threshold: float) -> tuple[str, float] | None:
    """Best candidate by (priority, -score) among those clearing threshold + literal check."""
    viable = []
    for asset_id, score in candidates:
        if score < threshold:
            continue
        rec = usable[asset_id]
        if shot["evidentiary"] and not _literal_ok(rec, shot["literal_terms"]):
            continue
        viable.append((_SOURCE_PRIORITY.get(rec.source, 3), -score, asset_id, score))
    if not viable:
        return None
    viable.sort()
    _, _, asset_id, score = viable[0]
    return asset_id, score


def _literal_ok(rec, literal_terms: list[str]) -> bool:
    if not literal_terms:
        return True
    haystack = f"{rec.title} {rec.description}".lower()
    return any(term in haystack for term in literal_terms)


def _fail_reason(candidates: list[tuple[str, float]], threshold: float) -> str:
    if not candidates:
        return "no usable assets in library for this query"
    best = candidates[0][1]
    if best >= threshold:
        return f"best score {best:.3f} cleared threshold but failed literal-term check"
    return f"best score {best:.3f} below threshold {threshold}"
