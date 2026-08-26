"""NARA catalog harvester (PLAN.md Layer 4: safest source, with LoC).

Uses the Catalog API v2 (requires free NARA_API_KEY; without it the source
is skipped, not faked). US federal records are PD by statute (17 USC 105),
but the item's useRestriction field is still checked: donated/restricted
materials exist inside NARA and must not be assumed free.
"""

from __future__ import annotations

import os

import requests

from ..ledger import AssetRecord

_BASE = "https://catalog.archives.gov/api/v2"

_TYPE_BY_KIND = {
    "document": "Textual Records",
    "still": "Photographs and other Graphic Materials",
    "footage": "Moving Images",
}


def search(query: str, kind: str, limit: int = 8) -> list[AssetRecord]:
    api_key = os.getenv("NARA_API_KEY", "")
    if not api_key:
        raise RuntimeError("NARA_API_KEY not set (get one free at catalog.archives.gov)")
    resp = requests.get(
        f"{_BASE}/records/search",
        params={"q": query, "limit": limit, "typeOfMaterials": _TYPE_BY_KIND.get(kind, "")},
        headers={"x-api-key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    hits = resp.json().get("body", {}).get("hits", {}).get("hits", [])
    records: list[AssetRecord] = []
    for hit in hits:
        rec = hit.get("_source", {}).get("record", {})
        digital = (rec.get("digitalObjects") or [{}])[0]
        if not digital.get("objectUrl"):
            continue
        use = (rec.get("useRestriction") or {}).get("status", "")
        naid = rec.get("naId", hit.get("_id", ""))
        records.append(AssetRecord(
            asset_id=f"nara:{naid}",
            source="nara",
            kind=kind,
            title=rec.get("title", ""),
            description=(rec.get("scopeAndContentNote") or "")[:1000],
            source_url=f"https://catalog.archives.gov/id/{naid}",
            media_url=digital["objectUrl"],
            # Item-level: "Unrestricted" federal records are PD; anything else
            # falls through the classifier to NEEDS_REVIEW.
            license_raw="public domain (US federal record, unrestricted)"
            if use.lower() in ("unrestricted", "") else use,
            attribution=f"U.S. National Archives, NAID {naid}",
        ))
    return records
