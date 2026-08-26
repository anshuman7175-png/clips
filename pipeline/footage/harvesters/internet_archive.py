"""Internet Archive / Prelinger harvester (PLAN.md Layer 4: MIXED source).

The rule from PLAN.md: item-level checks required, and even license-clean
PD films can contain copyrighted MUSIC synced inside them. Therefore:

1. Search via advancedsearch.php, then fetch item METADATA individually -
   only `licenseurl` on the item itself counts as license evidence.
2. Every FOOTAGE item, even when licenseurl is clean, carries a review_note
   ("check embedded music") and is classified NEEDS_REVIEW unless it is a
   silent-era or explicitly music-free item. Stills/documents with a clean
   licenseurl pass normally.
"""

from __future__ import annotations

import requests

from ..ledger import AssetRecord

_SEARCH = "https://archive.org/advancedsearch.php"
_META = "https://archive.org/metadata/"

_MEDIATYPE_BY_KIND = {"document": "texts", "still": "image", "footage": "movies"}


def search(query: str, kind: str, limit: int = 8) -> list[AssetRecord]:
    mediatype = _MEDIATYPE_BY_KIND.get(kind, "image")
    resp = requests.get(_SEARCH, params={
        "q": f"({query}) AND mediatype:({mediatype})",
        "fl[]": ["identifier", "title", "description", "licenseurl"],
        "rows": limit, "output": "json",
    }, timeout=30)
    resp.raise_for_status()
    docs = resp.json().get("response", {}).get("docs", [])

    records: list[AssetRecord] = []
    for doc in docs:
        ident = doc["identifier"]
        # ITEM-LEVEL check: the search index's licenseurl can be stale;
        # the metadata endpoint is authoritative for the item.
        meta = requests.get(_META + ident, timeout=30).json().get("metadata", {})
        license_url = meta.get("licenseurl", "") or doc.get("licenseurl", "")
        rights = meta.get("rights", "") or ""
        media_url = _best_file(ident, kind)

        rec = AssetRecord(
            asset_id=f"archive_org:{ident}",
            source="archive_org",
            kind=kind,
            title=str(doc.get("title", ""))[:300],
            description=str(doc.get("description", ""))[:1000],
            source_url=f"https://archive.org/details/{ident}",
            media_url=media_url,
            license_raw=rights,
            license_url=license_url,
            attribution=f"Internet Archive, {ident}",
        )
        # PLAN.md: PD films may contain copyrighted music -> force human review
        # for footage regardless of the item license.
        if kind == "footage":
            rec.license_raw = ""  # discard the pass; classifier -> NEEDS_REVIEW
            rec.license_url = ""
            rec.review_note = (
                f"IA footage (orig license: {license_url or rights or 'none'}) - "
                "verify no copyrighted music synced in film"
            )
        records.append(rec)
    return records


def _best_file(identifier: str, kind: str) -> str:
    try:
        files = requests.get(_META + identifier + "/files", timeout=30).json().get("result", [])
    except Exception:  # noqa: BLE001
        return ""
    want = (".mp4", ".mpeg") if kind == "footage" else (".jpg", ".png", ".pdf", ".jp2")
    candidates = [f["name"] for f in files if f.get("name", "").lower().endswith(want)]
    if not candidates:
        return ""
    best = max(candidates, key=len)  # crude: derivative names encode resolution
    return f"https://archive.org/download/{identifier}/{best}"
