"""Library of Congress harvester (PLAN.md Layer 4: safest source).

Uses the loc.gov JSON API. For footage we constrain to the National
Screening Room collection (curated, overwhelmingly PD); for stills and
documents we search photos/manuscripts formats. Rights are still read
PER ITEM from the `rights` field - "safest source" is a prior, not a pass.
"""

from __future__ import annotations

import requests

from ..ledger import AssetRecord

_BASE = "https://www.loc.gov"

_FORMAT_BY_KIND = {
    "document": "manuscripts",
    "still": "photos",
    "footage": "film-and-videos",
}


def search(query: str, kind: str, limit: int = 8) -> list[AssetRecord]:
    params = {"q": query, "fo": "json", "c": limit, "at": "results"}
    path = f"/{_FORMAT_BY_KIND.get(kind, 'photos')}/"
    if kind == "footage":
        params["partof"] = "national screening room"
    resp = requests.get(_BASE + path, params=params, timeout=30)
    resp.raise_for_status()
    records: list[AssetRecord] = []
    for item in resp.json().get("results", []):
        if item.get("access_restricted"):
            continue
        media_url = _best_media(item, kind)
        if not media_url:
            continue
        rights = item.get("rights") or ""
        if isinstance(rights, list):
            rights = " ".join(rights)
        records.append(AssetRecord(
            asset_id=f"loc:{item.get('id', item.get('url', '')).rstrip('/').rsplit('/', 1)[-1]}",
            source="loc",
            kind=kind,
            title=item.get("title", ""),
            description=" ".join(item.get("description") or [])[:1000],
            source_url=item.get("url", ""),
            media_url=media_url,
            license_raw=rights or "no known restrictions",  # LoC default statement
            attribution=f"Library of Congress, {item.get('title', '')}",
        ))
    return records


def _best_media(item: dict, kind: str) -> str:
    """Prefer the highest-resolution non-thumbnail resource."""
    urls = item.get("image_url") or []
    if kind == "footage":
        for res in item.get("resources") or []:
            if res.get("video"):
                return res["video"]
    return urls[-1] if urls else ""
