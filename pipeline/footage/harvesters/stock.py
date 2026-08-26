"""Stock harvesters: Pexels + Pixabay (PLAN.md Layer 4).

Stock is priority 3 of 4 (below real documents/archival, above generated).
Both licenses allow commercial use without attribution; the ledger still
records the license string per item so the manifest stays item-level.
"""

from __future__ import annotations

import os

import requests

from ..ledger import AssetRecord


def pexels_search(query: str, kind: str, limit: int = 8) -> list[AssetRecord]:
    api_key = os.getenv("PEXELS_API_KEY", "")
    if not api_key:
        raise RuntimeError("PEXELS_API_KEY not set")
    if kind == "document":
        return []  # stock never satisfies a document request (Layer 0.3)
    endpoint = ("https://api.pexels.com/videos/search" if kind == "footage"
                else "https://api.pexels.com/v1/search")
    resp = requests.get(endpoint, params={"query": query, "per_page": limit},
                        headers={"Authorization": api_key}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    records: list[AssetRecord] = []
    for item in data.get("videos" if kind == "footage" else "photos", []):
        if kind == "footage":
            files = sorted(item.get("video_files", []), key=lambda f: f.get("height") or 0)
            media = files[-1]["link"] if files else ""
        else:
            media = item.get("src", {}).get("original", "")
        if not media:
            continue
        records.append(AssetRecord(
            asset_id=f"pexels:{item['id']}",
            source="pexels",
            kind=kind,
            title=item.get("alt") or item.get("url", "").rstrip("/").rsplit("/", 1)[-1],
            source_url=item.get("url", ""),
            media_url=media,
            license_raw="Pexels License",
            attribution=f"Pexels / {item.get('photographer') or item.get('user', {}).get('name', '')}",
        ))
    return records


def pixabay_search(query: str, kind: str, limit: int = 8) -> list[AssetRecord]:
    api_key = os.getenv("PIXABAY_API_KEY", "")
    if not api_key:
        raise RuntimeError("PIXABAY_API_KEY not set")
    if kind == "document":
        return []
    endpoint = ("https://pixabay.com/api/videos/" if kind == "footage"
                else "https://pixabay.com/api/")
    resp = requests.get(endpoint, params={"key": api_key, "q": query, "per_page": limit},
                        timeout=30)
    resp.raise_for_status()
    records: list[AssetRecord] = []
    for item in resp.json().get("hits", []):
        if kind == "footage":
            videos = item.get("videos", {})
            media = (videos.get("large") or videos.get("medium") or {}).get("url", "")
        else:
            media = item.get("largeImageURL", "")
        if not media:
            continue
        records.append(AssetRecord(
            asset_id=f"pixabay:{item['id']}",
            source="pixabay",
            kind=kind,
            title=item.get("tags", ""),
            source_url=item.get("pageURL", ""),
            media_url=media,
            license_raw="Pixabay Content License",
            attribution=f"Pixabay / {item.get('user', '')}",
        ))
    return records
