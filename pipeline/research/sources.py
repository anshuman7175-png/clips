"""Primary source harvesters (PLAN.md Layer 1) - all free APIs.

Each client returns normalized source-doc dicts:

    {"id", "api", "citation", "url", "date", "text"}

`text` is the document's own words (OCR, opinion text, report abstract),
never a paraphrase - Layer 5 grounding entails script claims against these
exact strings, so a summary here would poison the whole trust chain.

Every client is best-effort: a dead API must never kill a research run,
so harvest_all() isolates failures per source.
"""

from __future__ import annotations

import re

import requests

_TIMEOUT = 30
_UA = {"User-Agent": "documentary-research-pipeline/1.0 (archival research)"}


def courtlistener(query: str, limit: int = 5) -> list[dict]:
    """CourtListener/RECAP opinion search (free, no key for search)."""
    resp = requests.get(
        "https://www.courtlistener.com/api/rest/v4/search/",
        params={"q": query, "type": "o", "order_by": "score desc"},
        headers=_UA, timeout=_TIMEOUT)
    resp.raise_for_status()
    docs = []
    for r in resp.json().get("results", [])[:limit]:
        year = (r.get("dateFiled") or "")[:4]
        docs.append({
            "id": f"courtlistener:{r.get('cluster_id', r.get('id', ''))}",
            "api": "courtlistener",
            "citation": f"{r.get('caseName', 'Unknown case')}, "
                        f"{r.get('court', '')} ({year})".strip(),
            "url": "https://www.courtlistener.com" + (r.get("absolute_url") or ""),
            "date": r.get("dateFiled") or "",
            "text": _clean(r.get("snippet") or ""),
        })
    return [d for d in docs if d["text"]]


def chronicling_america(query: str, limit: int = 5) -> list[dict]:
    """Chronicling America newspaper page OCR (LoC, public domain)."""
    resp = requests.get(
        "https://chroniclingamerica.loc.gov/search/pages/results/",
        params={"andtext": query, "format": "json", "rows": limit},
        headers=_UA, timeout=_TIMEOUT)
    resp.raise_for_status()
    docs = []
    for item in resp.json().get("items", [])[:limit]:
        docs.append({
            "id": f"chronam:{item.get('id', '').strip('/').replace('/', '_')}",
            "api": "chronicling_america",
            "citation": f"{item.get('title', 'Unknown paper')}, "
                        f"{item.get('date', '')} (Chronicling America)",
            "url": "https://chroniclingamerica.loc.gov" + item.get("id", ""),
            "date": _iso(item.get("date", "")),
            "text": _clean(item.get("ocr_eng") or "")[:4000],
        })
    return [d for d in docs if d["text"]]


def loc(query: str, limit: int = 5) -> list[dict]:
    """Library of Congress general search (loc.gov JSON API)."""
    resp = requests.get(
        "https://www.loc.gov/search/",
        params={"q": query, "fo": "json", "c": limit, "at": "results"},
        headers=_UA, timeout=_TIMEOUT)
    resp.raise_for_status()
    docs = []
    for item in resp.json().get("results", [])[:limit]:
        desc = item.get("description") or []
        text = _clean(" ".join(desc if isinstance(desc, list) else [desc]))
        docs.append({
            "id": f"loc:{(item.get('id') or item.get('url', '')).rstrip('/').rsplit('/', 1)[-1]}",
            "api": "loc",
            "citation": f"Library of Congress, {item.get('title', '')}",
            "url": item.get("url", ""),
            "date": item.get("date", "") or "",
            "text": text[:4000],
        })
    return [d for d in docs if d["text"]]


def nara(query: str, limit: int = 5) -> list[dict]:
    """NARA catalog search (catalog.archives.gov v1, no key)."""
    resp = requests.get(
        "https://catalog.archives.gov/api/v1",
        params={"q": query, "rows": limit},
        headers=_UA, timeout=_TIMEOUT)
    resp.raise_for_status()
    rows = (resp.json().get("opaResponse", {}).get("results", {})
            .get("result", []))
    docs = []
    for r in rows[:limit]:
        item = r.get("description", {}).get("item", {}) or \
            r.get("description", {}).get("fileUnit", {})
        title = item.get("title", "")
        scope = item.get("scopeAndContentNote", "")
        if not (title or scope):
            continue
        docs.append({
            "id": f"nara:{r.get('naId', '')}",
            "api": "nara",
            "citation": f"National Archives, {title} (NAID {r.get('naId', '')})",
            "url": f"https://catalog.archives.gov/id/{r.get('naId', '')}",
            "date": "",
            "text": _clean(f"{title}. {scope}")[:4000],
        })
    return docs


def fbi_vault(query: str, limit: int = 5) -> list[dict]:
    """FBI Vault (Plone site - HTML search, best-effort link+summary harvest)."""
    resp = requests.get(
        "https://vault.fbi.gov/search",
        params={"SearchableText": query},
        headers=_UA, timeout=_TIMEOUT)
    resp.raise_for_status()
    docs = []
    pattern = re.compile(
        r'<a[^>]+href="(https://vault\.fbi\.gov/[^"]+)"[^>]*class="state-published"'
        r'[^>]*>([^<]+)</a>.*?<span[^>]*>([^<]*)</span>', re.DOTALL)
    for url, title, summary in pattern.findall(resp.text)[:limit]:
        docs.append({
            "id": f"fbivault:{url.rstrip('/').rsplit('/', 1)[-1]}",
            "api": "fbi_vault",
            "citation": f"FBI Records: The Vault, {_clean(title)}",
            "url": url, "date": "",
            "text": _clean(f"{title}. {summary}"),
        })
    return [d for d in docs if len(d["text"]) > 20]


def cia_crest(query: str, limit: int = 5) -> list[dict]:
    """CIA CREST reading room (HTML search, best-effort harvest)."""
    resp = requests.get(
        "https://www.cia.gov/readingroom/search/site/" + requests.utils.quote(query),
        headers=_UA, timeout=_TIMEOUT)
    resp.raise_for_status()
    docs = []
    pattern = re.compile(
        r'<h3 class="title">\s*<a href="(/readingroom/[^"]+)">([^<]+)</a>.*?'
        r'<p class="search-snippet[^"]*">(.*?)</p>', re.DOTALL)
    for path, title, snippet in pattern.findall(resp.text)[:limit]:
        docs.append({
            "id": f"crest:{path.rstrip('/').rsplit('/', 1)[-1]}",
            "api": "cia_crest",
            "citation": f"CIA CREST, {_clean(title)}",
            "url": "https://www.cia.gov" + path, "date": "",
            "text": _clean(re.sub(r"<[^>]+>", " ", f"{title}. {snippet}")),
        })
    return [d for d in docs if len(d["text"]) > 20]


ALL_SOURCES = {
    "courtlistener": courtlistener,
    "chronicling_america": chronicling_america,
    "loc": loc,
    "nara": nara,
    "fbi_vault": fbi_vault,
    "cia_crest": cia_crest,
}


def harvest_all(query: str, limit_per_source: int = 5,
                sources: dict | None = None) -> dict:
    """Run every client; per-source failures are recorded, never fatal."""
    docs: list[dict] = []
    errors: dict[str, str] = {}
    seen: set[str] = set()
    for name, client in (sources or ALL_SOURCES).items():
        try:
            for doc in client(query, limit_per_source):
                if doc["id"] not in seen:
                    seen.add(doc["id"])
                    docs.append(doc)
        except Exception as err:  # noqa: BLE001 - isolate source failures
            errors[name] = f"{type(err).__name__}: {err}"
    return {"docs": docs, "errors": errors}


# --- helpers -----------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _iso(yyyymmdd: str) -> str:
    if re.fullmatch(r"\d{8}", yyyymmdd or ""):
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
    return yyyymmdd or ""
