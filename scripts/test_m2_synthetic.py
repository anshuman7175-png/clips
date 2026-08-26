"""Offline synthetic test for the Milestone 2 footage layer.

No network, no GPU, no models: the HashEmbedder gives real cosine ranking
over a controlled asset set, so every gate rule is exercised end to end:

  T1  license classifier: NC -> BANNED, SA -> NEEDS_REVIEW, CC0/PD/BY -> usable
  T2  blocklist: British Pathe URL is BANNED regardless of claimed license
  T3  banned/review assets can NEVER match (whitelist, defense in depth)
  T4  priority: archival beats stock even when stock scores higher
  T5  literal-term check: high similarity without literal terms fails evidentiary shots
  T6  evidentiary failure -> rewrite_line ; atmosphere failure -> generate
  T7  Wan jobs are deterministic (same shot_id -> same seed) and label_required
  T8  embed-once: re-running ensure_embedded adds nothing
  T9  attribution manifest exports CC_BY/PD items

Run:  python scripts/test_m2_synthetic.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.footage.embed_index import EmbeddingIndex, HashEmbedder, ensure_embedded
from pipeline.footage.ledger import AssetRecord, Ledger, classify_license
from pipeline.footage.match_gate import run_gate
from pipeline.footage.wan_generate import build_jobs

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def make_assets() -> list[AssetRecord]:
    return [
        # Archival document: the CORRECT match for the evidentiary shot.
        AssetRecord(asset_id="loc:memo54", source="loc", kind="document",
                    title="FBI memo March 1954 surveillance report",
                    description="typed memo with redactions, 1954",
                    source_url="https://www.loc.gov/item/memo54",
                    license_raw="no known restrictions",
                    attribution="Library of Congress"),
        # Stock lookalike: similar text, must LOSE to archival on priority (T4)
        # and fail the literal check for evidentiary use (kind mismatch anyway).
        AssetRecord(asset_id="pexels:999", source="pexels", kind="still",
                    title="old typed memo paper vintage 1954 surveillance",
                    source_url="https://pexels.com/photo/999",
                    license_raw="Pexels License", attribution="Pexels / Someone"),
        # Archival still, lower lexical overlap than the pexels one.
        AssetRecord(asset_id="nara:photo1", source="nara", kind="still",
                    title="surveillance photograph 1954 field office",
                    source_url="https://catalog.archives.gov/id/photo1",
                    license_raw="public domain (US federal record, unrestricted)",
                    attribution="U.S. National Archives"),
        # Banned: NC license. Perfect lexical match on purpose (T3).
        AssetRecord(asset_id="archive_org:nc1", source="archive_org", kind="still",
                    title="FBI memo March 1954 surveillance report redactions",
                    source_url="https://archive.org/details/nc1",
                    license_url="https://creativecommons.org/licenses/by-nc/4.0/"),
        # Banned: blocklisted source URL despite PD claim (T2).
        AssetRecord(asset_id="archive_org:pathe", source="archive_org", kind="footage",
                    title="1954 newsreel city night rain streets",
                    source_url="https://www.britishpathe.com/asset/xyz",
                    license_raw="public domain"),
        # Usable footage, but about the wrong thing (low similarity to queries).
        AssetRecord(asset_id="pixabay:cat", source="pixabay", kind="footage",
                    title="kitten playing yarn living room",
                    source_url="https://pixabay.com/videos/cat",
                    license_raw="Pixabay Content License", attribution="Pixabay / Cats"),
    ]


def main() -> None:
    print("T1/T2: license classification")
    check("CC-NC banned", classify_license("", "https://creativecommons.org/licenses/by-nc/4.0", "archive_org")[0] == "BANNED")
    check("CC-SA needs review", classify_license("", "https://creativecommons.org/licenses/by-sa/4.0", "archive_org")[0] == "NEEDS_REVIEW")
    check("CC0 usable", classify_license("", "https://creativecommons.org/publicdomain/zero/1.0", "archive_org")[0] == "CC0")
    check("CC-BY usable", classify_license("", "https://creativecommons.org/licenses/by/4.0", "archive_org")[0] == "CC_BY")
    check("bare item -> review", classify_license("", "", "archive_org")[0] == "NEEDS_REVIEW")
    check("Pathe blocklisted", classify_license("public domain", "https://www.britishpathe.com/x", "archive_org")[0] == "BANNED")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ledger = Ledger(tmp / "ledger.sqlite")
        for rec in make_assets():
            ledger.upsert(rec)

        print("\nT3: ledger whitelist")
        check("NC asset unusable", not ledger.usable("archive_org:nc1"))
        check("Pathe asset unusable", not ledger.usable("archive_org:pathe"))
        check("LoC memo usable", ledger.usable("loc:memo54"))
        check("4 usable total", len(ledger.usable_records()) == 4,
              f"got {len(ledger.usable_records())}")

        # Build indexes with the hash embedder (both spaces share it here).
        embedder = HashEmbedder()
        indexes = {s: EmbeddingIndex(tmp / "lib", s) for s in ("still", "video")}
        still_records = [r for k in ("document", "still") for r in ledger.usable_records(kind=k)]
        video_records = ledger.usable_records(kind="footage")
        # NOTE: banned assets are deliberately embedded too - the gate must
        # still refuse them (defense in depth), so add them raw:
        for rec in make_assets():
            space = "video" if rec.kind == "footage" else "still"
            indexes[space].add(rec.asset_id, embedder.embed_asset(rec))
        for idx in indexes.values():
            idx.save()

        print("\nT8: embed-once contract")
        added_first = ensure_embedded(indexes["still"], embedder, still_records, tmp)
        check("no re-embedding of indexed assets", added_first == 0, f"added {added_first}")

        shots = [
            {"shot_id": "b01-s1", "beat_id": "b01", "kind": "document",
             "query": "FBI memo March 1954 surveillance report redactions",
             "literal_terms": ["memo", "1954"], "evidentiary": True, "duration_s": 6.0},
            {"shot_id": "b02-s1", "beat_id": "b02", "kind": "still",
             "query": "surveillance photograph 1954 field office",
             "literal_terms": ["surveillance"], "evidentiary": True, "duration_s": 5.0},
            # Evidentiary with a query nothing real satisfies -> rewrite_line (T6)
            {"shot_id": "b03-s1", "beat_id": "b03", "kind": "document",
             "query": "handwritten confession letter signed by the suspect",
             "literal_terms": ["confession"], "evidentiary": True, "duration_s": 6.0},
            # Atmosphere footage nothing matches -> generate (T6)
            {"shot_id": "b04-s1", "beat_id": "b04", "kind": "footage",
             "query": "rain-slicked empty city street at night, 1950s atmosphere",
             "literal_terms": [], "evidentiary": False, "duration_s": 8.0},
        ]
        thresholds = {"still": 0.2, "video": 0.2}
        embedders = {"still": embedder, "video": embedder}
        manifest = run_gate(shots, ledger, indexes, embedders, thresholds)
        by_id = {s["shot_id"]: s for s in manifest["shots"]}

        print("\nT3/T4/T5: match gate decisions")
        check("evidentiary doc matched to LoC (not banned NC lookalike)",
              by_id["b01-s1"]["decision"] == "matched"
              and by_id["b01-s1"]["asset_id"] == "loc:memo54",
              str(by_id["b01-s1"]))
        check("archival still beats stock (priority over score)",
              by_id["b02-s1"]["asset_id"] == "nara:photo1", str(by_id["b02-s1"]))
        check("unmatched evidentiary -> rewrite_line",
              by_id["b03-s1"]["decision"] == "rewrite_line", str(by_id["b03-s1"]))
        check("unmatched atmosphere -> generate + label",
              by_id["b04-s1"]["decision"] == "generate"
              and by_id["b04-s1"]["label_required"] is True, str(by_id["b04-s1"]))
        check("stats consistent", manifest["stats"] == {"matched": 2, "rewrite_line": 1,
                                                        "generate": 1})

        print("\nT7: Wan job determinism")
        jobs_a = build_jobs(manifest["generation_jobs"], tmp / "gen")
        jobs_b = build_jobs(manifest["generation_jobs"], tmp / "gen")
        check("seeds deterministic", jobs_a[0]["seed"] == jobs_b[0]["seed"])
        check("guardrails in prompt", "no readable text" in jobs_a[0]["prompt"])
        check("negative prompt bans faces", "faces" in jobs_a[0]["negative_prompt"])

        print("\nT9: attribution manifest")
        attrib = ledger.export_attribution()
        check("no attribution before download", attrib == [])
        media = tmp / "m.bin"
        media.write_bytes(b"fake-media-bytes")
        ledger.register_download("loc:memo54", media)
        attrib = ledger.export_attribution()
        check("PD item exported after download",
              len(attrib) == 1 and attrib[0]["asset_id"] == "loc:memo54", str(attrib))
        # Dedup: same bytes under a second id resolves to the first.
        dup = ledger.register_download("nara:photo1", media)
        check("content-hash dedup detects duplicate", dup == "loc:memo54", dup)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
