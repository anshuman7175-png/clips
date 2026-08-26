"""Milestone 2 orchestrator: script -> shot plan -> harvest -> embed -> match gate.

Usage:
    python -m pipeline.run_milestone2 --run-id <m1-run-id> [--skip-gpu]
    # reads runs/<run-id>/02_rewrite.json produced by Milestone 1

Stage split for free compute (PLAN.md compute budget):
  CPU anywhere : 01_shot_plan (local LLM), 02_harvest (network), 05_generation_jobs
  GPU box      : 03_embed (SigLIP2/X-CLIP), 06_wan_render
  The 04_match stage needs the same embedder as 03_embed for query vectors,
  so it runs on the GPU box too - unless EMBEDDER=hash (offline dry runs).

Every stage checkpoints under runs/<run-id>/ exactly like Milestone 1.
The embedding library itself lives OUTSIDE the run dir (library/) because
it is shared across all videos: embed once, incrementally, never per video.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from .checkpoint import Run
from .config import load_config
from .footage import harvesters, wan_generate
from .footage.embed_index import EmbeddingIndex, ensure_embedded, get_embedder
from .footage.ledger import Ledger, dump_manifest
from .footage.match_gate import run_gate
from .footage.shot_planner import plan_shots

_KINDS_BY_SPACE = {"still": ("document", "still"), "video": ("footage",)}


def download_media(rec, dest_dir: Path) -> Path | None:
    """Stream an asset's media into the shared library (content-addressed later)."""
    if not rec.media_url:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(rec.media_url.split("?")[0]).suffix or ".bin"
    dest = dest_dir / (rec.asset_id.replace(":", "_") + ext)
    if dest.exists():
        return dest
    with requests.get(rec.media_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Milestone 1 run id (needs 02_rewrite.json)")
    ap.add_argument("--skip-gpu", action="store_true",
                    help="stop after harvest (embed/match need the GPU box unless EMBEDDER=hash)")
    args = ap.parse_args()

    cfg = load_config()
    run = Run(cfg.workdir, args.run_id)
    if not run.is_done("02_rewrite"):
        raise SystemExit(f"runs/{args.run_id}/02_rewrite.json not found - run Milestone 1 first")
    script = run.load("02_rewrite")

    ledger = Ledger(cfg.library_dir / "ledger.sqlite")

    # 1. Shot plan (local LLM) - beats become enforceable shot-request contracts.
    shots = run.stage("m2_01_shot_plan", lambda: plan_shots(cfg, script))
    print(f"  {len(shots)} shots planned "
          f"({sum(s['evidentiary'] for s in shots)} evidentiary)")

    # 2. Harvest (network, CPU) - item-level license classification into the ledger.
    def _harvest() -> dict:
        seen: set[str] = set()
        for shot in shots:
            for rec in harvesters.harvest(ledger, shot["query"], shot["kind"],
                                          cfg.harvest_limit_per_source):
                seen.add(rec.asset_id)
        return {"usable_harvested": len(seen),
                "review_queue": [r.asset_id for r in ledger.review_queue()]}
    harvest_report = run.stage("m2_02_harvest", _harvest)
    if harvest_report["review_queue"]:
        print(f"  NEEDS_REVIEW queue: {len(harvest_report['review_queue'])} items "
              "(resolve in library/ledger.sqlite before they can match)")

    if args.skip_gpu and cfg.embedder != "hash":
        print("\n--skip-gpu: stopping before embed/match. Re-run on the GPU box.")
        return

    # 3. Embed (GPU / hash) - incremental, into the SHARED library index.
    embedder_name = cfg.embedder if cfg.embedder != "auto" else "model"
    indexes, embedders = {}, {}
    for space in ("still", "video"):
        indexes[space] = EmbeddingIndex(cfg.library_dir, space)
        embedders[space] = get_embedder(
            space, "hash" if embedder_name == "hash" else space)

    def _embed() -> dict:
        added = {}
        for space, kinds in _KINDS_BY_SPACE.items():
            records = [r for k in kinds for r in ledger.usable_records(kind=k)]
            downloader = None if cfg.embedder == "hash" else download_media
            added[space] = ensure_embedded(indexes[space], embedders[space], records,
                                           cfg.library_dir / "media", downloader)
        return {"added": added,
                "index_sizes": {s: len(indexes[s]) for s in indexes}}
    run.stage("m2_03_embed", _embed)

    # 4. Match gate - threshold + literal + priority; decisions, not warnings.
    thresholds = {"still": cfg.still_match_threshold, "video": cfg.video_match_threshold}
    manifest = run.stage("m2_04_match",
                         lambda: run_gate(shots, ledger, indexes, embedders, thresholds))
    stats = manifest["stats"]
    print(f"  matched={stats['matched']} rewrite_line={stats['rewrite_line']} "
          f"generate={stats['generate']}")
    if manifest["rewrite_requests"]:
        print("  REWRITE REQUESTS (evidentiary shots with no real asset):")
        for req in manifest["rewrite_requests"]:
            print(f"    beat {req['beat_id']} / {req['shot_id']}: {req['reason']}")

    # 5. Wan generation jobs (CPU: spec only; GPU renders separately).
    jobs = run.stage("m2_05_generation_jobs", lambda: wan_generate.build_jobs(
        manifest["generation_jobs"], run.dir / "generated"))
    if jobs:
        wan_generate.dump_jobs(jobs, run.dir / "wan_jobs.json")
        print(f"  {len(jobs)} Wan jobs -> runs/{args.run_id}/wan_jobs.json "
              "(render with --render-wan on the GPU box)")

    # License/attribution manifest for the video description (Layer 5 trust + Layer 0.6).
    dump_manifest(ledger, run.dir / "license_manifest.json")
    print(f"\nDONE: shotlist in runs/{args.run_id}/m2_04_match.json")


if __name__ == "__main__":
    main()
