"""Milestone 3 orchestrator: alignment -> timeline -> EDL -> assembly export.

Usage:
    python -m pipeline.run_milestone3 --run-id <run-id> [--synthetic-align]
    # reads runs/<run-id>/02_rewrite.json   (Milestone 1)
    #       runs/<run-id>/05_direction.json (Milestone 1, voice director)
    #       runs/<run-id>/m2_04_match.json  (Milestone 2, shotlist manifest)
    # optional: runs/<run-id>/narration_master.wav for WhisperX alignment

Stage split for free compute (PLAN.md compute budget):
  GPU box      : m3_01_align with WhisperX (only if narration audio exists)
  CPU anywhere : everything else - the EDL grammar is pure Python, and
                 --synthetic-align derives timing from voice-director params.

Every stage checkpoints under runs/<run-id>/ exactly like Milestones 1-2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import Run
from .config import load_config
from .edit.align import align
from .edit.edl import build_edl
from .edit.remotion_export import export_props, write_ffmpeg_fallback
from .edit.rule_of_six import best_edl, score_edl
from .edit.timeline import build_timeline


def _media_paths(manifest: dict, run_dir: Path, library_dir: Path) -> dict[str, str]:
    """asset_id -> local media path. Library media for matched shots,
    runs/<id>/generated/ renders for Wan shots; missing files export as slates."""
    paths: dict[str, str] = {}
    media_dir = library_dir / "media"
    for shot in manifest["shots"]:
        if shot["decision"] == "generate":
            asset_id = f"generated:{shot['shot_id']}"
            candidates = list((run_dir / "generated").glob(f"{shot['shot_id']}.*"))
        else:
            asset_id = shot["asset_id"]
            stem = asset_id.replace(":", "_")
            candidates = list(media_dir.glob(f"{stem}.*")) if media_dir.exists() else []
        if candidates:
            paths[asset_id] = str(candidates[0])
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--synthetic-align", action="store_true",
                    help="skip WhisperX; derive timing from voice-director params")
    args = ap.parse_args()

    cfg = load_config()
    run = Run(cfg.workdir, args.run_id)
    for stage in ("02_rewrite", "05_direction", "m2_04_match"):
        if not run.is_done(stage):
            raise SystemExit(f"runs/{args.run_id}/{stage}.json missing - "
                             "run the earlier milestones first")
    script = run.load("02_rewrite")
    direction = run.load("05_direction")
    manifest = run.load("m2_04_match")

    narration = run.dir / "narration_master.wav"
    audio_path = narration if narration.exists() else None

    # 1. Forced alignment (GPU: WhisperX) or synthetic fallback (CPU).
    alignment = run.stage("m3_01_align", lambda: align(
        direction["lines"], audio_path, cfg.align_device,
        force_synthetic=args.synthetic_align))
    print(f"  aligner={alignment['aligner']} "
          f"duration={alignment['duration_s']:.1f}s "
          f"words={len(alignment['words'])}")

    # 2. Narration timeline with editorial pause beats.
    timeline = run.stage("m3_02_timeline", lambda: build_timeline(script, alignment))
    print(f"  {len(timeline['beats'])} beats, "
          f"{len(timeline['pause_beats'])} editorial pauses")

    # 3. EDL: N jitter-seeded candidates, Rule-of-Six picks the winner.
    seeds = list(range(cfg.edl_candidate_seeds))
    def _edl() -> dict:
        edl, report = best_edl(timeline, manifest, seeds)
        edl["rule_of_six"] = report
        return edl
    edl = run.stage("m3_03_edl", _edl)
    r6 = edl["rule_of_six"]["winner_score"]
    print(f"  {len(edl['events'])} events, winner seed={edl['seed']} "
          f"score={r6['total']} (emotion={r6['emotion']} story={r6['story']})")

    # 4. Assembly export: Remotion props + FFmpeg fallback script.
    def _export() -> dict:
        media = _media_paths(manifest, run.dir, cfg.library_dir)
        props = export_props(edl, audio_path, media, run.dir / "remotion_props.json")
        write_ffmpeg_fallback(edl, audio_path, media, run.dir / "assemble_ffmpeg.sh")
        missing = [s["eventId"] for s in props["sequences"] if s["missing"]]
        return {"media_resolved": len(media), "missing_events": missing}
    export = run.stage("m3_04_export", _export)
    if export["missing_events"]:
        print(f"  WARNING: {len(export['missing_events'])} events have no local "
              f"media yet (slates): {export['missing_events'][:6]}")

    print(f"\nDONE: EDL in runs/{args.run_id}/m3_03_edl.json")
    print(f"  Remotion: npx remotion render Documentary out.mp4 "
          f"--props=runs/{args.run_id}/remotion_props.json  (from remotion/)")
    print(f"  Fallback: bash runs/{args.run_id}/assemble_ffmpeg.sh")


if __name__ == "__main__":
    main()
