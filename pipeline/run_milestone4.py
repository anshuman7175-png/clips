"""Milestone 4 orchestrator: QC gates -> taste checklist -> packaging -> upload prep.

Usage:
    python -m pipeline.run_milestone4 --run-id <run-id> [--case cases/x/case.json]
                                      [--video runs/<id>/assembled.mp4]
                                      [--score-video] [--render-thumbs]
                                      [--upload --client-secrets secrets.json]

Reads (from Milestones 1-3):
    runs/<id>/02_rewrite.json        script
    runs/<id>/04_grounding.json      citation manifest
    runs/<id>/m2_04_match.json       shotlist manifest
    runs/<id>/license_manifest.json  attribution + review queue
    runs/<id>/m3_02_timeline.json    narration timeline
    runs/<id>/m3_03_edl.json         the EDL (QC failures map back to it)

Stage split for free compute (PLAN.md compute budget):
  CPU anywhere : ffmpeg technical tier, probe plan, taste checklist,
                 thumbnail job specs, disclosure/metadata/upload package.
  GPU box      : --score-video (Peak-End-Net + VQAThinker),
                 --render-thumbs (Qwen-Image).

The upload package is written even when QC fails - with the blockers listed -
so the human taste pass always has one file telling it what stands between
this run and publish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import Run
from .config import load_config
from .packaging.publish import (build_disclosure, build_metadata,
                                build_upload_package, persona_check,
                                upload_youtube)
from .packaging.thumbnails import (build_concepts, build_thumbnail_jobs,
                                   render_thumbnail_jobs)
from .qc.aesthetic import build_probe_plan, evaluate_aesthetics, score_probes
from .qc.taste_pass import build_checklist, write_checklist
from .qc.technical import run_technical_checks, run_vqa_checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--case", default=None, help="case.json (for title/citations)")
    ap.add_argument("--video", default=None, help="assembled video (default: runs/<id>/assembled.mp4)")
    ap.add_argument("--score-video", action="store_true",
                    help="GPU: Peak-End-Net aesthetic + VQAThinker scoring")
    ap.add_argument("--render-thumbs", action="store_true",
                    help="GPU: render the 3 thumbnail concepts with Qwen-Image")
    ap.add_argument("--upload", action="store_true", help="upload to YouTube (needs OAuth)")
    ap.add_argument("--client-secrets", default=None, help="OAuth client secrets json")
    args = ap.parse_args()

    cfg = load_config()
    run = Run(cfg.workdir, args.run_id)
    for stage in ("02_rewrite", "04_grounding", "m2_04_match",
                  "m3_02_timeline", "m3_03_edl"):
        if not run.is_done(stage):
            raise SystemExit(f"runs/{args.run_id}/{stage}.json missing - "
                             "run the earlier milestones first")
    script = run.load("02_rewrite")
    grounding = run.load("04_grounding")
    manifest = run.load("m2_04_match")
    timeline = run.load("m3_02_timeline")
    edl = run.load("m3_03_edl")
    license_manifest = _load_json(run.dir / "license_manifest.json",
                                  {"attribution": [], "needs_review": []})
    case = _load_json(Path(args.case), {}) if args.case else {}

    video = Path(args.video) if args.video else run.dir / "assembled.mp4"
    has_video = video.exists()
    if not has_video:
        print(f"  NOTE: {video} not found - video-dependent QC tiers are deferred")

    # 1. Technical QC (CPU ffmpeg tier; VQAThinker tier only with --score-video).
    def _technical() -> dict:
        if not has_video:
            return {"passed": None, "tier": "deferred", "failures": []}
        report = run_technical_checks(video, edl)
        if args.score_video:
            vqa = run_vqa_checks(video, edl)
            report = {"passed": report["passed"] and vqa["passed"],
                      "tier": "ffmpeg+vqa_thinker",
                      "failures": report["failures"] + vqa["failures"]}
        return report
    technical = run.stage("m4_01_technical", _technical)
    _print_report("technical", technical)

    # 2. Aesthetic QC: probe plan is always built (CPU); scoring needs the GPU.
    def _aesthetic() -> dict:
        plan = build_probe_plan(edl, timeline)
        if not (has_video and args.score_video):
            return {"passed": None, "plan": plan, "failures": [],
                    "note": "probe plan ready; re-run with --score-video on the GPU box"}
        scored = score_probes(video, plan)
        return {**evaluate_aesthetics(plan, scored), "plan": plan}
    aesthetic = run.stage("m4_02_aesthetic", _aesthetic)
    _print_report("aesthetic", aesthetic)

    # 3. Human taste pass checklist - every hotspot pre-located.
    checklist = run.stage("m4_03_taste", lambda: build_checklist(
        timeline, edl, grounding, [technical, aesthetic]))
    md_path = write_checklist(checklist, run.dir / "taste_checklist.md")
    print(f"  taste checklist: {md_path}")

    # 4. Thumbnails: 3 concepts, deterministic job specs; render on GPU.
    def _thumbs() -> dict:
        concepts = build_concepts(script, manifest)
        jobs = build_thumbnail_jobs(concepts, run.dir / "thumbnails")
        return {"concepts": concepts, "jobs": jobs}
    thumbs = run.stage("m4_04_thumbs", _thumbs)
    if args.render_thumbs:
        rendered = render_thumbnail_jobs(thumbs["jobs"])
        print(f"  rendered {len(rendered)} thumbnails")

    # 5. Persona rule - advice-giving narration is a publish blocker.
    violations = run.stage("m4_05_persona", lambda: persona_check(script))
    if violations:
        print(f"  PERSONA VIOLATIONS (publish blocked): {violations}")

    # 6. Disclosure + metadata + upload package.
    def _publish() -> dict:
        disclosure = build_disclosure(edl)
        metadata = build_metadata(case, script, timeline, grounding,
                                  license_manifest, disclosure)
        blockers = _blockers(technical, aesthetic, violations,
                             license_manifest, has_video)
        package_path = build_upload_package(
            run.dir, video, metadata, disclosure, thumbs["jobs"],
            {"passed": not blockers, "blockers": blockers})
        return {"disclosure": disclosure, "metadata": metadata,
                "blockers": blockers, "package": str(package_path)}
    publish = run.stage("m4_06_publish", _publish)

    print(f"\n  altered-content disclosure: {publish['disclosure']['altered_content']}")
    print(f"  upload package: {publish['package']}")
    if publish["blockers"]:
        print(f"  NOT ready to upload - blockers: {publish['blockers']}")
    else:
        print("  READY: human taste pass, then upload")

    # 7. Optional upload (after the human flips through taste_checklist.md).
    if args.upload:
        if not args.client_secrets:
            raise SystemExit("--upload needs --client-secrets <oauth json>")
        video_id = upload_youtube(Path(publish["package"]), Path(args.client_secrets))
        print(f"\nUPLOADED (private): https://youtu.be/{video_id}")


def _blockers(technical: dict, aesthetic: dict, violations: list,
              license_manifest: dict, has_video: bool) -> list[str]:
    blockers = []
    if not has_video:
        blockers.append("no assembled video")
    if technical["passed"] is False:
        blockers.append(f"technical QC: {len(technical['failures'])} failures")
    if aesthetic.get("passed") is False:
        blockers.append(f"aesthetic QC: aggregate {aesthetic['aggregate']} "
                        f"+ {len(aesthetic['failures'])} attribute failures")
    if violations:
        blockers.append(f"persona rule: {len(violations)} advice-giving lines")
    if license_manifest.get("needs_review"):
        blockers.append(f"license review queue: {len(license_manifest['needs_review'])} assets")
    return blockers


def _print_report(name: str, report: dict) -> None:
    status = {True: "PASS", False: "FAIL", None: "DEFERRED"}[report.get("passed")]
    line = f"  {name}: {status}"
    if report.get("failures"):
        line += f" ({len(report['failures'])} failures -> EDL timestamps)"
    if report.get("aggregate") is not None:
        line += f" aggregate={report['aggregate']}"
    print(line)


def _load_json(path: Path, default: dict) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


if __name__ == "__main__":
    main()
