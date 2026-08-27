"""Milestone 5 orchestrator: retention -> post-mortem -> channel aggregation.

Usage (~1 week after upload, when retention data has settled):
    # Live: OAuth browser flow, pulls the audience-retention report
    python -m pipeline.run_milestone5 --run-id <run-id> --video-id <yt-id> \
                                      --client-secrets secrets.json

    # Offline: a pre-saved retention curve (rows or canonical shape)
    python -m pipeline.run_milestone5 --run-id <run-id> --video-id <yt-id> \
                                      --retention-json curve.json

Reads (from Milestone 3):
    runs/<id>/m3_02_timeline.json    narration timeline (beats)
    runs/<id>/m3_03_edl.json         the EDL every dip maps back to

Writes:
    runs/<id>/m5_01_retention.json   canonical retention curve
    runs/<id>/m5_02_postmortem.json  dips/spikes mapped to EDL beats
    feedback/channel_state.json      channel-wide: adjusted Rule-of-Six
                                     weights + footage-priority findings

The channel aggregation re-runs every time (never checkpointed): it folds
EVERY runs/*/m5_02_postmortem.json on disk, so each new post-mortem sharpens
the weights that run_milestone3 loads for the next video. Everything here is
CPU + one API call - no GPU budget spent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import Run
from .config import load_config
from .edit.rule_of_six import WEIGHTS as BASE_WEIGHTS
from .feedback.aggregate import (adjust_weights, aggregate_postmortems,
                                 footage_findings, write_channel_state)
from .feedback.analytics import fetch_retention, normalize_curve
from .feedback.postmortem import build_postmortem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--video-id", required=True, help="published YouTube video id")
    ap.add_argument("--client-secrets", default=None, help="OAuth client secrets json")
    ap.add_argument("--retention-json", default=None,
                    help="pre-saved retention rows (offline; skips the API)")
    args = ap.parse_args()
    if not (args.client_secrets or args.retention_json):
        raise SystemExit("need --client-secrets (live) or --retention-json (offline)")

    cfg = load_config()
    run = Run(cfg.workdir, args.run_id)
    for stage in ("m3_02_timeline", "m3_03_edl"):
        if not run.is_done(stage):
            raise SystemExit(f"runs/{args.run_id}/{stage}.json missing - "
                             "run Milestone 3 first")
    timeline = run.load("m3_02_timeline")
    edl = run.load("m3_03_edl")

    # 1. Retention curve (one API call, or a file - checkpointed either way).
    def _retention() -> list[dict]:
        if args.retention_json:
            rows = json.loads(Path(args.retention_json).read_text(encoding="utf-8"))
        else:
            rows = fetch_retention(args.video_id, Path(args.client_secrets))
        return normalize_curve(rows)
    curve = run.stage("m5_01_retention", _retention)
    print(f"  {len(curve)} buckets, final watch ratio {curve[-1]['watch']:.2f}")

    # 2. Post-mortem: dips/spikes mapped to exact EDL beats.
    postmortem = run.stage("m5_02_postmortem", lambda: build_postmortem(
        args.video_id, curve, edl, timeline))
    print(f"  {len(postmortem['dips'])} dips, {len(postmortem['spikes'])} spikes")
    for dip in postmortem["dips"][:5]:
        print(f"    dip depth={dip['depth']:.3f} at {dip['t_start']:.0f}-"
              f"{dip['t_end']:.0f}s  beats={dip['beat_ids']}  "
              f"components={dip['components']}")

    # 3. Channel aggregation: every post-mortem on disk, re-folded fresh.
    #    Always adjusted from Murch's base weights - re-folding the full
    #    evidence is idempotent, so the same videos never compound twice.
    postmortems = _all_postmortems(cfg.workdir)
    aggregate = aggregate_postmortems(postmortems)
    weights = adjust_weights(dict(BASE_WEIGHTS), aggregate)
    findings = footage_findings(aggregate)
    state_path = write_channel_state(
        cfg.feedback_dir / "channel_state.json", aggregate, weights, findings)

    print(f"\n  channel state ({aggregate['n_videos']} videos): {state_path}")
    print(f"  Rule-of-Six weights: {weights}")
    for f in findings:
        print(f"  FINDING: {f['finding']}\n           -> {f['action']}")
    print("\nDONE: next run_milestone3 invocation cuts with these weights")


def _all_postmortems(workdir: Path) -> list[dict]:
    out = []
    for path in sorted(Path(workdir).glob("*/m5_02_postmortem.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


if __name__ == "__main__":
    main()
