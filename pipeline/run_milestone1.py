"""Milestone 1 orchestrator: case -> script -> validate -> ground -> voice -> master.

Usage:
    python -m pipeline.run_milestone1 --case cases/example/case.json [--run-id my-run]
                                      [--tts chatterbox|kokoro] [--skip-tts]

Every stage is checkpointed under runs/<run_id>/ and resumable (kill it and
re-run; completed stages are skipped). Designed to run stage-split across
free compute: LLM stages anywhere, TTS stage on Kaggle/Modal GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import Run
from .config import load_config
from .mastering import master
from .script_engine.generate import adversarial_rewrite, draft_script
from .script_engine.grounding import decompose_claims, ground_claims
from .script_engine.structure import ValidationError, validate_script
from .tts import get_engine
from .voice_director import direct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="path to case.json")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--tts", default=None, choices=["chatterbox", "kokoro"])
    ap.add_argument("--skip-tts", action="store_true", help="stop after grounding (no GPU needed)")
    args = ap.parse_args()

    cfg = load_config()
    if args.tts:
        object.__setattr__(cfg, "tts_engine", args.tts)  # frozen dataclass override
    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    run = Run(cfg.workdir, args.run_id or Path(args.case).parent.name)

    # 1. Draft (frontier LLM)
    draft = run.stage("01_draft", lambda: draft_script(cfg, case))

    # 2. Adversarial rewrite (frontier LLM)
    script = run.stage("02_rewrite", lambda: adversarial_rewrite(cfg, draft))

    # 3. Hard structural validation (no LLM) - fails loudly, per PLAN.md Layer 2
    try:
        report = run.stage("03_validate", lambda: validate_script(script, cfg))
    except ValidationError as err:
        print(f"\nSCRIPT REJECTED: {err}")
        print("Fix: delete runs/<id>/02_rewrite.json and re-run to regenerate.")
        raise SystemExit(1) from err
    print("\n".join(f"  {line}" for line in report))

    # 4. Claim grounding (local LLM) - citation manifest, per PLAN.md Layer 5
    manifest = run.stage("04_grounding", lambda: ground_claims(
        cfg, decompose_claims(cfg, script), case["sources"]))
    if manifest["contradicted"]:
        print(f"\nCONTRADICTED CLAIMS - HARD FAIL: {manifest['contradicted']}")
        raise SystemExit(1)
    if manifest["flagged"]:
        print(f"\nWARNING - ungrounded claims flagged for human review: {manifest['flagged']}")

    # 5. Voice direction (deterministic, seeded jitter)
    lines = run.stage("05_direction", lambda: direct(script))

    if args.skip_tts:
        print("\n--skip-tts: stopping before GPU stages. Re-run without it on a GPU box.")
        return

    # 6. TTS render (GPU stage - run on Kaggle/Modal free quota)
    raw_path = run.dir / "narration_raw.wav"
    run.stage("06_tts", lambda: get_engine(cfg).render(lines, raw_path))

    # 7. Loudnorm master + QC gate (-14 LUFS / -1 dBTP, per PLAN.md Layer 7)
    result = run.stage("07_master", lambda: master(raw_path, run.dir / "narration_master.wav", cfg))
    print(f"\nDONE: {result['path']}  ({result['integrated_lufs']} LUFS, TP {result['true_peak_db']} dB)")


if __name__ == "__main__":
    main()
