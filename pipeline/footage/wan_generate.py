"""Wan 2.2 generation branch (PLAN.md Layer 4).

Atmosphere/reconstruction shots ONLY - the match gate guarantees that no
evidentiary shot ever reaches this module. Two-phase design so the GPU
stage is schedulable on free quota:

1. build_jobs()  - CPU, anywhere: deterministic job specs (seeded from
   shot_id so a re-run regenerates the SAME shot - checkpointing needs
   reproducibility). Prompts are built with hard style guardrails: no
   faces of real people, no text/insignia, period-ambiguous atmosphere.
2. render_jobs() - GPU box (Kaggle/Modal): lazy-imports diffusers/Wan,
   renders each job, registers output in the ledger as GENERATED with
   dramatization=True (the assembly stage MUST burn the label in - it
   reads this flag, it cannot forget).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .ledger import AssetRecord, Ledger

# Guardrails appended to every prompt: generated shots must read as mood,
# never as forged evidence (Layer 0.3 + YouTube disclosure, Layer 9).
_STYLE = ("cinematic documentary reconstruction, atmospheric, shallow depth of field, "
          "muted period color grade, no readable text, no legible signage")
_NEGATIVE = ("recognizable faces, celebrity likeness, readable documents, watermarks, "
             "modern vehicles, modern clothing, text overlays, logos")


def build_jobs(generation_shots: list[dict], out_dir: Path) -> list[dict]:
    jobs = []
    for shot in generation_shots:
        seed = int(hashlib.sha256(shot["shot_id"].encode()).hexdigest()[:8], 16)
        jobs.append({
            "shot_id": shot["shot_id"],
            "beat_id": shot["beat_id"],
            "prompt": f"{shot['query']}, {_STYLE}",
            "negative_prompt": _NEGATIVE,
            "seed": seed,                      # deterministic: re-run == same shot
            "duration_s": shot["duration_s"],
            "fps": 16,
            "resolution": [1280, 720],
            "output": str(Path(out_dir) / f"{shot['shot_id']}.mp4"),
        })
    return jobs


def render_jobs(jobs: list[dict], ledger: Ledger, device: str = "cuda") -> list[dict]:
    """GPU stage. Renders pending jobs, registers each in the ledger."""
    try:
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline
    except ImportError as err:
        raise RuntimeError(
            "Wan 2.2 stage needs GPU deps: pip install diffusers transformers torch. "
            "Run this stage on Kaggle/Modal free quota."
        ) from err

    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16).to(device)

    results = []
    for job in jobs:
        out_path = Path(job["output"])
        if out_path.exists():  # resumable: skip already-rendered shots
            results.append({**job, "status": "cached"})
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames = pipe(
            prompt=job["prompt"],
            negative_prompt=job["negative_prompt"],
            num_frames=int(job["duration_s"] * job["fps"]) + 1,
            height=job["resolution"][1], width=job["resolution"][0],
            generator=torch.Generator(device).manual_seed(job["seed"]),
        ).frames[0]
        _export_video(frames, out_path, job["fps"])
        _register(ledger, job, out_path)
        results.append({**job, "status": "rendered"})
    return results


def _register(ledger: Ledger, job: dict, out_path: Path) -> None:
    rec = AssetRecord(
        asset_id=f"generated:{job['shot_id']}",
        source="generated",
        kind="footage",
        title=f"Generated reconstruction for {job['shot_id']}",
        description=job["prompt"],
        license_raw="self-generated (Wan 2.2, Apache 2.0 weights)",
        license_class="GENERATED",
        dramatization=True,  # assembly MUST render the on-screen label
        local_path=str(out_path),
    )
    rec.license_class = "GENERATED"  # classify() would say NEEDS_REVIEW; we assert provenance
    ledger.upsert(rec)
    # upsert() re-classifies; force the class back with a direct update.
    ledger.db.execute(
        "UPDATE assets SET license_class='GENERATED', license_reason='own Wan 2.2 output' "
        "WHERE asset_id=?", (rec.asset_id,))
    ledger.db.commit()
    ledger.register_download(rec.asset_id, out_path)


def _export_video(frames, out_path: Path, fps: int) -> None:
    from diffusers.utils import export_to_video
    export_to_video(frames, str(out_path), fps=fps)


def dump_jobs(jobs: list[dict], path: Path) -> None:
    path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
