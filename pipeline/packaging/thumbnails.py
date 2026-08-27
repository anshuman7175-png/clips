"""Thumbnail concepts + Qwen-Image render jobs (PLAN.md Layer 9).

Three DISTINCT concepts per video, so the upload can A/B (YouTube's
test-and-compare takes up to 3):

  document    - extreme close-up of the case's key document, one phrase
                legible. The evidence IS the hook.
  location    - the place where it happened, empty and quiet. Atmosphere.
  typographic - 3-5 word hook phrase as bold type on a dark field.

Same two-phase split as footage/wan_generate.py: build_thumbnail_jobs() is
CPU-only and deterministic (seeded from concept id); render_thumbnail_jobs()
lazy-imports diffusers on the GPU box. Thumbnails are GENERATED images of
real evidence/places - they never fabricate a new "fact" (persona rule)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

WIDTH, HEIGHT = 1280, 720  # YouTube thumbnail spec

_STYLE = ("cinematic documentary thumbnail, high contrast, moody film grain, "
          "muted desaturated palette, no text overlay, no watermark, 16:9")

_STOPWORDS = {"the", "a", "an", "of", "in", "at", "on", "and", "to", "for",
              "was", "were", "is", "are", "had", "his", "her", "its", "that"}


def build_concepts(script: dict, shotlist: dict) -> list[dict]:
    """Pure. Derive the 3 concepts from the script + matched shotlist."""
    beats = script["beats"]
    hook_beat = beats[0]  # cold open carries the hook by Layer 2 contract

    # Best evidentiary document shot -> the document concept subject.
    docs = [s for s in shotlist["shots"]
            if s["kind"] == "document" and s["decision"] == "matched"]
    doc_query = docs[0]["query"] if docs else "aged archival document, typewritten page"

    # Best location-ish shot (footage or still) -> the location concept.
    places = [s for s in shotlist["shots"]
              if s["kind"] in ("footage", "still") and s["decision"] == "matched"]
    place_query = places[0]["query"] if places else "empty small-town street at dusk"

    return [
        {"concept_id": "document",
         "prompt": f"extreme close-up of {doc_query}, dramatic raking light, "
                   f"shallow focus on one legible phrase, {_STYLE}"},
        {"concept_id": "location",
         "prompt": f"{place_query}, deserted, long shadows, unsettling calm, "
                   f"wide establishing shot, {_STYLE}"},
        {"concept_id": "typographic",
         "hook": _hook_phrase(hook_beat["text"]),
         "prompt": "near-black textured paper background, single hard spotlight "
                   "falling off to darkness, empty center composition reserved "
                   f"for a title overlay, {_STYLE}"},
    ]


def build_thumbnail_jobs(concepts: list[dict], out_dir: Path) -> list[dict]:
    """CPU phase: deterministic job specs, mirrored to disk for the GPU box."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for c in concepts:
        seed = int(hashlib.sha256(c["concept_id"].encode()).hexdigest()[:8], 16)
        jobs.append({
            "job_id": f"thumb-{c['concept_id']}",
            "concept_id": c["concept_id"],
            "prompt": c["prompt"],
            "hook_text": c.get("hook"),  # typographic overlay happens post-render
            "seed": seed,
            "width": WIDTH, "height": HEIGHT,
            "out_path": str(out_dir / f"thumb_{c['concept_id']}.png"),
        })
    (out_dir / "thumbnail_jobs.json").write_text(
        json.dumps(jobs, indent=2), encoding="utf-8")
    return jobs


def render_thumbnail_jobs(jobs: list[dict], device: str = "cuda") -> list[str]:
    """GPU phase: Qwen-Image via diffusers (lazy import, same pattern as Wan)."""
    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as err:
        raise RuntimeError(
            "thumbnail rendering needs the GPU box: pip install torch diffusers "
            "transformers accelerate (then re-run with --render-thumbs)"
        ) from err
    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to(device)
    rendered = []
    for job in jobs:
        image = pipe(
            prompt=job["prompt"],
            width=job["width"], height=job["height"],
            generator=torch.Generator(device).manual_seed(job["seed"]),
        ).images[0]
        image.save(job["out_path"])
        rendered.append(job["out_path"])
    return rendered


def _hook_phrase(text: str, max_words: int = 5) -> str:
    """Pull a short, punchy phrase from the cold open for the typographic card."""
    first = re.split(r"[.!?]", text)[0]
    words = [w for w in re.findall(r"[A-Za-z0-9:']+", first)]
    strong = [w for w in words if w.lower() not in _STOPWORDS]
    picked = (strong or words)[:max_words]
    return " ".join(picked).upper()
