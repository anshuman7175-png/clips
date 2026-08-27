# clips — Automated Documentary Pipeline

Fully-free automated pipeline for long-form mystery/documentary videos.
**The definitive blueprint is [PLAN.md](PLAN.md)** — every module traces back to it.

## Current status: all milestones code-complete (M1–M5 + Layer 1 research)

Research → script engine → voice → footage gate → edit engine → QC/packaging →
feedback loop. Every layer ships with an offline synthetic test suite:

```bash
python scripts/test_m1_synthetic.py         # script validators, grounding, voice, mastering QC
python scripts/test_m2_synthetic.py         # footage gate, license ledger, Wan jobs
python scripts/test_m3_synthetic.py         # Rule-of-Six EDL, assembly export
python scripts/test_m4_synthetic.py         # QC gates, disclosure, upload package
python scripts/test_m5_synthetic.py         # retention post-mortem, weight adjustment
python scripts/test_research_synthetic.py   # case scout, contradictions, novelty gate
```

What remains is live calibration: real harvests, threshold tuning on real assets,
the first end-to-end GPU render, and real retention data feeding M5.

## Quick start

```bash
pip install -r requirements.txt

# LLM endpoints (any OpenAI-compatible API)
export FRONTIER_API_BASE=https://api.openai.com/v1   # or your free-tier gateway
export FRONTIER_API_KEY=...
export FRONTIER_MODEL=gpt-4o
export LOCAL_API_BASE=http://localhost:11434/v1       # Ollama
export LOCAL_MODEL=qwen2.5:14b-instruct

# Run script + validation + grounding (no GPU needed)
python -m pipeline.run_milestone1 --case cases/example/case.json --skip-tts

# Full run including voice (GPU box: Kaggle/Modal; or --tts kokoro on CPU)
pip install chatterbox-tts soundfile numpy   # plus ffmpeg on PATH
python -m pipeline.run_milestone1 --case cases/example/case.json
```

All stages checkpoint to `runs/<run-id>/` and resume automatically — kill and
re-run at any point (built for Kaggle's 12h session cap).

## Pipeline stages (Milestone 1)

| Stage | What | Compute |
|---|---|---|
| 01_draft | Script from primary sources (frontier LLM) | API |
| 02_rewrite | Adversarial anti-cliche rewrite (frontier LLM) | API |
| 03_validate | Hard structural gates: cold open, act proportions, midpoint reversal, re-hook, open loops, ban list, honest ending | none |
| 04_grounding | Atomic claim decomposition + per-claim citation + NLI verdict (local LLM). Contradicted claim = hard fail | local |
| 05_direction | Per-sentence emotion → TTS params + pause map, seeded jitter | none |
| 06_tts | Chatterbox render (or Kokoro fallback) | GPU/CPU |
| 07_master | Two-pass loudnorm to -14 LUFS / -1 dBTP + QC gate | CPU |

## Adding a case

Create `cases/<name>/case.json`:

```json
{
  "title": "...",
  "contradiction": "the source conflict that fuels the midpoint reversal",
  "sources": [{ "id": "src1", "citation": "...", "text": "primary source excerpt" }]
}
```

## Pipeline entry points

| Stage | Command |
|---|---|
| Layer 1 research | `python -m pipeline.run_research --query "..." --title "..."` |
| M1 script + voice | `python -m pipeline.run_milestone1 --case cases/<name>/case.json` |
| M2 footage | `python -m pipeline.run_milestone2 --run-id <run-id>` |
| M3 edit + assembly | `python -m pipeline.run_milestone3 --run-id <run-id>` |
| M4 QC + packaging | `python -m pipeline.run_milestone4 --run-id <run-id>` |
| M5 feedback (~1 wk post-upload) | `python -m pipeline.run_milestone5 --run-id <run-id> --video-id <yt-id>` |

See [PLAN.md](PLAN.md) for the full blueprint and milestone status.
