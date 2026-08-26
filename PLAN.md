# The Definitive Plan (v4) — Fully-Free Automated Documentary Channel

This is the consolidated, final blueprint after four rounds of research. It is the single
source of truth for this repository. Every pipeline stage built here must trace back to a
section of this document.

Goal: produce long-form (8–15 min) mystery/documentary videos that are **substantively
excellent** (primary-source driven), **indistinguishable from human-made** in voice and
edit, **monetization-safe**, and produced on a **fully free stack** at 1–2 videos/week.

---

## Layer 0 — Non-negotiable principles

1. **Documents first.** Scripts are written FROM primary sources, never from vibes.
   Every factual claim must be grounded in a retrieved source chunk (see Layer 5).
2. **Novelty gate.** A case is only greenlit if the pipeline can surface facts/framings
   not present in existing top-ranking videos on the case.
3. **Footage priority order (never reversed):**
   real documents > real archival footage > stock > generated reconstruction (labeled).
4. **No fake resolution.** Endings state honestly what the documents can and cannot say.
5. **One human taste pass per video** (~20 min, fixed checklist). The pipeline's job is to
   make this review short, not to eliminate it.
6. **Item-level license logging** for every asset (source-level assumptions are not enough).

---

## Layer 1 — Substance (research & case selection)

- **Primary source APIs (free):** CourtListener/RECAP, CIA CREST, Chronicling America,
  FBI Vault, Library of Congress, NARA catalog.
- **Case scout:** harvests candidate cases; builds entity/timeline graphs (local LLM).
- **Contradiction detection** (frontier LLM, few calls): finds where sources disagree —
  this output fuels both the midpoint reversal (Layer 4) and the novelty gate.
- **Novelty gate:** compares extracted facts against transcripts of existing top videos;
  case fails if it cannot beat existing coverage on at least 3 substantive points.

## Layer 2 — Script engine

- **LLM split by stakes:**
  - Frontier model (free-tier APIs, few calls/video): script draft, adversarial rewrite,
    contradiction detection.
  - Local model (Ollama — Gemma/Qwen class): entity extraction, timeline graphs, emotion
    tagging, ban-list enforcement, QC checks, claim decomposition.
- **Three-act spine (enforced, not suggested):**
  - Act 1 (0–20%): cold open on the most arresting verified fact (first sentence = the
    surprising fact; no greetings, no "today we'll explore") → person/place → destabilizing event.
  - Act 2 (20–75%): investigation with a **mandatory midpoint reversal** — the moment the
    obvious explanation breaks (from contradiction detection).
  - Act 3 (75–100%): synthesis; honest ending; no fake resolution.
- **Retention constraints (hard validators):**
  - Re-hook at ~60–90s introducing new stakes or a contradiction.
  - Open-loop scheduler: a loop must be planted or paid off every 90–180 seconds.
    A script that goes 4 minutes without a loop event fails QC automatically.
  - Misdirection budget: at most one deliberate misdirect per video, and it must be resolved.
- **Adversarial rewrite pass:** a second frontier call attacks the draft for clichés,
  AI-isms (ban list), padding, and unearned claims.
- **Per-sentence emotion vectors** annotated for the Voice Director and the edit engine.

## Layer 3 — Voice

- **Primary TTS: Chatterbox (MIT).** Per-sentence rendering with its emotion-exaggeration
  control driven directly by the script's emotion vectors. Paralinguistic tags where supported.
- **Fallback: Kokoro (Apache 2.0)** — CPU-only drafts and high-throughput preview renders.
- **Ruled out:** F5-TTS (CC-BY-NC weights — not usable on a monetized channel);
  Fish Speech (unclear commercial license).
- **Voice Director:** converts emotion vectors into per-sentence render params
  (exaggeration, pace, pause length), inserts breath/pause splices between sentences.
- **WhisperX forced alignment** on the final audio → sub-100ms word timestamps, used by
  the edit engine (word-precise cuts), captions, and emphasis-driven effects.

## Layer 4 — Visuals

- **Footage matching gate:**
  - **X-CLIP** for video clips (temporal action match).
  - **SigLIP 2** for stills (documents, photos, maps — the majority of assets for obscure cases).
  - Literal-match threshold; on failure → either rewrite the script line OR generate.
- **Generated B-roll: Wan 2.2 (Apache 2.0)** — atmosphere/reconstruction shots ONLY,
  labeled on screen ("dramatization"), never implied as real evidence.
- **Upscale:** Real-ESRGAN for low-res archival assets.
- **Unification LUT / grade pass** so generated, archival, and stock shots grade-match.
- **Sources:** LoC National Screening Room and NARA (safest); Internet Archive/Prelinger
  (mixed — item-level checks; watch for copyrighted music inside PD films); British Pathé
  is NOT free. Pexels/Pixabay for stock. **BBC Sound Effects removed** (non-commercial only).

## Layer 5 — Verification (enforced contract, not principle)

- **Atomic claim decomposition** of the final script (local LLM).
- **Per-claim grounding:** every claim must cite a specific retrieved chunk of a primary
  source. Empty retrieval → auto-flag.
- **NLI check** (entailed / contradicted / neutral) between claim and cited passage
  (small local model).
- Output: a **citation manifest** per video (also used in descriptions for trust).

## Layer 6 — Edit

- **Rule-of-Six EDL scorer** (emotion > story > rhythm > eye-trace > 2D > 3D) with jitter
  injection to defeat templating.
- **Edit grammar:**
  - **J-cuts by default** at scene transitions (next audio leads picture 0.5–1.5s).
  - **L-cuts** when a document close-up should linger under moving narration.
  - **Pause-driven rhythm:** WhisperX inter-sentence silences are editorial beats
    (hold the shot, ambience swell), not dead air.
  - Emotion-modulated shot duration: high-tension → faster cuts; revelations → held shot.
  - Match-cut opportunism via embedding similarity between adjacent shots.
- **Word-level cut precision** from WhisperX alignment.

## Layer 7 — Sound & music

- **4-layer mix:** dialogue / music bed / ambience / spot effects.
- **SFX:** Freesound CC0 + YouTube Audio Library ONLY.
- **Music:** generated instrumental ambient beds (ACE-Step MIT / YuE Apache) — zero
  Content ID risk; YouTube Audio Library as fallback. Instrumental-only (lowest legal risk).
- **Mastering spec (hard QC gate):**
  - **-14 LUFS integrated, -1 dBTP true peak** (FFmpeg `loudnorm`, two-pass).
  - Dialogue anchored ~-16 to -15 LUFS short-term; music 15–20 dB under dialogue;
    ambience below that. Balance BEFORE loudnorm.
  - Renders outside spec fail automatically.

## Layer 8 — Assembly & QC

- **Remotion** (free for individuals) → FFmpeg encode.
- **Aesthetic QC gates before publish:**
  - **Peak-End-Net** (MIT) — aesthetic score + 10 attributes (Peak-End rule: peak moment
    and ending dominate perceived quality — exactly what Act 3 and the midpoint control).
  - **VQAThinker** — no-reference technical VQA (bad upscales, artifacts, jarring generated shots).
  - Failures map back to EDL timestamps with the failing attribute.
- **Human taste pass** (Layer 0, principle 5) with fixed checklist: cold open lands in 5s?
  midpoint genuinely surprises? any filler shot?

## Layer 9 — Packaging & publish

- **Thumbnails:** 3 genuinely distinct concepts per video (document close-up / location /
  typographic — not 3 tweaks). **Qwen-Image** (open weights, best text rendering) for
  typographic/composite thumbnails. Run **YouTube Test & Compare**; log outcomes locally.
- **Disclosure:** check YouTube's altered/synthetic content box whenever a video contains
  realistic generated reconstruction footage.
- **Persona rule:** the narrator narrates; it never presents as a human expert giving
  health/legal/finance advice (explicitly non-monetizable for AI personas).

## Layer 10 — Feedback loop (self-improvement)

- **Retention post-mortem** ~1 week after upload via YouTube Analytics API
  (`elapsedVideoTimeRatio`, `audienceWatchRatio`, `relativeRetentionPerformance`).
- Map retention dips to exact EDL beats (the pipeline generated the edit, so every
  timestamp maps to a script beat, shot type, and audio state).
- Aggregate findings across videos → adjust Rule-of-Six weights and footage-priority logic.

---

## Compute budget (fully free)

| Provider | Allocation |
|---|---|
| Kaggle | ~30 GPU-hrs/week (12h session cap) |
| Modal | $30/mo credits (~50 T4-hrs/mo) |
| Lightning AI | ~22 T4-hrs/mo |
| Colab free | unpredictable — do not build around it |

≈ 45–50 GPU-hrs/week combined. Per-video budget ≈ 4–8 GPU-hrs → **1–2 videos/week
sustainable fully free**, PROVIDED:
- The orchestrator is **checkpointed and resumable between stages** (Kaggle dies at 12h).
- The footage library is embedded **once, incrementally**, never per video.

## Legal ledger (quick reference)

| Asset/tool | Status |
|---|---|
| Chatterbox, Kokoro, Wan 2.2, X-CLIP, SigLIP 2, Real-ESRGAN, WhisperX, ACE-Step, Qwen-Image | Open/commercial-safe |
| F5-TTS | CC-BY-NC — BANNED |
| Fish Speech | Unclear — BANNED |
| BBC Sound Effects | Non-commercial — BANNED |
| British Pathé | Paid license — BANNED |
| Internet Archive / Prelinger | Item-level check required |
| Remotion | Free for individuals |

---

## Milestones

1. **M1 (this repo, first) — DONE:** Script engine (three-act spine + validators + claim grounding)
   → Voice Director → Chatterbox render → loudnorm master. Tests the riskiest assumption
   (voice quality) end to end.
2. **M2 — DONE:** Footage layer — X-CLIP/SigLIP gate, asset harvester, license ledger, Wan 2.2 branch.
   Implemented in `pipeline/footage/` (`run_milestone2.py` orchestrator; offline gate tests in
   `scripts/test_m2_synthetic.py`).
3. **M3:** Edit engine — Rule-of-Six EDL, J/L-cut grammar, WhisperX alignment, Remotion assembly.
4. **M4:** QC + packaging — aesthetic gates, thumbnails, disclosure, upload.
5. **M5:** Feedback loop — Analytics API post-mortems feeding the scorers.
