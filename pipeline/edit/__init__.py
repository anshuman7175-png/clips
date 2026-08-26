"""Milestone 3 - the edit engine (PLAN.md Layer 6).

align      : WhisperX forced alignment (sub-100ms word timestamps) with a
             deterministic synthetic fallback for offline/CPU runs.
timeline   : narration timeline - beats/sentences with editorial pause beats.
edl        : EDL builder - J/L-cut grammar, emotion-modulated durations,
             word-precise cuts, match-cut opportunism, seeded jitter.
rule_of_six: Murch Rule-of-Six scorer; picks the best EDL among jitter seeds.
remotion_export: Remotion props + FFmpeg fallback assembly script.
"""
