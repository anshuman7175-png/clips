"""Forced alignment (PLAN.md Layer 3 -> Layer 6 bridge).

Primary path: WhisperX on the mastered narration -> sub-100ms word timestamps.
Fallback path: deterministic synthetic alignment derived from the Voice
Director's per-line render params (pace proxy + spliced pauses). The synthetic
path exists so the WHOLE edit engine is testable offline on CPU - the EDL
grammar must not depend on which aligner produced the timestamps.

Output schema (identical for both paths):
{
  "aligner": "whisperx" | "synthetic",
  "duration_s": float,
  "lines": [{"line_id", "beat_id", "text", "start", "end", "pause_after_s"}],
  "words": [{"word", "start", "end", "line_id"}],
}
Inter-line silences (pause_after_s) are EDITORIAL BEATS, not dead air - the
EDL builder holds shots and swells ambience across them (Layer 6).
"""

from __future__ import annotations

from pathlib import Path

# Speaking-rate model for the synthetic path. Chatterbox's cfg_weight is a
# pace proxy (lower = slower); 0.55 is the Voice Director's neutral center.
_NEUTRAL_CFG = 0.55
_BASE_WPS = 2.5  # words/second at neutral pace (150 wpm)


def align(lines: list[dict], audio_path: Path | None = None,
          device: str = "cuda", force_synthetic: bool = False) -> dict:
    """lines: Voice Director output (05_direction.json)."""
    if not force_synthetic and audio_path is not None and Path(audio_path).exists():
        try:
            return whisperx_alignment(lines, Path(audio_path), device)
        except ImportError:
            print("  whisperx not installed - falling back to synthetic alignment")
    return synthetic_alignment(lines)


def synthetic_alignment(lines: list[dict]) -> dict:
    """Deterministic timing model from the Voice Director's render params."""
    out_lines, out_words = [], []
    t = 0.0
    for line in lines:
        words = line["text"].split()
        pace_scale = _NEUTRAL_CFG / max(0.30, float(line.get("cfg_weight", _NEUTRAL_CFG)))
        line_dur = (len(words) / _BASE_WPS) * pace_scale
        # Distribute by character weight (long words take longer to say).
        weights = [max(2, len(w)) for w in words]
        total_w = sum(weights) or 1
        start = t
        for w, wt in zip(words, weights):
            w_dur = line_dur * wt / total_w
            out_words.append({"word": w, "start": round(t, 3),
                              "end": round(t + w_dur, 3), "line_id": line["line_id"]})
            t += w_dur
        pause = float(line.get("pause_after_s", 0.4))
        out_lines.append({"line_id": line["line_id"], "beat_id": line["beat_id"],
                          "text": line["text"], "start": round(start, 3),
                          "end": round(t, 3), "pause_after_s": round(pause, 2)})
        t += pause
    return {"aligner": "synthetic", "duration_s": round(t, 3),
            "lines": out_lines, "words": out_words}


def whisperx_alignment(lines: list[dict], audio_path: Path, device: str) -> dict:
    """WhisperX transcribe + forced align, then snap words back onto the
    Voice Director's known line texts (we KNOW the script - alignment only
    supplies timing, never words)."""
    import whisperx  # noqa: F401 - GPU box only

    model = whisperx.load_model("large-v3", device, compute_type="float16")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=8)
    align_model, meta = whisperx.load_align_model(result["language"], device)
    aligned = whisperx.align(result["segments"], align_model, meta, audio, device)

    words = [{"word": w["word"].strip(), "start": round(float(w["start"]), 3),
              "end": round(float(w["end"]), 3)}
             for seg in aligned["segments"] for w in seg.get("words", [])
             if "start" in w and "end" in w]
    return _map_words_to_lines(lines, words)


def _map_words_to_lines(lines: list[dict], words: list[dict]) -> dict:
    """Greedy sequential mapping: consume aligned words per script line by
    word count. Robust to small ASR drift because the counts, not the
    spellings, drive the mapping."""
    out_lines, out_words = [], []
    cursor = 0
    for line in lines:
        n = len(line["text"].split())
        chunk = words[cursor:cursor + n]
        cursor += n
        if not chunk:
            continue
        for w in chunk:
            out_words.append({**w, "line_id": line["line_id"]})
        out_lines.append({"line_id": line["line_id"], "beat_id": line["beat_id"],
                          "text": line["text"], "start": chunk[0]["start"],
                          "end": chunk[-1]["end"],
                          "pause_after_s": float(line.get("pause_after_s", 0.4))})
    # Recompute measured pauses from actual inter-line silence where possible.
    for a, b in zip(out_lines, out_lines[1:]):
        a["pause_after_s"] = round(max(0.0, b["start"] - a["end"]), 2)
    duration = out_words[-1]["end"] + out_lines[-1]["pause_after_s"] if out_words else 0.0
    return {"aligner": "whisperx", "duration_s": round(duration, 3),
            "lines": out_lines, "words": out_words}
