"""Kokoro TTS (Apache 2.0) - CPU fallback / fast draft engine (PLAN.md Layer 3).

Kokoro has no emotion control, so the Voice Director's exaggeration is mapped
to speed only. Use for previews; final renders use Chatterbox.
Requires: pip install kokoro soundfile
"""

from __future__ import annotations

from pathlib import Path


class KokoroEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code="a")  # American English
        return self._pipeline

    def render(self, lines: list[dict], out_path: Path) -> dict:
        import numpy as np
        import soundfile as sf

        pipe = self._load()
        sr = 24000
        chunks: list[np.ndarray] = []
        for line in lines:
            # Map director pace proxy to Kokoro speed (0.85-1.1).
            speed = 0.85 + 0.5 * (line["cfg_weight"] - 0.3)
            audio_parts = [audio for _, _, audio in pipe(line["text"], voice=self.cfg.kokoro_voice, speed=speed)]
            audio = np.concatenate([np.asarray(a) for a in audio_parts])
            chunks.append(audio)
            chunks.append(np.zeros(int(line["pause_after_s"] * sr), dtype=audio.dtype))
            print(f"[tts] {line['line_id']} ({len(audio) / sr:.1f}s)")
        full = np.concatenate(chunks)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, full, sr)
        return {"path": str(out_path), "sample_rate": sr, "duration_s": round(len(full) / sr, 1)}
