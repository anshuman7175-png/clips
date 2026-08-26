"""Chatterbox TTS (MIT) - primary engine (PLAN.md Layer 3).

Renders per-sentence with the Voice Director's exaggeration/cfg_weight params,
then concatenates with directed pauses. Requires: pip install chatterbox-tts
(GPU strongly recommended; run this stage on Kaggle/Modal free quota).
"""

from __future__ import annotations

from pathlib import Path


class ChatterboxEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is None:
            from chatterbox.tts import ChatterboxTTS

            self._model = ChatterboxTTS.from_pretrained(device=self.cfg.chatterbox_device)
        return self._model

    def render(self, lines: list[dict], out_path: Path) -> dict:
        import numpy as np
        import soundfile as sf
        import torch

        model = self._load()
        sr = model.sr
        chunks: list[np.ndarray] = []
        for line in lines:
            wav = model.generate(
                line["text"],
                exaggeration=line["exaggeration"],
                cfg_weight=line["cfg_weight"],
            )
            audio = wav.squeeze().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
            chunks.append(audio)
            chunks.append(np.zeros(int(line["pause_after_s"] * sr), dtype=audio.dtype))
            print(f"[tts] {line['line_id']} ({len(audio) / sr:.1f}s)")
        full = np.concatenate(chunks)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, full, sr)
        return {"path": str(out_path), "sample_rate": sr, "duration_s": round(len(full) / sr, 1)}
