"""TTS engines (PLAN.md Layer 3): Chatterbox primary, Kokoro fallback."""

from __future__ import annotations


def get_engine(cfg):
    if cfg.tts_engine == "kokoro":
        from .kokoro_engine import KokoroEngine

        return KokoroEngine(cfg)
    from .chatterbox_engine import ChatterboxEngine

    return ChatterboxEngine(cfg)
