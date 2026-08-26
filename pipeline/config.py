"""Environment-driven configuration.

LLM split by stakes (PLAN.md Layer 2):
- FRONTIER_* : quality-critical, low-volume calls (script draft, adversarial rewrite).
  Any OpenAI-compatible endpoint (free-tier API, AI gateway, etc).
- LOCAL_*    : mechanical, high-volume calls (claim decomposition, NLI, emotion tagging).
  Defaults to a local Ollama server's OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LLMEndpoint:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Config:
    workdir: Path
    frontier: LLMEndpoint
    local: LLMEndpoint
    # Voice (PLAN.md Layer 3)
    tts_engine: str = "chatterbox"  # "chatterbox" | "kokoro"
    kokoro_voice: str = "am_michael"
    chatterbox_device: str = field(default_factory=lambda: os.getenv("TTS_DEVICE", "cuda"))
    # Mastering spec (PLAN.md Layer 7) - hard QC gate
    target_lufs: float = -14.0
    target_true_peak_db: float = -1.0
    lufs_tolerance: float = 1.0
    # Script constraints (PLAN.md Layer 2)
    target_minutes: float = 10.0
    words_per_minute: int = 150
    rehook_window_s: tuple[float, float] = (60.0, 90.0)
    max_loop_gap_s: float = 180.0


def load_config(workdir: str | Path = "runs") -> Config:
    return Config(
        workdir=Path(workdir),
        frontier=LLMEndpoint(
            base_url=os.getenv("FRONTIER_API_BASE", "https://api.openai.com/v1"),
            api_key=os.getenv("FRONTIER_API_KEY", ""),
            model=os.getenv("FRONTIER_MODEL", "gpt-4o"),
        ),
        local=LLMEndpoint(
            base_url=os.getenv("LOCAL_API_BASE", "http://localhost:11434/v1"),
            api_key=os.getenv("LOCAL_API_KEY", "ollama"),
            model=os.getenv("LOCAL_MODEL", "qwen2.5:14b-instruct"),
        ),
        tts_engine=os.getenv("TTS_ENGINE", "chatterbox"),
    )
