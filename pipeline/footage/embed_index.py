"""Incremental embedding library (PLAN.md compute budget: embed ONCE, never per video).

Two vector spaces, matching the gate's two models (Layer 4):
  "still" - SigLIP 2 image/text space (documents, photos, maps)
  "video" - X-CLIP video/text space (temporal action match on clips)

Persistence is a plain .npz + json manifest per space under the shared
library dir, so the index survives across runs, machines, and Kaggle
session deaths. `ensure_embedded` is hash-keyed by asset_id: an asset is
embedded exactly once in its lifetime.

Embedders are lazy optional imports (same pattern as the TTS engines):
the heavyweight models load only on the GPU box. EMBEDDER=hash selects a
deterministic dependency-free embedder so the entire gate logic is
testable offline - the tests exercise real cosine ranking, not mocks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pure-python fallback keeps hash-embedder tests dependency-free
    np = None


# ---------------------------------------------------------------- index

class EmbeddingIndex:
    def __init__(self, library_dir: Path, space: str):
        self.space = space
        self.dir = Path(library_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._vec_path = self.dir / f"{space}_vectors.json"
        self._vecs: dict[str, list[float]] = {}
        if self._vec_path.exists():
            self._vecs = json.loads(self._vec_path.read_text(encoding="utf-8"))

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self._vecs

    def __len__(self) -> int:
        return len(self._vecs)

    def add(self, asset_id: str, vector: list[float]) -> None:
        self._vecs[asset_id] = _l2_normalize(vector)

    def save(self) -> None:
        tmp = self._vec_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._vecs), encoding="utf-8")
        tmp.replace(self._vec_path)

    def search(self, query_vec: list[float], k: int = 20,
               allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        q = _l2_normalize(query_vec)
        scored = [
            (aid, _dot(q, v))
            for aid, v in self._vecs.items()
            if allowed_ids is None or aid in allowed_ids
        ]
        scored.sort(key=lambda t: -t[1])
        return scored[:k]


def _l2_normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------- embedders

class HashEmbedder:
    """Deterministic token-feature-hashing embedder (offline tests / CPU dry runs).

    Embeds TEXT only; assets are embedded from title+description metadata.
    Real semantic behavior (shared query/asset space, cosine ranking) without
    any model download, so gate thresholds and priority logic are testable.
    """

    dim = 256

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.sha1(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 20) % 2 == 0 else -1.0  # noqa: S324
        return vec

    def embed_asset(self, record, local_path: Path | None = None) -> list[float]:
        return self.embed_text(f"{record.title} {record.description}")


class SiglipEmbedder:
    """SigLIP 2 for stills/documents (PLAN.md Layer 4). GPU-box only."""

    def __init__(self, model_id: str = "google/siglip2-base-patch16-256", device: str = "cuda"):
        from transformers import AutoModel, AutoProcessor  # heavyweight, lazy
        import torch
        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def embed_text(self, text: str) -> list[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding="max_length").to(self.device)
        with self.torch.no_grad():
            return self.model.get_text_features(**inputs)[0].cpu().tolist()

    def embed_asset(self, record, local_path: Path | None = None) -> list[float]:
        from PIL import Image
        if not local_path:
            raise ValueError(f"{record.asset_id}: SigLIP needs a downloaded file")
        inputs = self.processor(images=Image.open(local_path).convert("RGB"),
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            return self.model.get_image_features(**inputs)[0].cpu().tolist()


class XClipEmbedder:
    """X-CLIP for video clips - temporal action match (PLAN.md Layer 4). GPU-box only."""

    frames = 8

    def __init__(self, model_id: str = "microsoft/xclip-base-patch32", device: str = "cuda"):
        from transformers import AutoModel, AutoProcessor
        import torch
        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def embed_text(self, text: str) -> list[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            return self.model.get_text_features(**inputs)[0].cpu().tolist()

    def embed_asset(self, record, local_path: Path | None = None) -> list[float]:
        from PIL import Image
        if not local_path:
            raise ValueError(f"{record.asset_id}: X-CLIP needs a downloaded file")
        frames = self._sample_frames(local_path)
        inputs = self.processor(videos=[frames], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            return self.model.get_video_features(**inputs)[0].cpu().tolist()

    def _sample_frames(self, path: Path) -> list:
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path),
                 "-vf", f"fps=1,select='not(mod(n\\,{max(1, 1)}))'",
                 "-frames:v", str(self.frames), f"{tmp}/f%02d.jpg"],
                check=True, capture_output=True,
            )
            files = sorted(Path(tmp).glob("f*.jpg"))
            imgs = [Image.open(f).convert("RGB") for f in files]
        # X-CLIP requires exactly `frames` frames: pad by repeating the last.
        while imgs and len(imgs) < self.frames:
            imgs.append(imgs[-1])
        return imgs


def get_embedder(space: str, name: str, device: str = "cuda"):
    if name == "hash":
        return HashEmbedder()
    if space == "still":
        return SiglipEmbedder(device=device)
    return XClipEmbedder(device=device)


def ensure_embedded(index: EmbeddingIndex, embedder, records, download_dir: Path,
                    downloader=None) -> int:
    """Embed only records not yet in the index (embed-once contract).

    `downloader(record, dest_dir) -> Path | None` fetches media when the
    embedder needs pixels; the HashEmbedder needs none. Returns count added.
    """
    added = 0
    for rec in records:
        if rec.asset_id in index:
            continue
        local = Path(rec.local_path) if rec.local_path else None
        if local is None and downloader is not None:
            local = downloader(rec, download_dir)
        try:
            index.add(rec.asset_id, embedder.embed_asset(rec, local))
            added += 1
        except Exception as err:  # noqa: BLE001 - one bad asset must not kill the batch
            print(f"[embed] SKIP {rec.asset_id}: {err}")
    if added:
        index.save()
    return added
