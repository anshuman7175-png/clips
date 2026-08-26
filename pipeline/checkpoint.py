"""Resumable, checkpointed stage runner (PLAN.md: Compute budget).

Each stage writes its JSON output to <workdir>/<run_id>/<stage>.json.
On re-run, completed stages are skipped, so a killed Kaggle session
resumes exactly where it stopped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable


class Run:
    def __init__(self, workdir: Path, run_id: str):
        self.dir = Path(workdir) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def stage_path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def is_done(self, name: str) -> bool:
        return self.stage_path(name).exists()

    def load(self, name: str) -> Any:
        return json.loads(self.stage_path(name).read_text(encoding="utf-8"))

    def save(self, name: str, data: Any) -> None:
        tmp = self.stage_path(name).with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.stage_path(name))

    def stage(self, name: str, fn: Callable[[], Any], force: bool = False) -> Any:
        """Run `fn` unless a checkpoint for `name` already exists."""
        if not force and self.is_done(name):
            print(f"[stage:{name}] checkpoint found, skipping")
            return self.load(name)
        print(f"[stage:{name}] running...")
        t0 = time.time()
        result = fn()
        self.save(name, result)
        print(f"[stage:{name}] done in {time.time() - t0:.1f}s")
        return result
