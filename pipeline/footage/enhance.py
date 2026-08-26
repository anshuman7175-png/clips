"""Asset enhancement: Real-ESRGAN upscale + unification grade (PLAN.md Layer 4).

- upscale():  Real-ESRGAN for low-res archival assets. Prefers the
  realesrgan-ncnn-vulkan CLI binary (runs on any GPU incl. Kaggle's) and
  falls back to ffmpeg lanczos scaling with a warning (never silently -
  a bad upscale is exactly what VQAThinker flags in M4).
- unify_grade(): one LUT/grade chain applied to EVERY visual asset so
  archival, stock, and generated shots sit in the same world. Uses a
  .cube LUT when provided (UNIFY_LUT env), else a conservative filmic
  ffmpeg chain (lifted blacks, gentle desaturation, slight warm bias).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_FALLBACK_GRADE = (
    "eq=saturation=0.88:gamma=1.02,"
    "colorbalance=rm=0.02:bm=-0.02,"
    "curves=all='0/0.03 0.5/0.5 1/0.98'"
)


def upscale(src: Path, dst: Path, scale: int = 2) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    binary = shutil.which("realesrgan-ncnn-vulkan")
    if binary:
        subprocess.run([binary, "-i", str(src), "-o", str(dst), "-s", str(scale)],
                       check=True, capture_output=True)
        return {"path": str(dst), "method": "real-esrgan"}
    # Fallback is honest about being a fallback.
    print(f"[enhance] WARNING: realesrgan-ncnn-vulkan not on PATH; "
          f"lanczos fallback for {src.name} (flag for VQA review)")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-vf", f"scale=iw*{scale}:ih*{scale}:flags=lanczos", str(dst)],
        check=True, capture_output=True)
    return {"path": str(dst), "method": "lanczos-fallback", "vqa_flag": True}


def unify_grade(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lut = os.getenv("UNIFY_LUT", "")
    vf = f"lut3d='{lut}'" if lut and Path(lut).exists() else _FALLBACK_GRADE
    is_video = src.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".mpeg")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf]
    if is_video:
        cmd += ["-c:a", "copy"]
    cmd.append(str(dst))
    subprocess.run(cmd, check=True, capture_output=True)
    return {"path": str(dst), "grade": "lut3d" if "lut3d" in vf else "fallback-filmic"}
