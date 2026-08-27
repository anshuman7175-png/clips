"""Packaging + publish prep (PLAN.md Layer 9).

Thumbnails (3 concepts to A/B), altered-content disclosure, persona rule,
and the YouTube upload package. The upload itself is the last step and is
optional - everything else works offline."""

from .thumbnails import build_concepts, build_thumbnail_jobs, render_thumbnail_jobs
from .publish import (build_disclosure, build_metadata, build_upload_package,
                      persona_check)

__all__ = ["build_concepts", "build_thumbnail_jobs", "render_thumbnail_jobs",
           "build_disclosure", "build_metadata", "build_upload_package",
           "persona_check"]
