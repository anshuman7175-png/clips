"""Aesthetic + technical QC gates before publish (PLAN.md Layer 8).

Every failure maps back to an EDL event id and timestamp with the failing
attribute - the QC layer never says "bad video", it says "e017 at 312.4s
fails sharpness". That mapping is what makes the human taste pass short.
"""

from .aesthetic import build_probe_plan, evaluate_aesthetics
from .technical import map_defects_to_events, run_technical_checks
from .taste_pass import write_checklist

__all__ = ["build_probe_plan", "evaluate_aesthetics",
           "map_defects_to_events", "run_technical_checks", "write_checklist"]
