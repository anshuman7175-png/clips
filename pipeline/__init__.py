"""Milestone 1 pipeline: script engine -> voice director -> TTS -> loudnorm master.

Every module traces back to a section of PLAN.md. Stages are checkpointed and
resumable (Layer: Compute budget) so the pipeline survives Kaggle session caps.
"""

__version__ = "0.1.0"
