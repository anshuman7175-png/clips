"""Hard structural validators for the three-act spine (PLAN.md Layer 2).

The script is a list of beats. Each beat:
{
  "id": "b01",
  "act": 1 | 2 | 3,
  "role": "cold_open" | "establish" | "destabilize" | "investigate"
          | "midpoint_reversal" | "synthesis" | "honest_ending",
  "text": "...",
  "loop": {"action": "plant" | "payoff", "loop_id": "L1"} | null,
  "emotion": {"tension": 0-1, "wonder": 0-1, "gravity": 0-1, "pace": 0-1}
}

Validators FAIL (raise) rather than warn: a script that violates the spine
never reaches the voice stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BANNED_OPENERS = re.compile(
    r"^\s*(hello|hi|hey|welcome|today (we|i)|in this video|have you ever|imagine)",
    re.IGNORECASE,
)

# AI-ism ban list enforced on every beat (PLAN.md Layer 2, adversarial rewrite).
BAN_LIST = [
    "delve", "tapestry", "unravel the mystery", "shrouded in mystery",
    "little did they know", "to this day", "send chills", "chilling tale",
    "but that's not all", "the plot thickens", "dive into", "buckle up",
    "testament to", "in the annals of",
]


@dataclass
class ValidationError(Exception):
    beat_id: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] beat={self.beat_id}: {self.detail}"


def beat_duration_s(beat: dict, wpm: int) -> float:
    return len(beat["text"].split()) / wpm * 60.0


def validate_script(script: dict, cfg) -> list[str]:
    """Validate the full script structure. Returns a report; raises on failure."""
    beats = script["beats"]
    report: list[str] = []

    # --- Cold open: first sentence must be a concrete claim, no greetings.
    first = beats[0]
    if first["role"] != "cold_open":
        raise ValidationError(first["id"], "cold_open", "first beat must be role=cold_open")
    if BANNED_OPENERS.search(first["text"]):
        raise ValidationError(first["id"], "cold_open", "greeting/preamble opener detected")
    report.append("cold_open: OK")

    # --- Ban list on every beat.
    for beat in beats:
        lower = beat["text"].lower()
        for phrase in BAN_LIST:
            if phrase in lower:
                raise ValidationError(beat["id"], "ban_list", f"banned phrase: '{phrase}'")
    report.append("ban_list: OK")

    # --- Act proportions (Act1 0-20%, Act2 20-75%, Act3 75-100%, +-8% slack).
    total = sum(beat_duration_s(b, cfg.words_per_minute) for b in beats)
    act_time = {1: 0.0, 2: 0.0, 3: 0.0}
    for b in beats:
        act_time[b["act"]] += beat_duration_s(b, cfg.words_per_minute)
    for act, (lo, hi) in {1: (0.10, 0.28), 2: (0.45, 0.65), 3: (0.15, 0.33)}.items():
        frac = act_time[act] / total
        if not (lo <= frac <= hi):
            raise ValidationError(f"act{act}", "act_proportions",
                                  f"act {act} is {frac:.0%} of runtime, expected {lo:.0%}-{hi:.0%}")
    report.append(f"act_proportions: OK (1={act_time[1]/total:.0%} 2={act_time[2]/total:.0%} 3={act_time[3]/total:.0%})")

    # --- Midpoint reversal must exist inside Act 2.
    reversals = [b for b in beats if b["role"] == "midpoint_reversal"]
    if len(reversals) != 1:
        raise ValidationError("-", "midpoint_reversal", f"expected exactly 1, found {len(reversals)}")
    report.append("midpoint_reversal: OK")

    # --- Re-hook: a loop plant or reversal-grade beat inside the re-hook window.
    lo, hi = cfg.rehook_window_s
    t = 0.0
    rehook_found = False
    for b in beats:
        t_end = t + beat_duration_s(b, cfg.words_per_minute)
        if t_end >= lo and t <= hi and (b.get("loop") or b["role"] in ("destabilize", "midpoint_reversal")):
            rehook_found = True
        t = t_end
    if not rehook_found:
        raise ValidationError("-", "rehook", f"no loop/stakes beat inside {lo:.0f}-{hi:.0f}s window")
    report.append("rehook: OK")

    # --- Open-loop scheduler: max gap between loop events (PLAN.md: fails at 4 min gap).
    t, last_loop_t = 0.0, 0.0
    open_loops: set[str] = set()
    for b in beats:
        if b.get("loop"):
            if t - last_loop_t > cfg.max_loop_gap_s + 60:  # hard fail past 4 min
                raise ValidationError(b["id"], "open_loops",
                                      f"{t - last_loop_t:.0f}s since last loop event (max {cfg.max_loop_gap_s:.0f}s)")
            last_loop_t = t
            lid = b["loop"]["loop_id"]
            if b["loop"]["action"] == "plant":
                open_loops.add(lid)
            else:
                open_loops.discard(lid)
        t += beat_duration_s(b, cfg.words_per_minute)
    if open_loops:
        raise ValidationError("-", "open_loops", f"unresolved loops at end: {sorted(open_loops)}")
    report.append("open_loops: OK (all planted loops paid off)")

    # --- Honest ending: last beat must be honest_ending, no fake resolution.
    if beats[-1]["role"] != "honest_ending":
        raise ValidationError(beats[-1]["id"], "honest_ending", "last beat must be role=honest_ending")
    report.append("honest_ending: OK")

    report.append(f"runtime_estimate: {total/60:.1f} min")
    return report
