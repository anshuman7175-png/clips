"""Human taste pass checklist (PLAN.md Layer 0 principle 5 + Layer 8).

The pipeline's job is to make the ~20-minute human review SHORT, not to
eliminate it. This module writes a fixed-checklist markdown file with every
hotspot pre-located: exact timestamps for the cold open, the midpoint
reversal, coverage gaps, generated shots, flagged claims, and QC failures.
The reviewer jumps straight to timestamps instead of scrubbing."""

from __future__ import annotations

from pathlib import Path

FIXED_CHECKLIST = (
    "Cold open lands in the first 5 seconds? (first sentence = the surprising fact)",
    "Midpoint reversal genuinely surprises? (jump to timestamp below)",
    "Ending is honest - states what the documents can and cannot say?",
    "Any filler shot? (start with the coverage gaps and QC failures below)",
    "Every generated shot visibly labeled DRAMATIZATION?",
    "Narration reads human - no AI-isms that survived the ban list?",
)


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def build_checklist(timeline: dict, edl: dict, grounding: dict,
                    qc_reports: list[dict]) -> dict:
    """Pure: collect all hotspots. Returned dict is checkpointed; the
    markdown is a rendering of it."""
    beats = timeline["beats"]
    midpoint = next((b for b in beats if b["role"] == "midpoint_reversal"), None)
    reveals = [(b["beat_id"], s["end"]) for b in beats
               for s in b["sentences"] if s["is_reveal"]]
    gaps = [{"event_id": e["event_id"], "t": e["video_in"],
             "until": e["video_out"]}
            for e in edl["events"] if e["coverage_gap"]]
    generated = [{"event_id": e["event_id"], "t": e["video_in"]}
                 for e in edl["events"] if e["source"] == "generated"]
    qc_failures = [f for r in qc_reports for f in r.get("failures", [])]
    return {
        "duration_s": timeline["duration_s"],
        "cold_open_window": [0.0, 5.0],
        "midpoint_t": midpoint["start"] if midpoint else None,
        "reveal_timestamps": [t for _, t in reveals],
        "coverage_gaps": gaps,
        "generated_events": generated,
        "flagged_claims": grounding.get("flagged", []),
        "qc_failures": qc_failures,
    }


def write_checklist(checklist: dict, out_path: Path) -> Path:
    lines = ["# Human taste pass (~20 min, fixed checklist)", ""]
    lines += [f"Runtime: {_ts(checklist['duration_s'])}", ""]

    lines.append("## Checklist")
    for item in FIXED_CHECKLIST:
        lines.append(f"- [ ] {item}")
    lines.append("")

    lines.append("## Jump points")
    lines.append(f"- Cold open: 00:00-00:05")
    if checklist["midpoint_t"] is not None:
        lines.append(f"- Midpoint reversal: {_ts(checklist['midpoint_t'])}")
    for t in checklist["reveal_timestamps"]:
        lines.append(f"- Reveal (held shot): {_ts(t)}")
    lines.append("")

    if checklist["coverage_gaps"]:
        lines.append("## Coverage gaps (previous shot stretched - check for filler)")
        for g in checklist["coverage_gaps"]:
            lines.append(f"- {g['event_id']}: {_ts(g['t'])} -> {_ts(g['until'])}")
        lines.append("")

    if checklist["generated_events"]:
        lines.append("## Generated shots (verify DRAMATIZATION label on screen)")
        for g in checklist["generated_events"]:
            lines.append(f"- {g['event_id']} at {_ts(g['t'])}")
        lines.append("")

    if checklist["flagged_claims"]:
        lines.append("## Ungrounded claims flagged by Layer 5")
        for c in checklist["flagged_claims"]:
            lines.append(f"- {c}")
        lines.append("")

    if checklist["qc_failures"]:
        lines.append("## Automated QC failures (event / time / attribute)")
        for f in checklist["qc_failures"]:
            lines.append(f"- {f['event_id']} at {_ts(f['t'])}: {f['attribute']}"
                         + (f" ({f['detail']})" if f.get("detail") else ""))
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
