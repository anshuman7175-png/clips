"""Offline synthetic test for Milestone 3 (no GPU, no network, no ffmpeg run).

Exercises the full edit engine on a synthetic script + shotlist:
alignment fallback -> timeline -> EDL grammar (J/L-cuts, reveals, pauses,
jitter, match-cuts) -> Rule-of-Six selection -> Remotion/FFmpeg export.

Run: python scripts/test_m3_synthetic.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.edit.align import synthetic_alignment
from pipeline.edit.edl import (EditContractError, J_CUT_LEAD_RANGE,
                               _order_for_match_cuts, build_edl)
from pipeline.edit.remotion_export import export_props, write_ffmpeg_fallback
from pipeline.edit.rule_of_six import WEIGHTS, best_edl, score_edl
from pipeline.edit.timeline import EDITORIAL_PAUSE_S, build_timeline, snap
from pipeline.voice_director import direct

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --- synthetic inputs --------------------------------------------------------

SCRIPT = {"beats": [
    {"id": "b01", "role": "cold_open", "emotion": {"tension": 0.8, "gravity": 0.3, "pace": 0.7},
     "text": "At 3:42 in the morning the phones went dead. Nobody at the exchange noticed for six minutes."},
    {"id": "b02", "role": "establish", "emotion": {"tension": 0.3, "gravity": 0.5, "pace": 0.4},
     "text": "The town of Marlin had one switchboard and two operators. Both kept meticulous logs of every call."},
    {"id": "b03", "role": "destabilize", "emotion": {"tension": 0.6, "gravity": 0.6, "pace": 0.5},
     "text": "The logs from that night are missing. Every other night from that decade survives intact."},
    {"id": "b04", "role": "investigate", "emotion": {"tension": 0.5, "gravity": 0.5, "pace": 0.6},
     "text": "Court records point to a maintenance order filed the day before. The order was signed by a man who did not work there."},
    {"id": "b05", "role": "midpoint_reversal", "emotion": {"tension": 0.9, "gravity": 0.8, "pace": 0.4},
     "text": "The signature matches the county clerk himself. He had reported the logs stolen a week earlier."},
    {"id": "b06", "role": "honest_ending", "emotion": {"tension": 0.3, "gravity": 0.7, "pace": 0.3},
     "text": "No charge was ever filed and the clerk retired quietly. What happened in those six minutes is still an open question."},
]}


def make_manifest() -> dict:
    """Shotlist manifest matching match_gate output. b02's last shot is a
    document so the b02->b03 boundary must L-cut; b05 has a generated
    dramatization; b06 has NO shots (coverage-gap path)."""
    def shot(sid, bid, kind, decision, dur, asset=None):
        return {"shot_id": sid, "beat_id": bid, "kind": kind, "query": "q",
                "duration_s": dur, "evidentiary": kind == "document",
                "decision": decision, "asset_id": asset, "score": 0.5,
                "license_class": "cc", "attribution": "Archive / CC-BY",
                "label_required": decision == "generate"}
    return {"shots": [
        shot("b01-s1", "b01", "footage", "matched", 5, "archive:night_town"),
        shot("b01-s2", "b01", "still", "matched", 4, "wikimedia:switchboard"),
        shot("b02-s1", "b02", "still", "matched", 5, "wikimedia:marlin_map"),
        shot("b02-s2", "b02", "document", "matched", 6, "loc:call_logs"),
        shot("b03-s1", "b03", "document", "matched", 6, "loc:log_ledger"),
        shot("b03-s2", "b03", "still", "matched", 4, "archive:empty_shelf"),
        shot("b04-s1", "b04", "document", "matched", 6, "courtlistener:maint_order"),
        shot("b04-s2", "b04", "still", "matched", 4, "wikimedia:county_office"),
        shot("b05-s1", "b05", "footage", "generate", 6, None),
        shot("b05-s2", "b05", "document", "matched", 5, "loc:theft_report"),
        # b06 intentionally has no shots.
    ], "stats": {}, "rewrite_requests": [], "generation_jobs": []}


def main() -> None:
    print("== M3 synthetic edit-engine test ==\n")

    # --- A: alignment fallback ----------------------------------------------
    print("A. synthetic alignment")
    direction = direct(SCRIPT, seed=7)
    # Deterministic editorial pause on the reveal line (b05 last sentence).
    b05_lines = [l for l in direction if l["beat_id"] == "b05"]
    b05_lines[-1]["pause_after_s"] = 1.3
    alignment = synthetic_alignment(direction)

    words = alignment["words"]
    check("A1 words strictly monotonic",
          all(a["end"] <= b["start"] + 1e-6 for a, b in zip(words, words[1:])))
    check("A2 words nested in their lines",
          all(any(l["line_id"] == w["line_id"] and
                  l["start"] - 1e-6 <= w["start"] and w["end"] <= l["end"] + 1e-6
                  for l in alignment["lines"]) for w in words))
    check("A3 every beat has lines",
          {l["beat_id"] for l in alignment["lines"]} == {b["id"] for b in SCRIPT["beats"]})
    check("A4 alignment deterministic",
          synthetic_alignment(direction) == alignment)

    # --- T: timeline ----------------------------------------------------------
    print("\nT. timeline")
    timeline = build_timeline(SCRIPT, alignment)
    check("T1 beats contiguous and ordered",
          all(abs(a["end"] - b["start"]) < 1e-6
              for a, b in zip(timeline["beats"], timeline["beats"][1:])))
    check("T2 editorial pause detected on reveal line",
          any(s["editorial_pause"] and s["pause_after_s"] >= EDITORIAL_PAUSE_S
              for b in timeline["beats"] if b["beat_id"] == "b05"
              for s in b["sentences"]))
    check("T3 reveal flagged on midpoint_reversal last sentence",
          [s["is_reveal"] for b in timeline["beats"] if b["beat_id"] == "b05"
           for s in b["sentences"]][-1] is True)
    check("T4 snap respects max drift",
          snap(100.0, timeline["word_boundaries"]) == 100.0)

    # --- E: EDL grammar -------------------------------------------------------
    print("\nE. EDL grammar")
    manifest = make_manifest()
    edl = build_edl(timeline, manifest, seed=3)
    events = edl["events"]
    beats = {b["beat_id"]: b for b in timeline["beats"]}

    check("E1 events sorted, positive durations",
          all(e["video_out"] > e["video_in"] for e in events) and
          all(a["video_in"] <= b["video_in"] for a, b in zip(events, events[1:])))

    j_cuts = [e for e in events if e["cut_style"] == "j_cut"]
    lo, hi = J_CUT_LEAD_RANGE
    check("E2 J-cuts exist at beat transitions with lead in range",
          len(j_cuts) >= 2 and
          all(lo <= e["audio_lead_s"] <= hi and
              abs(e["video_in"] - (beats[e["beat_id"]]["start"] + e["audio_lead_s"])) < 1e-6
              for e in j_cuts))

    b03_first = next(e for e in events if e["beat_id"] == "b03")
    b02_doc = [e for e in events if e["beat_id"] == "b02"][-1]
    check("E3 document before beat boundary L-cuts (lingers under next narration)",
          b02_doc["cut_style"] == "l_cut" and
          b02_doc["video_out"] > beats["b02"]["end"] and
          abs(b03_first["video_in"] - b02_doc["video_out"]) < 1e-6)

    b05_events = [e for e in events if e["beat_id"] == "b05"]
    check("E4 reveal held with ambience swell",
          any(e["hold"] and e["ambience_swell"] for e in b05_events))

    check("E5 generated shot carries DRAMATIZATION label",
          all(e["label"] == "DRAMATIZATION" for e in events
              if e["source"] == "generated") and
          any(e["source"] == "generated" for e in events))

    check("E6 coverage gap extends previous event across b06",
          not any(e["beat_id"] == "b06" for e in events) and
          any(e["coverage_gap"] and e["video_out"] >= beats["b06"]["end"] - 1e-6
              for e in events))

    check("E7 same seed -> identical EDL",
          build_edl(timeline, manifest, seed=3) == edl)
    check("E8 different seed -> different cut pattern (jitter)",
          build_edl(timeline, manifest, seed=4) != edl)

    bad = {**manifest, "shots": manifest["shots"][:1] + [
        {**manifest["shots"][0], "shot_id": "bX", "decision": "rewrite_line"}]}
    try:
        build_edl(timeline, bad, seed=1)
        check("E9 rewrite_line in manifest raises", False)
    except EditContractError:
        check("E9 rewrite_line in manifest raises", True)

    unlabeled = {**manifest, "shots": manifest["shots"][:1] + [
        {**manifest["shots"][0], "shot_id": "bY", "decision": "generate",
         "asset_id": None, "label_required": False}]}
    try:
        build_edl(timeline, unlabeled, seed=1)
        check("E10 unlabeled generated shot raises", False)
    except EditContractError:
        check("E10 unlabeled generated shot raises", True)

    # Match-cut opportunism: similarity should reorder the middle of a 3-chain.
    shots3 = [{"shot_id": s, "asset_id": f"a:{s}", "kind": "still",
               "decision": "matched"} for s in ("A", "B", "C")]
    sim = {("a:A", "a:C"): 0.9, ("a:C", "a:B"): 0.8, ("a:A", "a:B"): 0.1,
           ("a:B", "a:C"): 0.1, ("a:C", "a:A"): 0.1, ("a:B", "a:A"): 0.1}
    ordered = _order_for_match_cuts(shots3, lambda a, b: sim.get((a, b), 0.0))
    check("E11 match-cut ordering maximizes adjacent similarity",
          [s["shot_id"] for s in ordered] == ["A", "C", "B"])

    # --- R: Rule of Six --------------------------------------------------------
    print("\nR. Rule-of-Six")
    check("R1 Murch weights sum to 1.0", abs(sum(WEIGHTS.values()) - 1.0) < 1e-9)
    score = score_edl(edl, timeline)
    check("R2 all components in [0,1]",
          all(0.0 <= score[k] <= 1.0 for k in WEIGHTS))
    winner, report = best_edl(timeline, manifest, seeds=list(range(8)))
    check("R3 winner has max candidate score",
          report["winner_score"]["total"] ==
          max(c["total"] for c in report["candidates"]) and
          len(report["candidates"]) == 8)
    check("R4 winner is a valid EDL from a tried seed",
          winner["seed"] in list(range(8)) and winner["events"])

    # --- X: export --------------------------------------------------------------
    print("\nX. assembly export")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        media = {"archive:night_town": str(tmp / "night.mp4"),
                 "loc:call_logs": str(tmp / "logs.jpg")}
        props = export_props(edl, None, media, tmp / "props.json")
        check("X1 one sequence per event, frame-positive",
              len(props["sequences"]) == len(events) and
              all(s["durationFrames"] >= 1 for s in props["sequences"]))
        check("X2 unresolved media exported as missing slates",
              all(s["missing"] == (s["src"] is None) for s in props["sequences"]) and
              any(s["missing"] for s in props["sequences"]))
        check("X3 stills classified by extension",
              next(s for s in props["sequences"]
                   if s["src"] and s["src"].endswith(".jpg"))["isStill"] is True and
              next(s for s in props["sequences"]
                   if s["src"] and s["src"].endswith(".mp4"))["isStill"] is False)
        check("X4 props file is valid json",
              json.loads((tmp / "props.json").read_text())["fps"] == 30)

        script_path = write_ffmpeg_fallback(edl, None, media, tmp / "assemble.sh")
        text = script_path.read_text()
        check("X5 ffmpeg script covers every event",
              text.count("seg") >= len(events) and "concat" in text)
        check("X6 dramatization label burned in fallback",
              "drawtext=text=DRAMATIZATION" in text)
        check("X7 script executable", script_path.stat().st_mode & 0o111 != 0)

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
