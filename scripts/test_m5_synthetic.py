"""Offline synthetic test for Milestone 5 (no network, no OAuth, no GPU).

Exercises the whole feedback loop on the same synthetic case as the M3/M4
tests: synthetic retention curve with dips injected at KNOWN edit weaknesses
-> dip detection -> EDL/beat mapping -> component diagnosis -> cross-video
aggregation -> Rule-of-Six weight adjustment -> footage findings -> channel
state round-trip into the scorer.

Run: python scripts/test_m5_synthetic.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.edit.align import synthetic_alignment
from pipeline.edit.edl import build_edl
from pipeline.edit.rule_of_six import WEIGHTS, best_edl, score_edl
from pipeline.edit.timeline import build_timeline
from pipeline.feedback.aggregate import (LEARNING_RATE, WEIGHT_FLOOR,
                                         adjust_weights,
                                         aggregate_postmortems,
                                         footage_findings,
                                         load_channel_weights,
                                         write_channel_state)
from pipeline.feedback.analytics import normalize_curve, synthetic_curve
from pipeline.feedback.postmortem import (build_postmortem, exposure_stats,
                                          find_dips, find_spikes, map_dip)
from pipeline.voice_director import direct

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --- synthetic inputs (same case as the M3/M4 tests) --------------------------

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


def make_manifest(with_gap: bool = True) -> dict:
    def shot(sid, bid, kind, decision, dur, asset=None):
        return {"shot_id": sid, "beat_id": bid, "kind": kind,
                "query": f"query for {sid}", "duration_s": dur,
                "evidentiary": kind == "document", "decision": decision,
                "asset_id": asset, "score": 0.5, "license_class": "cc",
                "attribution": "Archive / CC-BY",
                "label_required": decision == "generate"}
    shots = [
        shot("b01-s1", "b01", "footage", "matched", 5, "archive:night_town"),
        shot("b01-s2", "b01", "still", "matched", 4, "wikimedia:switchboard"),
        shot("b02-s1", "b02", "still", "matched", 5, "wikimedia:marlin_map"),
        shot("b02-s2", "b02", "document", "matched", 6, "loc:call_logs"),
        shot("b03-s1", "b03", "document", "matched", 6, "loc:log_ledger"),
        shot("b04-s1", "b04", "document", "matched", 6, "courtlistener:maint_order"),
        shot("b04-s2", "b04", "still", "matched", 4, "wikimedia:county_office"),
        shot("b05-s1", "b05", "footage", "generate", 6, None),
        shot("b05-s2", "b05", "document", "matched", 5, "loc:theft_report"),
    ]
    if not with_gap:  # b06 covered in the control video
        shots.append(shot("b06-s1", "b06", "still", "matched", 6, "archive:sunset"))
    return {"shots": shots, "stats": {}, "rewrite_requests": [], "generation_jobs": []}


def make_edit(with_gap: bool = True, seed: int = 3) -> tuple[dict, dict]:
    direction = direct(SCRIPT, seed=7)
    alignment = synthetic_alignment(direction)
    timeline = build_timeline(SCRIPT, alignment)
    edl = build_edl(timeline, make_manifest(with_gap), seed=seed)
    return timeline, edl


def ratio_of(edl: dict, t: float) -> float:
    return t / edl["duration_s"]


def main() -> None:
    print("== M5 synthetic feedback-loop test ==\n")

    timeline, edl = make_edit(with_gap=True)
    events = edl["events"]
    duration = edl["duration_s"]

    # Locate the known weaknesses to inject dips exactly there.
    gen_event = next(e for e in events if e["source"] == "generated")
    gap_event = next(e for e in events if e["coverage_gap"])
    gen_mid = ratio_of(edl, (gen_event["video_in"] + gen_event["video_out"]) / 2)
    gap_mid = ratio_of(edl, (gap_event["video_in"] + gap_event["video_out"]) / 2)

    # --- R: curve plumbing ----------------------------------------------------
    print("R. retention curve")
    curve = synthetic_curve(dips=[(gen_mid, 0.10, 0.04), (gap_mid, 0.08, 0.04)],
                            spikes=[(0.05, 0.03)])
    check("R1 curve has 100 buckets, monotone x",
          len(curve) == 100 and
          all(a["x"] < b["x"] for a, b in zip(curve, curve[1:])))
    api_rows = [[p["x"], p["watch"], 0.5] for p in curve]
    check("R2 normalize_curve round-trips API rows and dict rows",
          normalize_curve(api_rows)[0]["watch"] == curve[0]["watch"] and
          normalize_curve(curve)[3]["x"] == curve[3]["x"])
    try:
        normalize_curve([])
        check("R3 empty curve rejected", False)
    except ValueError:
        check("R3 empty curve rejected", True)

    # --- D: dip detection -------------------------------------------------------
    print("\nD. dip detection")
    dips = find_dips(curve)
    check("D1 exactly the 2 injected dips found (baseline decay ignored)",
          len(dips) == 2, f"found {len(dips)}")
    check("D2 dips sorted by depth, deepest ~0.10",
          dips and abs(dips[0]["depth"] - 0.10) < 0.03)
    flat = synthetic_curve()  # decay only, no dips
    check("D3 clean decay produces no dips", find_dips(flat) == [])
    spikes = find_spikes(curve)
    check("D4 rewatch spike detected near its center",
          any(abs(s["ratio"] - 0.05) < 0.03 for s in spikes))

    # --- M: EDL mapping ---------------------------------------------------------
    print("\nM. EDL mapping + diagnosis")
    mapped = [map_dip(d, edl, timeline) for d in dips]
    by_depth = {round(m["depth"], 1): m for m in mapped}
    gen_dip = next(m for m in mapped if gen_event["event_id"] in m["event_ids"])
    gap_dip = next(m for m in mapped if gap_event["event_id"] in m["event_ids"])
    check("M1 dips land on the events they were injected at",
          gen_dip is not None and gap_dip is not None and gen_dip != gap_dip)
    check("M2 generated dip diagnosed as space_3d with source attribute",
          "space_3d" in gen_dip["components"] and
          "source:generated" in gen_dip["attributes"])
    check("M3 coverage-gap dip diagnosed as story with gap flag",
          "story" in gap_dip["components"] and
          "flag:coverage_gap" in gap_dip["attributes"])
    check("M4 timestamps and beats resolved",
          all(0 <= m["t_start"] < m["t_end"] <= duration + 2 and m["beat_ids"]
              for m in mapped))

    # --- E: exposure ------------------------------------------------------------
    print("\nE. exposure stats")
    exposure = exposure_stats(edl)
    check("E1 attribute seconds sum sanely against runtime",
          abs(exposure["by_attribute"]["source:generated"] -
              (gen_event["video_out"] - gen_event["video_in"])) < 0.01 and
          exposure["total_s"] > 0)

    # --- A: aggregation + weight adjustment ---------------------------------------
    print("\nA. aggregation")
    pm1 = build_postmortem("vid_one", curve, edl, timeline)
    # Second video: same weaknesses dip again -> a channel-level pattern.
    timeline2, edl2 = make_edit(with_gap=True, seed=9)
    gen2 = next(e for e in edl2["events"] if e["source"] == "generated")
    gen2_mid = (gen2["video_in"] + gen2["video_out"]) / 2 / edl2["duration_s"]
    curve2 = synthetic_curve(dips=[(gen2_mid, 0.12, 0.04)])
    pm2 = build_postmortem("vid_two", curve2, edl2, timeline2)

    agg = aggregate_postmortems([pm1, pm2])
    check("A1 both videos folded, pressure on space_3d + story",
          agg["n_videos"] == 2 and
          agg["component_pressure"].get("space_3d", 0) > 0 and
          agg["component_pressure"].get("story", 0) > 0)
    check("A2 generated lift exceeds archival lift (exposure-normalized)",
          agg["attribute_lift"]["source:generated"] >
          agg["attribute_lift"].get("source:archival", 0.0))

    weights = adjust_weights(dict(WEIGHTS), agg)
    check("A3 weights renormalize to 1 with floor respected",
          abs(sum(weights.values()) - 1.0) < 0.01 and
          all(v >= WEIGHT_FLOOR for v in weights.values()))
    check("A4 implicated components gain weight, bounded by learning rate",
          weights["space_3d"] > WEIGHTS["space_3d"] and
          all(weights[k] <= WEIGHTS[k] * (1 + LEARNING_RATE) + 1e-9
              for k in WEIGHTS))
    check("A5 no pressure -> weights unchanged",
          adjust_weights(dict(WEIGHTS), {"component_pressure": {},
                                         "attribute_lift": {}, "n_videos": 0})
          == {k: round(v, 4) for k, v in WEIGHTS.items()})

    findings = footage_findings(agg)
    check("A6 footage finding targets generated shots",
          any("generated" in f["finding"] for f in findings))
    single = footage_findings(aggregate_postmortems([pm1]))
    check("A7 one video is an anecdote - no logic changes recommended",
          len(single) == 1 and "more post-mortems" in single[0]["action"])

    # --- S: channel state round-trip ------------------------------------------------
    print("\nS. channel state -> scorer")
    with tempfile.TemporaryDirectory() as td:
        state_path = Path(td) / "feedback" / "channel_state.json"
        write_channel_state(state_path, agg, weights, findings)
        write_channel_state(state_path, agg, weights, findings)  # second cycle
        state = json.loads(state_path.read_text())
        check("S1 state persists weights, findings, bounded history",
              state["weights"] == weights and state["findings"] and
              len(state["history"]) == 2)
        loaded = load_channel_weights(state_path)
        check("S2 loader returns the adjusted weights",
              loaded == weights)
        check("S3 loader returns None when no feedback exists",
              load_channel_weights(Path(td) / "nope.json") is None)

        # The loop actually closes: the scorer accepts adjusted weights and
        # they can change which candidate EDL wins.
        manifest = make_manifest(with_gap=True)
        base_best, base_report = best_edl(timeline, manifest, list(range(6)))
        adj_best, adj_report = best_edl(timeline, manifest, list(range(6)),
                                        weights=loaded)
        s_base = score_edl(base_best, timeline)
        s_adj = score_edl(adj_best, timeline, loaded)
        check("S4 scorer consumes adjusted weights end to end",
              adj_report["weights"] == loaded and
              base_report["weights"] == WEIGHTS and
              0.0 <= s_adj["total"] <= 1.0 and 0.0 <= s_base["total"] <= 1.0)

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
