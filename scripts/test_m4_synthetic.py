"""Offline synthetic test for Milestone 4 (no GPU, no network, no ffmpeg run).

Exercises the full QC + packaging layer on the same synthetic case the M3
test uses: probe plan (Peak-End weighting) -> aesthetic evaluation (injected
scores) -> technical defect mapping -> taste checklist -> thumbnails ->
disclosure / persona / metadata -> upload package.

Run: python scripts/test_m4_synthetic.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.edit.align import synthetic_alignment
from pipeline.edit.edl import build_edl
from pipeline.edit.timeline import build_timeline
from pipeline.packaging.publish import (build_disclosure, build_metadata,
                                        build_upload_package, persona_check)
from pipeline.packaging.thumbnails import (build_concepts, build_thumbnail_jobs,
                                           _hook_phrase)
from pipeline.qc.aesthetic import (ATTRIBUTE_MIN, END_FRACTION,
                                   build_probe_plan, evaluate_aesthetics)
from pipeline.qc.taste_pass import build_checklist, write_checklist
from pipeline.qc.technical import (map_defects_to_events,
                                   parse_black_intervals,
                                   parse_freeze_intervals)
from pipeline.voice_director import direct

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --- synthetic inputs (same case as the M3 test) -----------------------------

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

CASE = {
    "title": "The Six Missing Minutes of Marlin",
    "tags": ["marlin", "cold case"],
    "sources": [
        {"id": "src1", "citation": "Marlin County Court Records, 1948, Box 12"},
        {"id": "src2", "citation": "Library of Congress, Telephony Collection"},
    ],
}


def make_manifest() -> dict:
    def shot(sid, bid, kind, decision, dur, asset=None):
        return {"shot_id": sid, "beat_id": bid, "kind": kind,
                "query": f"query for {sid}", "duration_s": dur,
                "evidentiary": kind == "document", "decision": decision,
                "asset_id": asset, "score": 0.5, "license_class": "cc",
                "attribution": "Archive / CC-BY",
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
    ], "stats": {}, "rewrite_requests": [], "generation_jobs": []}


GROUNDING = {
    "claims": [
        {"claim_id": "b01-c01", "source_id": "src2", "verdict": "entailed"},
        {"claim_id": "b04-c01", "source_id": "src1", "verdict": "entailed"},
        {"claim_id": "b03-c01", "source_id": None, "verdict": "neutral"},
    ],
    "flagged": ["b03-c01"],
    "contradicted": [],
}

LICENSE_MANIFEST = {
    "attribution": [
        {"asset_id": "loc:call_logs", "title": "Switchboard call logs",
         "source_url": "https://loc.gov/x", "license": "PD", "attribution": ""},
        {"asset_id": "archive:empty_shelf", "title": "Records room",
         "source_url": "https://archive.org/y", "license": "CC_BY",
         "attribution": "photo by J. Doe"},
    ],
    "needs_review": [],
}


def main() -> None:
    print("== M4 synthetic QC + packaging test ==\n")

    direction = direct(SCRIPT, seed=7)
    b05_lines = [l for l in direction if l["beat_id"] == "b05"]
    b05_lines[-1]["pause_after_s"] = 1.3
    alignment = synthetic_alignment(direction)
    timeline = build_timeline(SCRIPT, alignment)
    manifest = make_manifest()
    edl = build_edl(timeline, manifest, seed=3)
    events = edl["events"]

    # --- P: probe plan (Peak-End weighting) ----------------------------------
    print("P. probe plan")
    plan = build_probe_plan(edl, timeline)
    check("P1 one probe per event",
          len(plan["probes"]) == len(events) and
          {p["event_id"] for p in plan["probes"]} == {e["event_id"] for e in events})
    check("P2 held reveal probed as peak",
          any(p["segment"] == "peak" for p in plan["probes"]))
    end_start = plan["duration_s"] * (1 - END_FRACTION)
    check("P3 final 10% probed as end",
          all(p["t"] >= end_start for p in plan["probes"] if p["segment"] == "end") and
          any(p["segment"] == "end" for p in plan["probes"]))
    check("P4 probes inside their events",
          all(any(e["event_id"] == p["event_id"] and
                  e["video_in"] <= p["t"] <= e["video_out"] for e in events)
              for p in plan["probes"]))

    # --- A: aesthetic evaluation (injected scores) ----------------------------
    print("\nA. aesthetic gate")
    def scored(overall_by_segment, bad_event=None):
        out = []
        for p in plan["probes"]:
            scores = {"overall": overall_by_segment[p["segment"]],
                      "sharpness": 0.8, "composition": 0.7}
            if bad_event and p["event_id"] == bad_event:
                scores["sharpness"] = ATTRIBUTE_MIN - 0.1
            out.append({**p, "scores": scores})
        return out

    good = evaluate_aesthetics(plan, scored({"peak": 0.9, "end": 0.8, "body": 0.5}))
    check("A1 strong peak+end passes despite mediocre body",
          good["passed"] and good["aggregate"] > 0.6)

    weak_end = evaluate_aesthetics(plan, scored({"peak": 0.9, "end": 0.1, "body": 0.5}))
    check("A2 weak ending sinks the aggregate (Peak-End rule)",
          weak_end["aggregate"] < good["aggregate"] - 0.2)

    bad_id = plan["probes"][2]["event_id"]
    attr_fail = evaluate_aesthetics(plan, scored(
        {"peak": 0.9, "end": 0.8, "body": 0.5}, bad_event=bad_id))
    check("A3 attribute failure maps to EDL event + timestamp + attribute",
          not attr_fail["passed"] and len(attr_fail["failures"]) == 1 and
          attr_fail["failures"][0]["event_id"] == bad_id and
          attr_fail["failures"][0]["attribute"] == "sharpness" and
          "t" in attr_fail["failures"][0])

    # --- T: technical defect mapping ------------------------------------------
    print("\nT. technical gate")
    stderr = ("[blackdetect] black_start:2.0 black_end:3.0 black_duration:1.0\n"
              "[blackdetect] black_start:5.0 black_end:5.1 black_duration:0.1\n"
              "[freezedetect] lavfi.freezedetect.freeze_start: 10.0\n"
              "[freezedetect] lavfi.freezedetect.freeze_end: 14.5\n")
    blacks = parse_black_intervals(stderr)
    freezes = parse_freeze_intervals(stderr)
    check("T1 short black dips ignored, real blacks kept",
          len(blacks) == 1 and blacks[0]["start"] == 2.0)
    check("T2 freeze intervals parsed", len(freezes) == 1)

    mapped = map_defects_to_events(blacks + freezes, edl)
    black_hits = [f for f in mapped if f["attribute"] == "black"]
    check("T3 black maps to overlapping event ids",
          black_hits and all(
              any(e["event_id"] == f["event_id"] and
                  e["video_in"] < 3.0 and e["video_out"] > 2.0 for e in events)
              for f in black_hits))
    freeze_hits = [f for f in mapped if f["attribute"] == "freeze"]
    check("T4 freezes only fail FOOTAGE events (stills hold by design)",
          all(next(e for e in events if e["event_id"] == f["event_id"])["kind"]
              == "footage" for f in freeze_hits))

    # --- C: taste checklist ----------------------------------------------------
    print("\nC. taste checklist")
    checklist = build_checklist(timeline, edl, GROUNDING, [attr_fail])
    check("C1 midpoint + reveal + gaps + generated located",
          checklist["midpoint_t"] is not None and
          checklist["reveal_timestamps"] and
          checklist["coverage_gaps"] and checklist["generated_events"])
    check("C2 flagged claims and QC failures carried over",
          checklist["flagged_claims"] == ["b03-c01"] and
          checklist["qc_failures"] == attr_fail["failures"])
    with tempfile.TemporaryDirectory() as td:
        md = write_checklist(checklist, Path(td) / "taste.md").read_text()
        check("C3 markdown renders every section",
              "DRAMATIZATION" in md and "Midpoint reversal" in md and
              "b03-c01" in md and "sharpness" in md)

    # --- H: thumbnails -----------------------------------------------------------
    print("\nH. thumbnails")
    concepts = build_concepts(SCRIPT, manifest)
    check("H1 exactly 3 distinct concepts",
          [c["concept_id"] for c in concepts] == ["document", "location", "typographic"]
          and len({c["prompt"] for c in concepts}) == 3)
    check("H2 document concept uses the evidentiary shot query",
          "query for b02-s2" in concepts[0]["prompt"])
    check("H3 typographic hook is short and punchy",
          1 <= len(concepts[2]["hook"].split()) <= 5 and
          concepts[2]["hook"].isupper())
    check("H4 hook phrase strips stopwords",
          "THE" not in _hook_phrase("The man in the dark hat vanished").split())
    with tempfile.TemporaryDirectory() as td:
        jobs = build_thumbnail_jobs(concepts, Path(td) / "thumbs")
        jobs2 = build_thumbnail_jobs(concepts, Path(td) / "thumbs")
        check("H5 jobs deterministic, 1280x720, specs mirrored to disk",
              [j["seed"] for j in jobs] == [j["seed"] for j in jobs2] and
              all(j["width"] == 1280 and j["height"] == 720 for j in jobs) and
              json.loads((Path(td) / "thumbs/thumbnail_jobs.json").read_text()))

    # --- D: disclosure + persona -------------------------------------------------
    print("\nD. disclosure + persona")
    disclosure = build_disclosure(edl)
    check("D1 generated shot forces altered-content disclosure",
          disclosure["altered_content"] and disclosure["generated_events"] and
          disclosure["statement"])
    clean_edl = {**edl, "events": [e for e in events if e["source"] != "generated"]}
    check("D2 all-archival video needs no disclosure",
          not build_disclosure(clean_edl)["altered_content"])
    check("D3 persona rule passes clean narration", persona_check(SCRIPT) == [])
    advice = {"beats": [{"id": "bX", "role": "establish",
                         "text": "You should consult your doctor about this."}]}
    v = persona_check(advice)
    check("D4 advice-giving narration is flagged with beat id",
          len(v) == 2 and all(x["beat_id"] == "bX" for x in v))

    # --- M: metadata + upload package ---------------------------------------------
    print("\nM. metadata + upload package")
    metadata = build_metadata(CASE, SCRIPT, timeline, GROUNDING,
                              LICENSE_MANIFEST, disclosure)
    desc = metadata["description"]
    check("M1 title from case, within limit",
          metadata["title"] == CASE["title"] and len(metadata["title"]) <= 100)
    check("M2 chapters start at 00:00 and cover every beat",
          "00:00 Cold Open" in desc and desc.count("\n0") >= len(timeline["beats"]) - 1)
    check("M3 only entailed sources cited",
          "Marlin County Court Records" in desc and
          "Library of Congress" in desc and desc.count("- ") >= 2)
    check("M4 CC-BY attribution text included",
          "photo by J. Doe" in desc and metadata["cc_by_count"] == 1)
    check("M5 disclosure statement in description",
          disclosure["statement"] in desc)
    check("M6 tags harvested + within budget",
          "documentary" in metadata["tags"] and "Marlin" in metadata["tags"] and
          sum(len(t) + 1 for t in metadata["tags"]) <= 481)

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        jobs = build_thumbnail_jobs(concepts, run_dir / "thumbnails")
        pkg_path = build_upload_package(
            run_dir, run_dir / "assembled.mp4", metadata, disclosure, jobs,
            {"passed": False, "blockers": ["no assembled video"]})
        pkg = json.loads(pkg_path.read_text())
        check("M7 package written even when blocked, with blockers listed",
              pkg["ready_to_upload"] is False and pkg["blockers"] and
              pkg["youtube"]["status"]["privacyStatus"] == "private")
        check("M8 synthetic-media flag mirrors disclosure",
              pkg["youtube"]["status"]["containsSyntheticMedia"] is True)
        check("M9 3 thumbnails in package",
              len(pkg["thumbnails"]) == 3 and
              pkg["thumbnails"][2]["hook_text"])

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
