"""Offline synthetic test for Layer 1 (research) - no network, no LLM.

Exercises the case scout (entity/timeline graph with the deterministic date
backstop), contradiction detection (injected judge, call budget), the novelty
gate (lexical coverage vs synthetic transcripts), and case-file assembly
against the exact schema Milestone 1 consumes.

Run: python scripts/test_research_synthetic.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.research.case_file import build_case_file, slug, write_case_file
from pipeline.research.case_scout import build_graph, parse_dates
from pipeline.research.contradiction import (MAX_PAIRS, find_contradictions,
                                             select_reversal)
from pipeline.research.novelty import (MIN_NOVEL_POINTS, extract_facts,
                                       fact_covered, novelty_gate)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --- synthetic primary sources (SS Poet-like case) ----------------------------

DOCS = [
    {"id": "doc:report", "api": "nara",
     "citation": "Coast Guard Marine Casualty Report (1981)",
     "url": "https://example.gov/report", "date": "1981-05-01",
     "text": ("The SS Poet departed Philadelphia on 24 October 1980 with a "
              "crew of 34. The vessel passed inspection on 13 October 1980. "
              "The Marine Board concluded the most probable cause was rapid "
              "flooding in heavy weather. No distress call was received.")},
    {"id": "doc:hearing", "api": "courtlistener",
     "citation": "House Subcommittee hearing record (1981)",
     "url": "https://example.gov/hearing", "date": "1981-07-15",
     "text": ("Testimony established the SS Poet was not reported overdue "
              "until 3 November 1980. The company assumed radio failure, "
              "which crew families disputed, noting the ship carried "
              "redundant communication equipment.")},
    {"id": "doc:paper", "api": "chronicling_america",
     "citation": "Philadelphia Inquirer, 1980-11-09 (Chronicling America)",
     "url": "https://example.gov/paper", "date": "1980-11-09",
     "text": ("The search for the missing grain ship began on November 8, "
              "1980 and covered three hundred thousand square miles of the "
              "Atlantic without finding wreckage.")},
]

# Fake extractor: deterministic, keyed off document text. It deliberately
# OMITS the 13 October inspection date so the regex backstop must catch it.
def fake_extractor(text: str) -> dict:
    if "departed Philadelphia" in text:
        return {"entities": [{"name": "SS Poet", "type": "vessel"},
                             {"name": "Marine Board", "type": "organization"}],
                "events": [{"date": "1980-10-24",
                            "description": "SS Poet departed Philadelphia with a crew of 34"}]}
    if "reported overdue" in text:
        return {"entities": [{"name": "SS Poet", "type": "vessel"},
                             {"name": "House Subcommittee", "type": "organization"}],
                "events": [{"date": "1980-11-03",
                            "description": "SS Poet reported overdue ten days after last contact"}]}
    return {"entities": [{"name": "Philadelphia Inquirer", "type": "organization"}],
            "events": [{"date": "1980-11-08",
                        "description": "Search began covering 300,000 square miles"}]}


def make_judge(log: list):
    def judge(a: dict, b: dict) -> dict:
        log.append((a["id"], b["id"]))
        pair = {a["id"], b["id"]}
        if pair == {"doc:report", "doc:hearing"}:
            return {"contradicts": True, "severity": 0.9,
                    "point": ("The casualty report attributes the silence to rapid "
                              "flooding, while hearing testimony shows the ship "
                              "carried redundant communication equipment.")}
        if pair == {"doc:report", "doc:paper"}:
            return {"contradicts": True, "severity": 0.4,
                    "point": "Search start dates differ between report and newspaper."}
        return {"contradicts": False, "point": "", "severity": 0.0}
    return judge


def main() -> None:
    print("== Layer 1 synthetic research test ==\n")

    # --- D: deterministic date parsing --------------------------------------
    print("D. date parsing")
    hits = parse_dates(DOCS[0]["text"])
    check("D1 'DD Month YYYY' parsed to ISO",
          any(h["date"] == "1980-10-24" for h in hits))
    check("D2 context is the containing sentence",
          any("passed inspection" in h["context"] for h in hits
              if h["date"] == "1980-10-13"))
    check("D3 'Month DD, YYYY' parsed",
          any(h["date"] == "1980-11-08" for h in parse_dates(DOCS[2]["text"])))

    # --- G: case scout graph --------------------------------------------------
    print("\nG. entity/timeline graph")
    graph = build_graph(DOCS, fake_extractor)
    poet = next(e for e in graph["entities"] if e["name"] == "SS Poet")
    check("G1 entities merged across documents",
          poet["doc_ids"] == ["doc:report", "doc:hearing"])
    check("G2 entities ranked by document spread",
          graph["entities"][0]["name"] == "SS Poet")
    dated = [e["date"] for e in graph["timeline"] if e["date"]]
    check("G3 timeline sorted chronologically", dated == sorted(dated))
    check("G4 date backstop catches events the extractor missed",
          any(e["date"] == "1980-10-13" for e in graph["timeline"]))
    check("G5 edges link entities to co-mentioned events",
          any(ed["entity"] == "SS Poet" and
              "SS Poet" in graph["timeline"][ed["event_index"]]["description"]
              for ed in graph["edges"]))

    # --- C: contradiction detection --------------------------------------------
    print("\nC. contradiction detection")
    calls: list = []
    contradictions = find_contradictions(DOCS, make_judge(calls))
    check("C1 only contradicting pairs kept, sharpest first",
          len(contradictions) == 2 and contradictions[0]["severity"] == 0.9)
    check("C2 pairwise budget respected",
          len(calls) == min(3, MAX_PAIRS))
    calls2: list = []
    find_contradictions(DOCS * 2, make_judge(calls2), max_pairs=4)
    check("C3 budget truncates large doc sets", len(calls2) == 4)
    reversal = select_reversal(contradictions)
    check("C4 reversal is the sharpest contradiction",
          reversal is not None and "redundant communication" in reversal["point"])
    check("C5 no contradictions -> no reversal", select_reversal([]) is None)

    # --- N: novelty gate ----------------------------------------------------------
    print("\nN. novelty gate")
    facts = extract_facts(graph, contradictions)
    check("N1 facts = dated events + contradiction points, deduped",
          len([f for f in facts if f["kind"] == "timeline"]) == len(dated) and
          len([f for f in facts if f["kind"] == "contradiction"]) == 2)

    covering = (" ".join(f["fact"] for f in facts))  # a video that told everything
    verdict_all = novelty_gate(facts, [covering])
    check("N2 full coverage fails the gate",
          not verdict_all["passed"] and len(verdict_all["novel_points"]) == 0)

    partial = ("the poet departed philadelphia october 1980 crew "
               "vessel passed inspection october 1980")
    verdict_partial = novelty_gate(facts, [partial])
    check("N3 partial coverage passes when >= 3 points stay novel",
          verdict_partial["passed"] and
          len(verdict_partial["novel_points"]) >= MIN_NOVEL_POINTS and
          len(verdict_partial["covered_points"]) >= 1)

    check("N4 untouched topic (no transcripts) passes on substance",
          novelty_gate(facts, [])["passed"])
    check("N5 coverage heuristic is order-insensitive",
          fact_covered("crew of 34 departed Philadelphia",
                       "departed from philadelphia was a crew of 34 sailors"))

    # --- F: case file -------------------------------------------------------------
    print("\nF. case file")
    case = build_case_file("The Vanishing of the SS Poet", DOCS, reversal,
                           graph, verdict_partial)
    check("F1 M1 schema: title + contradiction + sources[{id,citation,text}]",
          case["title"] and case["contradiction"] == reversal["point"] and
          all(s["id"] == f"src{i}" for i, s in enumerate(case["sources"], 1)) and
          all(s["citation"] and s["text"] for s in case["sources"]))
    check("F2 contradiction sources remapped to src ids",
          set(case["contradiction_sources"]) <= {s["id"] for s in case["sources"]})
    check("F3 tags harvested from top entities", "ss poet" in case["tags"])
    check("F4 novelty verdict embedded",
          case["novelty"]["passed"] and case["novelty"]["novel_points"])
    check("F5 slug is filesystem-safe",
          slug("The Vanishing of the SS Poet") == "the-vanishing-of-the-ss-poet")
    with tempfile.TemporaryDirectory() as td:
        path = write_case_file(case, Path(td))
        loaded = json.loads(path.read_text(encoding="utf-8"))
        check("F6 case file round-trips to cases/<slug>/case.json",
              path.name == "case.json" and
              path.parent.name == "the-vanishing-of-the-ss-poet" and
              loaded == case)

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
