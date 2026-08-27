"""Layer 1 orchestrator: harvest -> case scout -> contradictions -> novelty -> case file.

Usage:
    python -m pipeline.run_research --query "SS Poet disappearance 1980" \
        --title "The Disappearance of the SS Poet (1980)" \
        [--transcripts transcripts/ss-poet] [--run-id research-poet]

Stage split (PLAN.md compute budget) - everything here is CPU + network:
  r_01_harvest        : primary-source APIs (network only)
  r_02_graph          : entity/timeline extraction (LOCAL_* LLM, per document)
  r_03_contradictions : pairwise disagreement judging (FRONTIER_*, <= 6 calls)
  r_04_novelty        : facts vs existing-coverage transcripts (pure CPU)
  r_05_case           : cases/<slug>/case.json - written ONLY if the gate passes

--transcripts points at a directory of .txt transcripts of existing top videos
on the topic (harvest them manually or with yt-dlp --write-auto-sub). With no
directory the topic is treated as uncovered and the gate passes on substance.

Every stage checkpoints under runs/<run-id>/ exactly like the milestones.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint import Run
from .config import load_config
from .research import case_scout, contradiction, novelty, sources
from .research.case_file import build_case_file, write_case_file


def load_transcripts(transcripts_dir: Path | None) -> list[str]:
    if not transcripts_dir or not transcripts_dir.exists():
        return []
    return [p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(transcripts_dir.glob("*.txt"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="case search query")
    ap.add_argument("--title", required=True, help="working title for the case")
    ap.add_argument("--transcripts", type=Path, default=None,
                    help="dir of .txt transcripts of existing top videos")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--cases-dir", type=Path, default=Path("cases"))
    args = ap.parse_args()

    cfg = load_config()
    from .research.case_file import slug
    run = Run(cfg.workdir, args.run_id or f"research-{slug(args.title)}")

    # 1. Harvest (network) - per-source failures reported, never fatal.
    harvest = run.stage("r_01_harvest",
                        lambda: sources.harvest_all(args.query,
                                                    cfg.harvest_limit_per_source))
    print(f"  {len(harvest['docs'])} documents from "
          f"{len(sources.ALL_SOURCES) - len(harvest['errors'])} sources")
    for name, err in harvest["errors"].items():
        print(f"  [source down] {name}: {err}")
    if len(harvest["docs"]) < 2:
        raise SystemExit("fewer than 2 documents harvested - a case needs "
                         "sources that can disagree; refine the query")

    # 2. Case scout (local LLM) - entity/timeline graph.
    graph = run.stage("r_02_graph", lambda: case_scout.build_graph(
        harvest["docs"], case_scout.llm_extractor(cfg)))
    print(f"  {len(graph['entities'])} entities, "
          f"{len(graph['timeline'])} timeline events")

    # 3. Contradictions (frontier LLM, budgeted).
    contradictions = run.stage("r_03_contradictions",
                               lambda: contradiction.find_contradictions(
                                   harvest["docs"], contradiction.llm_judge(cfg)))
    reversal = contradiction.select_reversal(contradictions)
    if reversal:
        print(f"  sharpest contradiction ({reversal['severity']:.2f}): "
              f"{reversal['point'][:100]}")
    else:
        print("  NO documented contradiction found - weak reversal fuel")

    # 4. Novelty gate (pure CPU).
    transcripts = load_transcripts(args.transcripts)
    verdict = run.stage("r_04_novelty", lambda: novelty.novelty_gate(
        novelty.extract_facts(graph, contradictions), transcripts))
    print(f"  novelty: {len(verdict['novel_points'])} novel / "
          f"{len(verdict['covered_points'])} covered "
          f"(vs {verdict['existing_videos']} existing videos)")

    # 5. Case file - only a passing case reaches cases/.
    case = run.stage("r_05_case", lambda: build_case_file(
        args.title, harvest["docs"], reversal, graph, verdict))
    if not verdict["passed"]:
        raise SystemExit(f"NOVELTY GATE FAILED: {len(verdict['novel_points'])} "
                         f"novel points < {verdict['required']} required. "
                         "Case file kept in the run dir for inspection only.")
    path = write_case_file(case, args.cases_dir)
    print(f"\nDONE: {path} - feed it to Milestone 1:")
    print(f"  python -m pipeline.run_milestone1 --case {path}")


if __name__ == "__main__":
    main()
