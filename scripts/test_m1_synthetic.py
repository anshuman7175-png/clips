"""Offline synthetic test for the Milestone 1 script/voice/master layer.

No network, no GPU, no ffmpeg: LLM calls are stubbed at the chat_json boundary
and ffmpeg at the _run boundary, so every M1 contract is exercised end to end:

  V   structure validators (PLAN.md Layer 2) fail loudly on every violation
  G   grounding manifest (Layer 5): claim ids, verdict routing, batching
  D   voice director (Layer 3): determinism, jitter, bounds, reveal handling
  M   mastering (Layer 7): two-pass loudnorm flow + hard QC gate
  C   checkpoint runner: stages skip on re-run, force re-executes

Run:  python scripts/test_m1_synthetic.py
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline.mastering as mastering
import pipeline.script_engine.grounding as grounding
from pipeline.checkpoint import Run
from pipeline.script_engine.structure import ValidationError, validate_script
from pipeline.voice_director import direct, split_sentences

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def expect_raise(name: str, fn, rule: str) -> None:
    try:
        fn()
    except ValidationError as e:
        check(name, e.rule == rule, f"raised rule={e.rule}, expected {rule}")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception type: {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


CFG = SimpleNamespace(
    words_per_minute=150,
    rehook_window_s=(60.0, 90.0),
    max_loop_gap_s=180.0,
    target_lufs=-14.0,
    target_true_peak_db=-1.0,
    lufs_tolerance=1.0,
)


def sentences(n_words: int, n_sentences: int, prefix: str) -> str:
    """Deterministic filler text with an exact word count, split into sentences."""
    per = n_words // n_sentences
    out = []
    w = 0
    for s in range(n_sentences):
        count = per if s < n_sentences - 1 else n_words - w
        out.append(" ".join(f"{prefix}{s}w{i}" for i in range(count)) + ".")
        w += count
    return " ".join(out)


def make_valid_script() -> dict:
    """A script that satisfies every validator at 150 wpm.

    Timeline (2.5 words/s): b01 0-24s, b02 24-48s, b03 48-64s (destabilize,
    overlaps the 60-90s re-hook window), acts split 25% / 58% / 17%.
    """
    emo = {"tension": 0.5, "wonder": 0.3, "gravity": 0.5, "pace": 0.5}
    hot = {"tension": 0.9, "wonder": 0.2, "gravity": 0.8, "pace": 0.3}

    def beat(bid, act, role, words, nsent, loop=None, emotion=emo):
        return {"id": bid, "act": act, "role": role,
                "text": sentences(words, nsent, bid), "loop": loop, "emotion": emotion}

    return {
        "title": "The Vanished Ledger",
        "beats": [
            beat("b01", 1, "cold_open", 60, 3, {"action": "plant", "loop_id": "L1"}),
            beat("b02", 1, "establish", 60, 3),
            beat("b03", 1, "destabilize", 40, 2, emotion=hot),
            beat("b04", 2, "investigate", 100, 4, {"action": "plant", "loop_id": "L2"}),
            beat("b05", 2, "investigate", 100, 4, {"action": "payoff", "loop_id": "L1"}),
            beat("b06", 2, "midpoint_reversal", 80, 3, {"action": "plant", "loop_id": "L3"}, emotion=hot),
            beat("b07", 2, "investigate", 100, 4, {"action": "payoff", "loop_id": "L2"}),
            beat("b08", 3, "synthesis", 60, 3, {"action": "payoff", "loop_id": "L3"}),
            beat("b09", 3, "honest_ending", 50, 2),
        ],
    }


# ---------------------------------------------------------------- V. validators
def test_validators() -> None:
    print("\nV. structure validators")
    base = make_valid_script()

    report = validate_script(base, CFG)
    check("V1 valid script passes all gates",
          any("cold_open: OK" in r for r in report)
          and any("honest_ending: OK" in r for r in report)
          and any(r.startswith("runtime_estimate") for r in report))

    s = copy.deepcopy(base)
    s["beats"][0]["text"] = "Welcome back everyone. " + s["beats"][0]["text"]
    expect_raise("V2 greeting opener rejected", lambda: validate_script(s, CFG), "cold_open")

    s = copy.deepcopy(base)
    s["beats"][0]["role"] = "establish"
    expect_raise("V3 first beat must be cold_open", lambda: validate_script(s, CFG), "cold_open")

    s = copy.deepcopy(base)
    s["beats"][4]["text"] += " Little did they know what came next."
    expect_raise("V4 banned phrase rejected", lambda: validate_script(s, CFG), "ban_list")

    s = copy.deepcopy(base)
    s["beats"][5]["role"] = "investigate"
    expect_raise("V5 missing midpoint reversal rejected", lambda: validate_script(s, CFG), "midpoint_reversal")

    s = copy.deepcopy(base)
    s["beats"][6]["role"] = "midpoint_reversal"
    expect_raise("V6 duplicate reversal rejected", lambda: validate_script(s, CFG), "midpoint_reversal")

    s = copy.deepcopy(base)
    s["beats"][7]["loop"] = None  # L3 never paid off
    expect_raise("V7 unresolved loop rejected", lambda: validate_script(s, CFG), "open_loops")

    s = copy.deepcopy(base)
    # Kill every re-hook trigger inside the 60-90s window: b03 (48-64s) loses
    # its destabilize role, b04 (64-104s) loses its loop plant (and its payoff
    # in b07 so the loop books stay balanced).
    s["beats"][2]["role"] = "establish"
    s["beats"][3]["loop"] = None
    s["beats"][6]["loop"] = None
    expect_raise("V8 missing re-hook rejected", lambda: validate_script(s, CFG), "rehook")

    s = copy.deepcopy(base)
    # Bloat act 2 far past 65% of runtime.
    s["beats"][4]["text"] = sentences(900, 20, "pad")
    expect_raise("V9 act proportions enforced", lambda: validate_script(s, CFG), "act_proportions")

    s = copy.deepcopy(base)
    s["beats"][-1]["role"] = "synthesis"
    expect_raise("V10 honest ending required", lambda: validate_script(s, CFG), "honest_ending")

    s = copy.deepcopy(base)
    # Stretch the gap between loop events past max_loop_gap_s + 60s (240s hard
    # limit) while keeping act proportions legal: only loop events are b01's
    # plant at t=0 and b08's payoff, pushed to t=256s by padding act 2 and 3.
    for b in s["beats"]:
        b["loop"] = None
    s["beats"][0]["loop"] = {"action": "plant", "loop_id": "L1"}
    s["beats"][4]["text"] = sentences(200, 6, "gap")   # act 2 -> 192s (61.5%)
    s["beats"][7]["text"] = sentences(90, 4, "pad3")   # act 3 -> 56s (17.9%)
    s["beats"][7]["loop"] = {"action": "payoff", "loop_id": "L1"}
    expect_raise("V11 loop gap past 4 min rejected", lambda: validate_script(s, CFG), "open_loops")


# ---------------------------------------------------------------- G. grounding
def test_grounding() -> None:
    print("\nG. grounding manifest")
    script = {"beats": [
        {"id": "b01", "text": "The ledger vanished in March 1954. Two clerks reported it."},
        {"id": "b02", "text": "The auditor denied any loss."},
    ]}
    sources = [{"id": "src1", "citation": "NARA RG-60", "text": "In March 1954 the ledger disappeared."}]

    decompose_calls: list[dict] = []

    def fake_decompose(endpoint, system, user, **kw):
        payload = json.loads(user)
        decompose_calls.append(payload)
        return {"claims": [
            {"text": s.strip()} for s in payload["text"].split(".") if s.strip()
        ]}

    real = grounding.chat_json
    grounding.chat_json = fake_decompose
    try:
        cfg = SimpleNamespace(local=None)
        claims = grounding.decompose_claims(cfg, script)
    finally:
        grounding.chat_json = real

    check("G1 one decompose call per beat", len(decompose_calls) == 2)
    check("G2 claim ids are beat-scoped and sequential",
          [c["claim_id"] for c in claims] == ["b01-c01", "b01-c02", "b02-c01"]
          and all(c["beat_id"] in ("b01", "b02") for c in claims))

    # Grounding: 12 claims forces two batches of <=10; verdicts route correctly.
    many = [{"claim_id": f"b01-c{i:02d}", "beat_id": "b01", "text": f"claim {i}"} for i in range(1, 13)]
    ground_calls: list[str] = []

    def fake_ground(endpoint, system, user, **kw):
        ground_calls.append(user)
        batch = json.loads(user.split("CLAIMS:\n", 1)[1])
        results = []
        for c in batch:
            n = int(c["claim_id"].split("-c")[1])
            if n == 1:
                results.append({"claim_id": c["claim_id"], "source_id": "src1",
                                "quote": "the ledger disappeared", "verdict": "entailed"})
            elif n == 2:
                results.append({"claim_id": c["claim_id"], "source_id": "src1",
                                "quote": None, "verdict": "contradicted"})
            elif n == 3:
                results.append({"claim_id": c["claim_id"], "source_id": None,
                                "quote": None, "verdict": "entailed"})
            else:
                results.append({"claim_id": c["claim_id"], "source_id": "src1",
                                "quote": None, "verdict": "neutral"})
        return {"results": results}

    grounding.chat_json = fake_ground
    try:
        manifest = grounding.ground_claims(SimpleNamespace(local=None), many, sources)
    finally:
        grounding.chat_json = real

    check("G3 batching keeps local context small", len(ground_calls) == 2)
    check("G4 sources block included in every call",
          all("[src1] (NARA RG-60)" in u for u in ground_calls))
    check("G5 contradicted claims hard-listed", manifest["contradicted"] == ["b01-c02"])
    check("G6 empty retrieval auto-flagged even when 'entailed'", "b01-c03" in manifest["flagged"])
    check("G7 neutral claims flagged, entailed+cited clean",
          "b01-c04" in manifest["flagged"] and "b01-c01" not in manifest["flagged"]
          and "b01-c01" not in manifest["contradicted"])
    check("G8 manifest carries every claim with its text",
          len(manifest["claims"]) == 12 and all("text" in c for c in manifest["claims"]))


# ---------------------------------------------------------------- D. voice director
def test_voice_director() -> None:
    print("\nD. voice director")
    script = make_valid_script()

    check("D1 sentence splitter handles . ! ?",
          split_sentences("One fact. Really! Why? Last one.")
          == ["One fact.", "Really!", "Why?", "Last one."])

    a = direct(script, seed=7)
    b = direct(script, seed=7)
    c = direct(script, seed=8)
    check("D2 same seed -> identical direction", a == b)
    check("D3 different seed -> different jitter", a != c)

    n_sent = sum(len(split_sentences(bt["text"])) for bt in script["beats"])
    check("D4 one line per sentence, beat-scoped ids",
          len(a) == n_sent and all(l["line_id"].startswith(l["beat_id"] + "-s") for l in a))

    check("D5 params inside engine bounds",
          all(0.0 < l["exaggeration"] <= 1.0 and 0.30 <= l["cfg_weight"] <= 0.70
              and l["pause_after_s"] > 0 for l in a))

    check("D6 no two consecutive sentences render identically",
          all(not (a[i]["exaggeration"] == a[i + 1]["exaggeration"]
                   and a[i]["cfg_weight"] == a[i + 1]["cfg_weight"]
                   and a[i]["pause_after_s"] == a[i + 1]["pause_after_s"])
              for i in range(len(a) - 1)))

    # Reveal handling: last sentence of the midpoint reversal slows and holds.
    rev = [l for l in a if l["beat_id"] == "b06"]
    others = [l for l in a if l["beat_id"] == "b06"][:-1]
    check("D7 reveal sentence gets the longest pause in its beat",
          rev[-1]["pause_after_s"] >= max(l["pause_after_s"] for l in others))
    check("D8 reveal pause is a held beat (>=1.1s)", rev[-1]["pause_after_s"] >= 1.1)


# ---------------------------------------------------------------- M. mastering
def test_mastering() -> None:
    print("\nM. mastering QC gate")

    def loudnorm_stderr(i: float, tp: float) -> str:
        return json.dumps({
            "input_i": f"{i:.2f}", "input_tp": f"{tp:.2f}",
            "input_lra": "6.00", "input_thresh": f"{i - 10:.2f}",
            "output_i": f"{i:.2f}", "output_tp": f"{tp:.2f}",
        })

    real_run = mastering._run

    def make_fake(final_i: float, final_tp: float, calls: list):
        state = {"n": 0}

        def fake(cmd: list[str]):
            calls.append(cmd)
            state["n"] += 1
            if state["n"] == 1:      # pass 1: measure raw mix
                return loudnorm_stderr(-23.0, -6.0)
            if state["n"] == 2:      # pass 2: render (no JSON needed)
                return ""
            return loudnorm_stderr(final_i, final_tp)  # QC re-measure
        return fake

    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "mix.wav", Path(td) / "master.wav"
        calls: list = []
        mastering._run = make_fake(-14.2, -1.4, calls)
        try:
            result = mastering.master(src, out, CFG)
        finally:
            mastering._run = real_run
        check("M1 in-spec master passes QC",
              result["qc_passed"] and result["integrated_lufs"] == -14.2)
        norm = " ".join(calls[1])
        check("M2 pass 2 is linear loudnorm seeded with pass-1 measurements",
              "linear=true" in norm and "measured_I=-23.00" in norm and "measured_TP=-6.00" in norm)
        check("M3 target spec is -14 LUFS / -1 dBTP at 48kHz",
              "I=-14.0" in norm and "TP=-1.0" in norm and "48000" in calls[1])
        check("M4 three ffmpeg invocations (measure, render, verify)", len(calls) == 3)

        # Out-of-spec master must hard-fail the run.
        mastering._run = make_fake(-11.0, -1.4, [])
        try:
            mastering.master(src, out, CFG)
            check("M5 loud master (-11 LUFS) fails QC", False, "did not raise")
        except RuntimeError as e:
            check("M5 loud master (-11 LUFS) fails QC", "MASTERING QC FAILED" in str(e))
        finally:
            mastering._run = real_run

        mastering._run = make_fake(-14.0, 0.5, [])
        try:
            mastering.master(src, out, CFG)
            check("M6 true-peak overshoot fails QC", False, "did not raise")
        except RuntimeError as e:
            check("M6 true-peak overshoot fails QC", "MASTERING QC FAILED" in str(e))
        finally:
            mastering._run = real_run


# ---------------------------------------------------------------- C. checkpoints
def test_checkpoints() -> None:
    print("\nC. checkpointed stage runner")
    with tempfile.TemporaryDirectory() as td:
        run = Run(Path(td), "video-001")
        counter = {"n": 0}

        def work():
            counter["n"] += 1
            return {"value": counter["n"]}

        first = run.stage("01_draft", work)
        second = run.stage("01_draft", work)  # must load checkpoint, not re-run
        check("C1 completed stage skipped on resume",
              counter["n"] == 1 and first == second == {"value": 1})

        forced = run.stage("01_draft", work, force=True)
        check("C2 force re-runs and overwrites", counter["n"] == 2 and forced == {"value": 2})

        run2 = Run(Path(td), "video-001")  # fresh process, same workdir
        resumed = run2.stage("01_draft", work)
        check("C3 checkpoint survives process restart",
              counter["n"] == 2 and resumed == {"value": 2})
        check("C4 checkpoint is valid JSON on disk",
              json.loads(run2.stage_path("01_draft").read_text())["value"] == 2)


def main() -> None:
    print("== Milestone 1 synthetic test (script -> voice -> master) ==")
    test_validators()
    test_grounding()
    test_voice_director()
    test_mastering()
    test_checkpoints()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
