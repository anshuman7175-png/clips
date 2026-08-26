"""EDL builder (PLAN.md Layer 6) - the edit grammar, enforced.

Input contract: the M2 shotlist manifest is the ONLY thing this module may
cut from (match_gate module doc). If the manifest still contains
"rewrite_line" decisions, the script was never fixed - editing it would put
unproven claims on screen, so we fail loudly instead.

Grammar implemented here:
- J-cuts by default at beat (scene) transitions: next beat's audio leads its
  picture by 0.5-1.5s -> the incoming event's video_in starts BEFORE the beat's
  audio, i.e. the previous picture hands over while the new narration has
  already begun. Encoded as audio_lead_s on the incoming event.
- L-cuts when a document close-up should linger: if the outgoing event is a
  document shot, its picture extends 0.5-1.0s under the next beat's narration
  instead of J-cutting.
- Pause-driven rhythm: editorial pauses (timeline.pause_beats) mark the
  covering event hold=True + ambience_swell=True. No cut lands inside them.
- Emotion-modulated duration: tension compresses shots, gravity and reveals
  stretch them. Reveal sentences get a HELD shot to the end of their pause.
- Word-level precision: every internal cut snaps to a word boundary.
- Match-cut opportunism: with a similarity function, shots inside a beat are
  ordered to maximize adjacent embedding similarity (hard cuts read as
  intentional match cuts).
- Jitter: all magic numbers are sampled per-cut from a seeded RNG so no two
  videos - and no two candidate EDLs - share a cutting template.
"""

from __future__ import annotations

import itertools
import random

from .timeline import snap

J_CUT_LEAD_RANGE = (0.5, 1.5)   # next audio leads picture (PLAN.md Layer 6)
L_CUT_EXTEND_RANGE = (0.5, 1.0)  # document picture lingers under narration
MIN_EVENT_S = 1.6                # never flash-cut below this


class EditContractError(RuntimeError):
    """The shotlist violates the edit engine's input contract."""


def build_edl(timeline: dict, shotlist: dict, seed: int,
              similarity=None) -> dict:
    _enforce_contract(shotlist)
    rng = random.Random(seed)
    boundaries = timeline["word_boundaries"]

    shots_by_beat: dict[str, list[dict]] = {}
    for shot in shotlist["shots"]:
        shots_by_beat.setdefault(shot["beat_id"], []).append(shot)

    events: list[dict] = []
    n = 0
    for bi, beat in enumerate(timeline["beats"]):
        shots = shots_by_beat.get(beat["beat_id"], [])
        if not shots:
            # Coverage gap: extend the previous event across this beat rather
            # than cutting to nothing; flagged for the human taste pass.
            if events:
                events[-1]["video_out"] = beat["end"]
                events[-1]["coverage_gap"] = True
            continue
        shots = _order_for_match_cuts(shots, similarity)
        spans = _allocate(beat, shots, rng, boundaries)

        for si, (shot, (t0, t1)) in enumerate(zip(shots, spans)):
            n += 1
            is_first_in_beat = si == 0
            event = {
                "event_id": f"e{n:03d}",
                "beat_id": beat["beat_id"],
                "shot_id": shot["shot_id"],
                "asset_id": shot["asset_id"] or f"generated:{shot['shot_id']}",
                "kind": shot["kind"],
                "source": "generated" if shot["decision"] == "generate"
                          else shot["asset_id"].split(":")[0],
                "video_in": t0,
                "video_out": t1,
                "cut_style": "hard",
                "audio_lead_s": 0.0,
                "hold": False,
                "ambience_swell": False,
                "coverage_gap": False,
                "label": "DRAMATIZATION" if shot["label_required"] else None,
                "attribution": shot.get("attribution"),
            }
            if is_first_in_beat and events and bi > 0:
                prev = events[-1]
                if prev["kind"] == "document":
                    # L-cut: the document lingers under the new narration.
                    extend = round(rng.uniform(*L_CUT_EXTEND_RANGE), 2)
                    prev["video_out"] = round(prev["video_out"] + extend, 3)
                    prev["cut_style"] = "l_cut"
                    event["video_in"] = prev["video_out"]
                else:
                    # J-cut (default): this beat's audio started at beat.start;
                    # its picture arrives audio_lead_s later.
                    lead = round(rng.uniform(*J_CUT_LEAD_RANGE), 2)
                    event["cut_style"] = "j_cut"
                    event["audio_lead_s"] = lead
                    event["video_in"] = round(beat["start"] + lead, 3)
                    prev["video_out"] = event["video_in"]
            events.append(event)

        _apply_pause_rhythm(events, beat)

    events = [e for e in events if e["video_out"] - e["video_in"] > 0.05]
    return {"seed": seed, "duration_s": timeline["duration_s"], "events": events}


def _enforce_contract(shotlist: dict) -> None:
    bad = [s["shot_id"] for s in shotlist["shots"] if s["decision"] == "rewrite_line"]
    if bad:
        raise EditContractError(
            f"shotlist contains unresolved rewrite_line decisions ({bad}); "
            "rewrite those script lines and re-run M2 before editing")
    for s in shotlist["shots"]:
        if s["decision"] == "generate" and not s["label_required"]:
            raise EditContractError(
                f"{s['shot_id']}: generated shot without dramatization label")


def _allocate(beat: dict, shots: list[dict], rng: random.Random,
              boundaries: list[float]) -> list[tuple[float, float]]:
    """Slice the beat span across its shots: requested durations as weights,
    emotion-modulated, cuts snapped to word boundaries, reveals held."""
    emo = beat["emotion"]
    # tension -> faster cuts; gravity -> longer, weightier shots.
    tempo = 1.0 - 0.30 * emo["tension"] + 0.20 * emo["gravity"]
    weights = [max(1.0, s["duration_s"]) * tempo * rng.uniform(0.85, 1.15)
               for s in shots]
    total_w = sum(weights)
    span = beat["end"] - beat["start"]

    spans, t = [], beat["start"]
    for i, w in enumerate(weights):
        if i == len(weights) - 1:
            t1 = beat["end"]
        else:
            t1 = snap(t + span * w / total_w, boundaries)
            t1 = max(t1, t + MIN_EVENT_S)
            t1 = min(t1, beat["end"])
        spans.append((round(t, 3), round(t1, 3)))
        t = t1
    return spans


def _apply_pause_rhythm(events: list[dict], beat: dict) -> None:
    """Editorial pauses + reveals: the covering event holds, ambience swells,
    and no cut may land inside the pause window."""
    beat_events = [e for e in events if e["beat_id"] == beat["beat_id"]]
    for sent in beat["sentences"]:
        if not (sent["editorial_pause"] or sent["is_reveal"]):
            continue
        p0, p1 = sent["end"], round(sent["end"] + sent["pause_after_s"], 3)
        for e in beat_events:
            if e["video_in"] < p1 and e["video_out"] > p0:  # overlaps the pause
                e["ambience_swell"] = True
                if sent["is_reveal"]:
                    e["hold"] = True
                    e["video_out"] = max(e["video_out"], p1)  # hold through it
                elif p0 < e["video_out"] < p1:
                    e["video_out"] = p1  # push the cut out of the pause
    # Re-chain: later events must start where the (possibly extended) previous ends.
    for a, b in zip(beat_events, beat_events[1:]):
        if b["video_in"] < a["video_out"]:
            b["video_in"] = a["video_out"]


def _order_for_match_cuts(shots: list[dict], similarity) -> list[dict]:
    """Order shots within a beat to maximize adjacent embedding similarity
    (match-cut opportunism). Beats plan 1-3 shots, so brute force is exact."""
    if similarity is None or len(shots) < 3:
        return shots
    def chain_score(order):
        return sum(similarity(a["asset_id"], b["asset_id"]) or 0.0
                   for a, b in zip(order, order[1:])
                   if a["asset_id"] and b["asset_id"])
    return list(max(itertools.permutations(shots), key=chain_score))
