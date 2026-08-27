"""YouTube Analytics retention fetch (PLAN.md Layer 10).

Pulls the audience-retention report for one video:
    dimensions = elapsedVideoTimeRatio  (100 buckets, 0.00 .. 0.99)
    metrics    = audienceWatchRatio, relativeRetentionPerformance

- fetch_retention(): live API call (lazy googleapiclient import, same OAuth
  pattern as packaging.publish.upload_youtube - one browser flow, no secrets
  stored in the repo).
- normalize_curve(): API rows -> the canonical curve shape used everywhere
  downstream: [{"x": ratio, "watch": float, "relative": float|None}, ...].
- synthetic_curve(): deterministic offline curve with injected dips/spikes so
  the whole post-mortem layer is testable with no network and no channel.
"""

from __future__ import annotations

import math
from pathlib import Path

CURVE_POINTS = 100  # the API's elapsedVideoTimeRatio resolution


def fetch_retention(video_id: str, client_secrets: Path) -> list[dict]:
    """Live audience-retention rows for `video_id` (needs OAuth browser flow)."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as err:
        raise RuntimeError(
            "analytics fetch needs: pip install google-api-python-client "
            "google-auth-oauthlib") from err
    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_secrets),
        ["https://www.googleapis.com/auth/yt-analytics.readonly"])
    yta = build("youtubeAnalytics", "v2", credentials=flow.run_local_server(port=0))
    response = yta.reports().query(
        ids="channel==MINE",
        startDate="2000-01-01",   # lifetime; retention is per-video anyway
        endDate="2100-01-01",
        metrics="audienceWatchRatio,relativeRetentionPerformance",
        dimensions="elapsedVideoTimeRatio",
        filters=f"video=={video_id};audienceType==ORGANIC",
    ).execute()
    return response.get("rows", [])


def normalize_curve(rows: list) -> list[dict]:
    """API rows (or pre-saved JSON rows) -> canonical curve, sorted by x.

    Accepts either raw API rows ([x, watch, relative]) or already-shaped
    dicts, so a curve saved to disk round-trips unchanged.
    """
    curve = []
    for row in rows:
        if isinstance(row, dict):
            curve.append({"x": float(row["x"]), "watch": float(row["watch"]),
                          "relative": row.get("relative")})
        else:
            x, watch = float(row[0]), float(row[1])
            relative = float(row[2]) if len(row) > 2 and row[2] is not None else None
            curve.append({"x": x, "watch": watch, "relative": relative})
    curve.sort(key=lambda p: p["x"])
    if not curve:
        raise ValueError("empty retention curve - video may be too new "
                         "(retention needs ~1 week of data)")
    return curve


def synthetic_curve(dips: list[tuple[float, float, float]] = (),
                    spikes: list[tuple[float, float]] = (),
                    start: float = 0.95, end: float = 0.40,
                    n: int = CURVE_POINTS) -> list[dict]:
    """Deterministic retention curve for offline tests.

    dips   : (center_ratio, depth, width) - viewers PERMANENTLY lost around
             center (real abandonment is a step down, not a notch).
    spikes : (center_ratio, height) - temporary rewatch bumps.
    """
    curve = []
    for i in range(n):
        x = i / n
        watch = start + (end - start) * x  # gentle baseline decay
        for center, depth, width in dips:
            # smooth step: 0 before the dip window, `depth` lost after it
            t = (x - (center - width / 2)) / max(width, 1e-6)
            watch -= depth * min(1.0, max(0.0, t))
        for center, height in spikes:
            watch += height * math.exp(-((x - center) ** 2) / (2 * 0.01 ** 2))
        curve.append({"x": round(x, 2), "watch": round(max(0.0, watch), 5),
                      "relative": None})
    return curve
