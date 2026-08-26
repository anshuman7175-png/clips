"""Item-level license ledger (PLAN.md Layer 0 principle 6 + Layer 4 sources).

Every asset that can ever appear in an EDL passes through this ledger.
Source-level assumptions are NOT enough: each item carries its own license
record, classified at ingest, and the match gate refuses any asset whose
class is not explicitly usable.

License classes (ordered by legal safety):
  PD           - US public domain (LoC, NARA, pre-1929, US-gov works)
  CC0          - explicit CC0 dedication
  CC_BY        - usable WITH attribution (attribution manifest is exported)
  STOCK_FREE   - Pexels/Pixabay license (free commercial, no attribution req.)
  GENERATED    - produced by our own Wan 2.2 branch (dramatization label req.)
  NEEDS_REVIEW - item-level evidence insufficient -> human must resolve
  BANNED       - known-bad source or license (CC-NC, SA, British Pathe, ...)

The gate is a whitelist: anything not in USABLE_CLASSES cannot be matched.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

USABLE_CLASSES = {"PD", "CC0", "CC_BY", "STOCK_FREE", "GENERATED"}

# Hard blocklist (PLAN.md legal ledger). Checked against source URLs at ingest.
BANNED_URL_PATTERNS = re.compile(
    r"(britishpathe\.com|sound-effects\.bbcrewind|bbcsfx)", re.IGNORECASE
)

# licenseurl / rights-statement classification, most-specific first.
_LICENSE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"creativecommons\.org/publicdomain/zero", re.I), "CC0"),
    (re.compile(r"creativecommons\.org/publicdomain/mark", re.I), "PD"),
    (re.compile(r"creativecommons\.org/licenses/by-nc", re.I), "BANNED"),   # NC: never
    (re.compile(r"creativecommons\.org/licenses/by-sa", re.I), "NEEDS_REVIEW"),  # SA: viral
    (re.compile(r"creativecommons\.org/licenses/by-nd", re.I), "NEEDS_REVIEW"),  # ND: no edits
    (re.compile(r"creativecommons\.org/licenses/by/", re.I), "CC_BY"),
    (re.compile(r"rightsstatements\.org/vocab/NoC-US", re.I), "PD"),
    (re.compile(r"rightsstatements\.org/vocab/InC", re.I), "BANNED"),
    (re.compile(r"\bpublic domain\b", re.I), "PD"),
    (re.compile(r"no known (copyright|restrictions)", re.I), "PD"),
    (re.compile(r"pexels license", re.I), "STOCK_FREE"),
    (re.compile(r"pixabay (content )?license", re.I), "STOCK_FREE"),
]


def classify_license(raw: str | None, license_url: str | None, source: str) -> tuple[str, str]:
    """Return (license_class, reason). Whitelist logic: default is NEEDS_REVIEW."""
    blob = " ".join(filter(None, [raw, license_url]))
    if BANNED_URL_PATTERNS.search(blob):
        return "BANNED", "blocklisted source/license URL"
    for pattern, cls in _LICENSE_RULES:
        if pattern.search(blob):
            return cls, f"matched rule: {pattern.pattern}"
    # Source-level PD is only trusted for the two archives PLAN.md marks safest,
    # and even then the item still records WHY it was classified.
    if source in ("loc", "nara") and not blob.strip():
        return "NEEDS_REVIEW", "LoC/NARA item missing explicit rights field"
    return "NEEDS_REVIEW", "no license evidence at item level"


@dataclass
class AssetRecord:
    asset_id: str            # "<source>:<native_id>"
    source: str              # loc | nara | archive_org | pexels | pixabay | generated
    kind: str                # document | still | footage
    title: str
    description: str = ""
    source_url: str = ""
    media_url: str = ""      # direct downloadable media
    local_path: str = ""     # set after download
    license_raw: str = ""
    license_url: str = ""
    license_class: str = "NEEDS_REVIEW"
    license_reason: str = ""
    attribution: str = ""    # required text for CC_BY
    sha256: str = ""         # content hash after download (dedup key)
    dramatization: bool = False  # generated reconstruction -> on-screen label
    review_note: str = ""    # e.g. "PD film may contain copyrighted music"
    created_at: float = field(default_factory=time.time)

    def classify(self) -> "AssetRecord":
        self.license_class, self.license_reason = classify_license(
            self.license_raw, self.license_url, self.source
        )
        if BANNED_URL_PATTERNS.search(self.source_url or ""):
            self.license_class, self.license_reason = "BANNED", "blocklisted source URL"
        return self


_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  asset_id TEXT PRIMARY KEY, source TEXT, kind TEXT, title TEXT, description TEXT,
  source_url TEXT, media_url TEXT, local_path TEXT,
  license_raw TEXT, license_url TEXT, license_class TEXT, license_reason TEXT,
  attribution TEXT, sha256 TEXT, dramatization INTEGER, review_note TEXT, created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_assets_class ON assets(license_class);
CREATE INDEX IF NOT EXISTS idx_assets_sha ON assets(sha256);
"""


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    def upsert(self, rec: AssetRecord) -> AssetRecord:
        rec.classify()
        d = asdict(rec)
        d["dramatization"] = int(d["dramatization"])
        self.db.execute(
            f"INSERT OR REPLACE INTO assets ({','.join(d)}) VALUES ({','.join(':' + k for k in d)})",
            d,
        )
        self.db.commit()
        return rec

    def get(self, asset_id: str) -> AssetRecord | None:
        row = self.db.execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        return self._to_record(row) if row else None

    def usable(self, asset_id: str) -> bool:
        rec = self.get(asset_id)
        return bool(rec and rec.license_class in USABLE_CLASSES)

    def usable_records(self, kind: str | None = None) -> list[AssetRecord]:
        q = f"SELECT * FROM assets WHERE license_class IN ({','.join('?' * len(USABLE_CLASSES))})"
        args: list = sorted(USABLE_CLASSES)
        if kind:
            q += " AND kind=?"
            args.append(kind)
        return [self._to_record(r) for r in self.db.execute(q, args)]

    def review_queue(self) -> list[AssetRecord]:
        rows = self.db.execute("SELECT * FROM assets WHERE license_class='NEEDS_REVIEW'")
        return [self._to_record(r) for r in rows]

    def register_download(self, asset_id: str, local_path: Path) -> str:
        """Record content hash after download; returns existing asset_id on dedup hit."""
        digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
        dup = self.db.execute(
            "SELECT asset_id FROM assets WHERE sha256=? AND asset_id!=?", (digest, asset_id)
        ).fetchone()
        self.db.execute(
            "UPDATE assets SET local_path=?, sha256=? WHERE asset_id=?",
            (str(local_path), digest, asset_id),
        )
        self.db.commit()
        return dup["asset_id"] if dup else asset_id

    def export_attribution(self) -> list[dict]:
        """Attribution manifest for the video description (CC_BY + provenance trust)."""
        rows = self.db.execute(
            "SELECT * FROM assets WHERE license_class IN ('CC_BY','PD','CC0') AND local_path != ''"
        )
        return [
            {
                "asset_id": r["asset_id"],
                "title": r["title"],
                "source_url": r["source_url"],
                "license": r["license_class"],
                "attribution": r["attribution"],
            }
            for r in rows
        ]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> AssetRecord:
        d = dict(row)
        d["dramatization"] = bool(d["dramatization"])
        return AssetRecord(**d)


def dump_manifest(ledger: Ledger, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "attribution": ledger.export_attribution(),
                "needs_review": [r.asset_id for r in ledger.review_queue()],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
