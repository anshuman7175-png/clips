"""Asset harvesters (PLAN.md Layer 4 sources).

Each harvester exposes  search(query, kind, limit) -> list[AssetRecord]
and is responsible for ITEM-LEVEL license extraction, never source-level
assumptions. Harvest order follows the footage priority (Layer 0.3):

  real documents/archival (LoC, NARA)  >  Internet Archive (item-checked)
  >  stock (Pexels, Pixabay)           >  generated (separate Wan branch)

A failing source is logged and skipped - one dead API must never stall
the pipeline (free-tier APIs are flaky by nature).
"""

from __future__ import annotations

from ..ledger import AssetRecord, Ledger
from . import internet_archive, loc, nara, stock

# Priority-ordered: index doubles as the tie-break rank in the match gate.
SOURCE_ORDER = ["loc", "nara", "archive_org", "pexels", "pixabay"]

_MODULES = {
    "loc": loc,
    "nara": nara,
    "archive_org": internet_archive,
    "pexels": stock.pexels_search,
    "pixabay": stock.pixabay_search,
}


def harvest(ledger: Ledger, query: str, kind: str, limit_per_source: int = 8) -> list[AssetRecord]:
    """Search every source for `query`, classify licenses, persist to ledger.

    Returns only records the ledger classifies as usable; NEEDS_REVIEW items
    are persisted too (for the human review queue) but not returned.
    """
    usable: list[AssetRecord] = []
    for name in SOURCE_ORDER:
        mod = _MODULES[name]
        search_fn = mod if callable(mod) else mod.search
        try:
            records = search_fn(query, kind, limit_per_source)
        except Exception as err:  # noqa: BLE001 - a dead free API must not stall the run
            print(f"[harvest:{name}] SKIPPED ({type(err).__name__}: {err})")
            continue
        for rec in records:
            ledger.upsert(rec)
            if rec.license_class in ("PD", "CC0", "CC_BY", "STOCK_FREE"):
                usable.append(rec)
        print(f"[harvest:{name}] {len(records)} items, "
              f"{sum(r.license_class not in ('BANNED', 'NEEDS_REVIEW') for r in records)} usable")
    return usable
