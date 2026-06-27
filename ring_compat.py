"""
ring_compat — the single adapter that lets storage-independent cognitive
logic ported from `cypher-tempre-self-model` run unchanged against this
repo's signed SQLite chain.

The skill speaks "rings" (plain dicts with `index` / `ring_type` / `payload`
/ `prev_hash` / `ring_hash` / `poq`). This repo speaks signed, hash-linked
`Record`s whose architectural metadata lives inside `content["_meta"]`.
Most ported modules (PoQ verdicts, immune, continuum, recall, chronosynaptic,
consensus, Cambium growth) only need a tiny read/write shim:

  - read side  : `record_to_ring()` / `load_rings()` present repo Records in
                 the ring shape the ported logic expects.
  - write side : `seal_ring()` seals a skill-style payload as a proper repo
                 record — routed through `chain.append` + `build_meta` so it
                 is signed, hash-linked, and carries correct `_meta`.

Naming: this logic keeps the repo's neutral register. `ring` here is just this
repo's record presented in a generic dict shape — no new storage.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from chain import Chain, Record
from metadata import build_meta


def record_to_ring(rec: Record) -> dict:
    """Present a repo `Record` in the skill's ring shape (read side).

    `payload` maps to the record's content dict (where skill payloads live),
    and `ring_hash` maps to this repo's tamper-evidence anchor, `record_hash`.
    `poq` is lifted out of `content["_meta"]` for convenience.
    """
    content = rec.content if isinstance(rec.content, dict) else {"value": rec.content}
    meta = content.get("_meta", {}) if isinstance(content, dict) else {}
    return {
        "index": rec.index,
        "ring_type": rec.type,
        "timestamp": rec.timestamp,
        "prev_hash": rec.prior_hash,
        "ring_hash": rec.record_hash,          # repo's tamper-evidence anchor
        "payload": content,                    # skill payloads live in content
        "poq": meta.get("poq") if isinstance(meta, dict) else None,
    }


def load_rings(chain: Chain, exclude_quarantined: bool = True) -> list[dict]:
    """All records as rings, with quarantined records excluded by default.

    Mirrors the skill's "active self excludes molted scars": quarantined
    records are filtered through the repo's existing protected-zone membrane
    so ported cognition reasons only over the agent's active history.
    """
    recs = list(chain.iter_records())
    if exclude_quarantined:
        import protected_zones
        recs = protected_zones.filter_quarantined(recs)
    return [record_to_ring(r) for r in recs]


def seal_ring(
    chain: Chain,
    ring_type: str,
    payload: dict,
    *,
    source: str = "assistant",
    salience: Optional[float] = None,
    poq: Optional[dict] = None,
    refs: Optional[Iterable[str]] = None,
    difficulty: int = 0,
    **meta_kwargs: Any,
) -> Record:
    """Seal a skill-style payload as a proper repo record (write side).

    The `_meta` block is built by `metadata.build_meta` and nested under the
    canonical `content["_meta"]` key — NOT spread at content top-level — so
    `metadata.read_meta` finds it and the record carries real architectural
    metadata. Extra metadata fields (epistemic_class, exposure, supersedes,
    modalities_activated, ...) pass through via `**meta_kwargs`.
    """
    content = dict(payload)
    content["_meta"] = build_meta(
        ring_type,
        source=source,
        salience=salience,
        poq=poq,
        **meta_kwargs,
    )
    return chain.append(ring_type, content, refs=refs, difficulty=difficulty)


def ring_text(ring: dict) -> str:
    """Flatten a ring's payload into a single string of its text values.

    Used by ported cognitive modules (PoQ proxies, recall labeling, immune
    scan) that score over the textual content of past rings.
    """
    return " ".join(_strings(ring.get("payload", {})))


def _strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k == "_meta":
                continue  # metadata is not content; don't pollute text scoring
            out += _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _strings(v)
    return out
