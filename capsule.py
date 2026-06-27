"""
capsule — Experience Capsules: signed, portable, verifiable bundles of Rings.

An Experience Capsule is a selection of one agent's records, exported as a
self-contained signed bundle that another agent (or the same agent on another
machine) can verify and import. This is the build spec's `.cphyx` exchange
format and the foundation for the spec's 5D "fleet Timechains / shared
experience markets" tier — built using only cryptography the chain already
has (Ed25519 signatures, SHA-256 hashes, canonical JSON, Merkle roots). No
network, no tokens, no consensus.

What a capsule IS:
  - A JSON document containing a header and a list of records, where each
    record carries its ORIGINAL signature and hashes intact, exactly as it
    sat on the origin chain. Verification re-checks those signatures against
    the origin agent's public key — so a capsule is tamper-evident end to
    end, independent of who is holding it.
  - A Merkle root over the included records' hashes, so the set as a whole is
    committed and a single altered/added/removed record is detectable.

What a capsule is NOT:
  - It is NOT a way to graft another agent's history into your own as if it
    were your own first-person experience. Imported records are appended as
    NEW records of type `imported_capsule` (build spec `source:
    imported_capsule`), attributed to the origin agent, with their original
    identity preserved inside the content. They are the agent's memory of
    *what another agent reported*, never silently relabeled as its own.
  - It does NOT bypass the append-only rule. Import only ever appends. It
    never rewrites or deletes anything, and `/verify` on the local chain is
    unaffected (the imported records are ordinary signed records authored by
    the importing key, wrapping the foreign payload).

Trust model (this matters):
  Importing a capsule means ingesting another party's claims. Two safeguards:
    1. Cryptographic: every included record's original signature is verified
       against the stated origin pubkey, and the capsule's Merkle root is
       recomputed and checked, before anything is imported. A capsule that
       fails verification is rejected wholesale — partial import of a
       tampered bundle is never allowed.
    2. Epistemic: imported content is recorded with a deliberately cautious
       epistemic class (never `known`; demoted to at most `inferred`, and
       `disputed`/`speculative` preserved if lower). It is `source`-tagged as
       imported and carries the origin pubkey, so retrieval and PoQ treat it
       as attributed third-party memory, not ground truth. Exposure is
       forced to `private` on import — an imported capsule never re-exports
       onward without a fresh explicit decision.

Exposure on EXPORT (the read-side of the protected-zone membrane):
  Records are filtered by their `_meta.exposure` before they may leave:
    - private    -> never exported
    - quarantine -> never exported (it's untrusted input the chain remembered)
    - summary    -> exported, but only the summary/title, not full content
    - shared     -> exported in full (this is what `shared` is FOR)
    - public     -> exported in full
  This is what finally makes the `exposure` field load-bearing: it gates what
  may cross the agent's boundary.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from chain import (
    Chain,
    Record,
    canonical_json,
    sha256,
    sha256_hex,
    merkle_root,
    _verify_signature,
    GENESIS_PRIOR_HASH,
)
from cryptography.exceptions import InvalidSignature
from metadata import (
    read_meta,
    SOURCE_PEER_AGENT,
    EXPOSURE_PRIVATE,
    EXPOSURE_SUMMARY,
    EXPOSURE_SHARED,
    EXPOSURE_PUBLIC,
    EXPOSURE_QUARANTINE,
    EPISTEMIC_KNOWN,
    EPISTEMIC_INFERRED,
    EPISTEMIC_SPECULATIVE,
    EPISTEMIC_DISPUTED,
    EPISTEMIC_USER_CONTEXT,
)


# Capsule format version. Bumped if the on-disk JSON layout changes; an
# importer checks it and refuses a format it doesn't understand rather than
# silently mis-parsing.
#   v1 — original layout.
#   v2 — adds per-record `summary_commitment` (issue #1): redacted records
#        carry an origin signature over their summary, making the summary text
#        verifiable. A v2 verifier requires the commitment on redacted
#        records, so it will not accept a v1 capsule's unsigned summaries.
CAPSULE_FORMAT_VERSION = 2

# The record type imported records are appended under. Distinct from every
# native record type so imported memory is always identifiable and filterable
# (build spec `source: imported_capsule`).
IMPORTED_RECORD_TYPE = "imported_capsule"

# Which exposure classes may leave the agent at all, and whether they export
# in full or summarized form. This is the export-side enforcement of the
# protected-zone membrane.
_EXPORT_FULL = {EXPOSURE_SHARED, EXPOSURE_PUBLIC}
_EXPORT_SUMMARY_ONLY = {EXPOSURE_SUMMARY}
_EXPORT_NEVER = {EXPOSURE_PRIVATE, EXPOSURE_QUARANTINE}

# On import, an origin record's epistemic class is demoted to no stronger than
# this. Imported claims are never treated as the importer's own ground truth.
_IMPORT_EPISTEMIC_CEILING = EPISTEMIC_INFERRED

# Ordering of epistemic strength, strongest first, for the import demotion.
_EPISTEMIC_STRENGTH = [
    EPISTEMIC_KNOWN,
    EPISTEMIC_USER_CONTEXT,
    EPISTEMIC_INFERRED,
    EPISTEMIC_SPECULATIVE,
    EPISTEMIC_DISPUTED,
]


class CapsuleError(Exception):
    """Raised when a capsule fails to build, parse, or verify."""


@dataclass
class CapsuleRecord:
    """
    One record inside a capsule. Mirrors the signed fields of chain.Record so
    its original signature can be re-verified, plus a `body` that is either
    the full content or a summary-only projection depending on exposure.
    """
    index: int
    prior_hash: str
    timestamp: int
    type: str
    body: Any            # full content, or summary projection for summary-only
    refs: list
    pubkey: str
    content_hash: str
    record_hash: str
    signature: str
    exposure: str
    epistemic_class: str
    redacted: bool       # True if body was reduced to a summary on export
    # For redacted records only: the origin's Ed25519 signature over a
    # canonical commitment binding (origin record_hash + summary body). This
    # makes the summary text verifiable in its own right — provably what the
    # origin endorsed as the summary of that record — rather than merely an
    # unverified projection. Empty string for non-redacted records (their full
    # body is already content-hash verifiable).
    summary_commitment: str = ""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "prior_hash": self.prior_hash,
            "timestamp": self.timestamp,
            "type": self.type,
            "body": self.body,
            "refs": list(self.refs),
            "pubkey": self.pubkey,
            "content_hash": self.content_hash,
            "record_hash": self.record_hash,
            "signature": self.signature,
            "exposure": self.exposure,
            "epistemic_class": self.epistemic_class,
            "redacted": self.redacted,
            "summary_commitment": self.summary_commitment,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CapsuleRecord":
        try:
            return cls(
                index=int(d["index"]),
                prior_hash=str(d["prior_hash"]),
                timestamp=int(d["timestamp"]),
                type=str(d["type"]),
                body=d["body"],
                refs=list(d.get("refs", [])),
                pubkey=str(d["pubkey"]),
                content_hash=str(d["content_hash"]),
                record_hash=str(d["record_hash"]),
                signature=str(d["signature"]),
                exposure=str(d.get("exposure", EXPOSURE_SHARED)),
                epistemic_class=str(d.get("epistemic_class", EPISTEMIC_INFERRED)),
                redacted=bool(d.get("redacted", False)),
                summary_commitment=str(d.get("summary_commitment", "")),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CapsuleError(f"malformed capsule record: {e}") from e


# ---------------------------------------------------------------------------
# Summary projection for summary-only exposure
# ---------------------------------------------------------------------------

def _summary_commitment_message(origin_record_hash: str, summary_body: Any) -> bytes:
    """
    The canonical bytes the origin signs to commit to a redacted record's
    summary. Binds the summary body to the specific origin record_hash, so a
    summary signature cannot be lifted from one record and replayed onto
    another. Deterministic (canonical JSON) so verification recomputes the
    exact same bytes.
    """
    return canonical_json({
        "kind": "ct-capsule-summary-commitment-v1",
        "origin_record_hash": origin_record_hash,
        "summary": summary_body,
    })


def _summary_projection(content: Any) -> Any:
    """
    Reduce a record's content to a shareable summary for `summary` exposure.

    We keep only a title/summary if present, plus the _meta block (which is
    non-secret schema). The full text/body is dropped. This deliberately
    BREAKS the content_hash for the exported body — a summary-only record is
    flagged `redacted=True` and is NOT re-verifiable against the original
    content_hash (you can't: we removed content). Its record_hash/signature
    are still carried for provenance, but the importer treats a redacted
    record as unverifiable-content and imports it only as a low-trust
    attributed note (see verify()/import).
    """
    if not isinstance(content, dict):
        return {"summary": "(summary withheld)"}
    out: dict = {}
    for key in ("title", "summary"):
        if key in content and isinstance(content[key], str):
            out[key] = content[key]
    if not out:
        out["summary"] = "(summary withheld)"
    if "_meta" in content and isinstance(content["_meta"], dict):
        out["_meta"] = content["_meta"]
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_capsule(
    chain: Chain,
    *,
    indices: Optional[list[int]] = None,
    type_filter: Optional[str] = None,
    min_salience: float = 0.0,
    after_ms: Optional[int] = None,
    before_ms: Optional[int] = None,
    tags: Optional[list[str]] = None,
    title: str = "",
    note: str = "",
) -> dict:
    """
    Build an Experience Capsule from a chain.

    Selection: if `indices` is given, exactly those records are considered
    (in chain order); otherwise every record is considered. The candidate set
    is then filtered by:
      - exposure: private/quarantine are dropped entirely; summary is
        included summary-only; shared/public are included in full.
      - type_filter: if set, only records of that type.
      - min_salience: drop records below this salience.
      - after_ms / before_ms: keep only records whose timestamp (ms since
        epoch) falls in the half-open window [after_ms, before_ms). Either
        bound may be omitted. Lets an operator export "the last week" or "that
        project's window" rather than the whole chain.
      - tags: if set, keep only records whose content carries a `tags` list
        intersecting the requested tags. Records without a `tags` field are
        dropped when a tag filter is active (they can't match). Inert when
        `tags` is None — forward-compatible with a future tagging scheme
        without requiring one now.

    Genesis is never auto-included unless explicitly named in `indices` AND
    its exposure permits export (genesis defaults to `summary`, so by default
    it would export summary-only). The returned dict is a JSON-serializable
    capsule. Use `write_capsule` to persist it.

    Raises CapsuleError if the selection is empty after filtering — an empty
    capsule is almost always a mistake (wrong filter), so we fail loud.
    """
    if indices is not None:
        candidates = [r for i in indices if (r := chain.get(i)) is not None]
    else:
        candidates = list(chain.iter_records())

    tag_set = set(tags) if tags else None

    selected: list[CapsuleRecord] = []
    for rec in candidates:
        meta = read_meta(rec)
        exposure = meta.exposure
        if exposure in _EXPORT_NEVER:
            continue
        if type_filter is not None and rec.type != type_filter:
            continue
        if meta.salience < min_salience:
            continue
        if after_ms is not None and rec.timestamp < after_ms:
            continue
        if before_ms is not None and rec.timestamp >= before_ms:
            continue
        if tag_set is not None:
            rec_tags = rec.content.get("tags") if isinstance(rec.content, dict) else None
            if not isinstance(rec_tags, list) or not (tag_set & set(rec_tags)):
                continue

        redacted = exposure in _EXPORT_SUMMARY_ONLY
        body = _summary_projection(rec.content) if redacted else rec.content
        # For a redacted record, sign a commitment binding the summary body to
        # the origin record_hash, using the chain's signing key. This makes
        # the summary text verifiable in its own right (issue #1): a verifier
        # can confirm the origin endorsed exactly this summary for exactly this
        # record, rather than trusting an arbitrary projection.
        summary_commitment = ""
        if redacted:
            msg = _summary_commitment_message(rec.record_hash, body)
            summary_commitment = chain.signing_key.sign(msg).hex()
        selected.append(CapsuleRecord(
            index=rec.index,
            prior_hash=rec.prior_hash,
            timestamp=rec.timestamp,
            type=rec.type,
            body=body,
            refs=list(rec.refs),
            pubkey=rec.pubkey,
            content_hash=rec.content_hash,
            record_hash=rec.record_hash,
            signature=rec.signature,
            exposure=exposure,
            epistemic_class=meta.epistemic_class,
            redacted=redacted,
            summary_commitment=summary_commitment,
        ))

    if not selected:
        raise CapsuleError(
            "no records selected for export (all filtered out by "
            "exposure/type/salience). Refusing to write an empty capsule."
        )

    # Merkle root over the included records' record_hash bytes, committing the
    # set as a whole. Order is chain order (by index), which is stable.
    leaves = [bytes.fromhex(cr.record_hash) for cr in selected]
    root = merkle_root(leaves).hex()

    header = {
        "capsule_format_version": CAPSULE_FORMAT_VERSION,
        "origin_pubkey": chain.pubkey_hex,
        "created_at": int(time.time() * 1000),
        "record_count": len(selected),
        "merkle_root": root,
        "title": title,
        "note": note,
    }
    # The capsule_id binds the header to the contents: a hash over the header
    # (minus the id itself) and the ordered record hashes.
    id_payload = {
        "header": header,
        "record_hashes": [cr.record_hash for cr in selected],
    }
    capsule_id = sha256_hex(canonical_json(id_payload))

    return {
        "capsule_id": capsule_id,
        "header": header,
        "records": [cr.to_dict() for cr in selected],
    }


def write_capsule(capsule: dict, path: str) -> None:
    """Persist a capsule dict to a .cphyx (JSON) file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(capsule, f, ensure_ascii=False, indent=2)


def read_capsule(path: str) -> dict:
    """Load a capsule dict from a .cphyx (JSON) file."""
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise CapsuleError(f"not a valid capsule file: {e}") from e


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_capsule(capsule: dict) -> tuple[bool, str]:
    """
    Verify a capsule end to end, WITHOUT importing it:

      - format version is understood
      - header well-formed; origin_pubkey present
      - each non-redacted record's content_hash matches its body
      - each record's record_hash matches its signing payload
      - each record's original signature verifies against origin_pubkey
      - the Merkle root recomputes to the header value
      - the capsule_id recomputes to the stated value

    Redacted (summary-only) records are exempt from the content_hash check
    (their body was intentionally reduced); their record_hash/signature are
    still verified for provenance, but the importer flags them.

    Returns (ok, message). On failure, message identifies the first problem.
    A capsule that fails ANY check is rejected wholesale — partial trust in a
    tampered bundle is never granted.
    """
    if not isinstance(capsule, dict):
        return False, "capsule is not an object"
    if capsule.get("header", {}).get("capsule_format_version") != CAPSULE_FORMAT_VERSION:
        return False, "unsupported capsule_format_version"

    header = capsule["header"]
    origin = header.get("origin_pubkey")
    if not isinstance(origin, str) or not origin:
        return False, "missing origin_pubkey"

    try:
        records = [CapsuleRecord.from_dict(d) for d in capsule.get("records", [])]
    except CapsuleError as e:
        return False, str(e)
    if not records:
        return False, "capsule contains no records"

    for cr in records:
        if cr.pubkey != origin:
            return False, f"record {cr.index} pubkey does not match origin_pubkey"

        # content_hash check (skipped for redacted/summary-only bodies).
        if not cr.redacted:
            recomputed = sha256_hex(canonical_json(cr.body))
            if recomputed != cr.content_hash:
                return False, f"content_hash mismatch at origin index {cr.index}"

        # record_hash check: rebuild the exact signing payload chain.py used.
        signing_payload = {
            "index": cr.index,
            "prior_hash": cr.prior_hash,
            "timestamp": cr.timestamp,
            "type": cr.type,
            "content": cr.body if not cr.redacted else _SIGNING_CONTENT_UNAVAILABLE,
            "refs": list(cr.refs),
            "pubkey": cr.pubkey,
            "content_hash": cr.content_hash,
        }
        if not cr.redacted:
            recomputed_rh = sha256_hex(canonical_json(signing_payload))
            if recomputed_rh != cr.record_hash:
                return False, f"record_hash mismatch at origin index {cr.index}"

        # signature check against the carried record_hash (always — provenance
        # holds even for redacted records).
        try:
            _verify_signature(
                bytes.fromhex(cr.pubkey),
                bytes.fromhex(cr.signature),
                bytes.fromhex(cr.record_hash),
            )
        except (InvalidSignature, ValueError) as e:
            return False, f"bad signature at origin index {cr.index}: {e}"

        # Summary commitment check (issue #1): a redacted record carries the
        # origin's signature over (origin record_hash + summary body). Verify
        # it so the summary text is provably the origin's endorsed summary for
        # this specific record, not an arbitrary substitution. A redacted
        # record MUST carry a valid commitment — a missing or bad one is a
        # tamper signal, not a soft warning.
        if cr.redacted:
            if not cr.summary_commitment:
                return False, (
                    f"redacted record {cr.index} missing summary commitment"
                )
            msg = _summary_commitment_message(cr.record_hash, cr.body)
            try:
                _verify_signature(
                    bytes.fromhex(cr.pubkey),
                    bytes.fromhex(cr.summary_commitment),
                    msg,
                )
            except (InvalidSignature, ValueError) as e:
                return False, (
                    f"bad summary commitment at origin index {cr.index}: {e}"
                )
        elif cr.summary_commitment:
            # A non-redacted record has no legitimate reason to carry a summary
            # commitment (export only signs commitments for redacted records).
            # Its presence means the capsule was edited after export. The field
            # is inert on import, so this isn't exploitable — but a capsule
            # should be fully tamper-evident, and an unexpected field is a
            # tamper signal, so reject rather than silently ignore.
            return False, (
                f"non-redacted record {cr.index} carries an unexpected "
                f"summary commitment"
            )

    # Merkle root recompute.
    leaves = [bytes.fromhex(cr.record_hash) for cr in records]
    recomputed_root = merkle_root(leaves).hex()
    if recomputed_root != header.get("merkle_root"):
        return False, "merkle_root mismatch (record set altered)"

    # capsule_id recompute.
    id_payload = {
        "header": header,
        "record_hashes": [cr.record_hash for cr in records],
    }
    if sha256_hex(canonical_json(id_payload)) != capsule.get("capsule_id"):
        return False, "capsule_id mismatch (header or contents altered)"

    # Provenance is fully verified: signatures, record_hashes, set membership
    # (Merkle), and id all check out. Redacted records' SUMMARY BODIES are now
    # also verified, via the per-record summary commitment (issue #1) — the
    # summary text is provably the origin's endorsed summary for that record.
    # The full original content is still withheld by design, so a reader knows
    # only the summary, not the underlying record; but the summary itself is no
    # longer an unverified projection. Import still demotes redacted records
    # (the agent learns only a summary, which is inherently less than the full
    # record), but the demotion is now about completeness, not authenticity.
    redacted_n = sum(1 for cr in records if cr.redacted)
    if redacted_n:
        return True, (
            f"capsule ok ({len(records)} records from {origin[:16]}...; "
            f"{redacted_n} summary-only, summaries commitment-verified)"
        )
    return True, f"capsule ok ({len(records)} records from {origin[:16]}...)"


# Sentinel used only when rebuilding a redacted record's signing payload,
# where the original content is unavailable. Redacted records skip the
# record_hash recompute, so this is never actually hashed; it exists to make
# the payload dict shape explicit rather than silently omitting the key.
_SIGNING_CONTENT_UNAVAILABLE = {"__redacted__": True}


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def _demote_epistemic(origin_class: str) -> str:
    """
    Demote an imported record's epistemic class to no stronger than the import
    ceiling (`inferred`). A class already weaker than the ceiling
    (speculative, disputed) is preserved. Unknown classes default to the
    ceiling.
    """
    try:
        origin_rank = _EPISTEMIC_STRENGTH.index(origin_class)
    except ValueError:
        return _IMPORT_EPISTEMIC_CEILING
    ceiling_rank = _EPISTEMIC_STRENGTH.index(_IMPORT_EPISTEMIC_CEILING)
    # Higher rank index = weaker. Keep the weaker (max index) of the two.
    return _EPISTEMIC_STRENGTH[max(origin_rank, ceiling_rank)]


def already_imported(chain: Chain, capsule_id: str) -> bool:
    """
    True if a capsule with this id has already been imported into this chain.
    Replay/dedup guard: re-importing the same capsule is a no-op rather than a
    duplication. Scans imported_capsule records' content for the capsule_id.
    """
    for rec in chain.iter_records():
        if rec.type != IMPORTED_RECORD_TYPE:
            continue
        if isinstance(rec.content, dict) and rec.content.get("capsule_id") == capsule_id:
            return True
    return False


def import_capsule(
    chain: Chain,
    capsule: dict,
    *,
    build_meta_fn,
    skip_if_imported: bool = True,
) -> dict:
    """
    Verify and import a capsule into `chain`. Appends one new record per
    included origin record, of type `imported_capsule`, attributed to the
    origin agent and recorded with cautious epistemic class and private
    exposure. Append-only: nothing is rewritten or deleted.

    `build_meta_fn` is metadata.build_meta (injected to avoid a hard import
    cycle and to let callers pass a wrapped version in tests).

    Returns a summary dict: {ok, reason, imported_count, capsule_id,
    skipped (bool)}. Raises CapsuleError only on a verification failure —
    a tampered or malformed capsule is never partially imported.
    """
    ok, msg = verify_capsule(capsule)
    if not ok:
        raise CapsuleError(f"refusing to import: {msg}")

    capsule_id = capsule["capsule_id"]
    origin = capsule["header"]["origin_pubkey"]

    if skip_if_imported and already_imported(chain, capsule_id):
        return {
            "ok": True,
            "reason": "already imported (dedup)",
            "imported_count": 0,
            "capsule_id": capsule_id,
            "skipped": True,
        }

    records = [CapsuleRecord.from_dict(d) for d in capsule["records"]]
    imported = 0
    for cr in records:
        imported_epistemic = _demote_epistemic(cr.epistemic_class)
        # A redacted record carries only a summary, not the full original
        # content — but as of issue #1 that summary is commitment-verified
        # (the origin signed it), so it is authentic, just incomplete. We no
        # longer force it to `speculative` on that basis; the standard import
        # demotion (to `inferred` or weaker) applies, same as any other
        # imported record. The fact that it is a summary is preserved in
        # `origin_redacted` on the content for a reader that cares.
        # Wrap the origin payload. The local record is authored by THIS
        # chain's key (ordinary append), but its content preserves the full
        # origin provenance so the foreign signature remains independently
        # checkable later and the agent never mistakes it for its own memory.
        content = {
            "capsule_id": capsule_id,
            "origin_pubkey": origin,
            "origin_index": cr.index,
            "origin_record_hash": cr.record_hash,
            "origin_signature": cr.signature,
            "origin_type": cr.type,
            "origin_redacted": cr.redacted,
            "imported_body": cr.body,
            "_meta": build_meta_fn(
                IMPORTED_RECORD_TYPE,
                # Provenance: another agent's signed claim, imported. Recorded
                # as `peer_agent` (now a first-class source in
                # metadata.VALID_SOURCES), with the full origin identity also
                # preserved in the content above so the foreign signature
                # stays independently checkable.
                source=SOURCE_PEER_AGENT,
                epistemic_class=imported_epistemic,
                # Imported memory never re-exports onward without a fresh
                # explicit decision: force private exposure.
                exposure=EXPOSURE_PRIVATE,
                # Imported claims are low-to-moderate salience by default —
                # attributed third-party notes, not the agent's own
                # high-salience conclusions.
                salience=0.35,
                confidence=0.6,
            ),
        }
        chain.append(IMPORTED_RECORD_TYPE, content)
        imported += 1

    return {
        "ok": True,
        "reason": "imported",
        "imported_count": imported,
        "capsule_id": capsule_id,
        "skipped": False,
    }
