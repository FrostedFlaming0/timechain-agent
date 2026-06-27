# Experience Capsule format (`.cphyx`)

This document specifies the on-disk format, verification algorithm, and trust
model for Experience Capsules, so a third party can implement a compatible
importer/exporter without reading `capsule.py`. The reference implementation
is `capsule.py`; where this document and the code disagree, the code is
authoritative, but they are intended to match.

**Format version: 2.** A verifier must reject a capsule whose
`header.capsule_format_version` it does not understand.

---

## 1. What a capsule is

An Experience Capsule is a JSON document containing a header and an ordered
list of records selected from one agent's timechain. Each record carries its
**original** signature and hashes, exactly as they sat on the origin chain, so
the bundle is tamper-evident end to end and independent of who holds it. A
Merkle root over the record hashes commits the set as a whole, and a
`capsule_id` binds the header to the contents.

A capsule uses only primitives the timechain already has — Ed25519 signatures,
SHA-256 hashes, canonical JSON, Merkle roots. There is no network protocol, no
token, and no consensus. A capsule is a file.

---

## 2. Canonical JSON

All hashing and signing operate over **canonical JSON**: UTF-8, sorted object
keys, no insignificant whitespace (`separators=(",", ":")`), `ensure_ascii`
false. This is the same `canonical_json` the chain uses (see `chain.py`). Two
implementations MUST produce byte-identical canonical JSON for the same logical
value, or hashes will not match.

Floats are avoided in signed/hashed structures; the fields below are strings,
integers, lists, and nested objects only.

---

## 3. Document structure

```json
{
  "capsule_id": "<hex sha256>",
  "header": {
    "capsule_format_version": 2,
    "origin_pubkey": "<hex ed25519 public key>",
    "created_at": 1730000000000,
    "record_count": 3,
    "merkle_root": "<hex sha256>",
    "title": "",
    "note": ""
  },
  "records": [ <CapsuleRecord>, ... ]
}
```

### 3.1 `CapsuleRecord`

```json
{
  "index": 7,
  "prior_hash": "<hex>",
  "timestamp": 1730000000000,
  "type": "observation",
  "body": { ... },
  "refs": ["<hex>", ...],
  "pubkey": "<hex ed25519 public key>",
  "content_hash": "<hex sha256>",
  "record_hash": "<hex sha256>",
  "signature": "<hex ed25519 signature>",
  "exposure": "shared",
  "epistemic_class": "inferred",
  "redacted": false,
  "summary_commitment": ""
}
```

- `index`, `prior_hash`, `timestamp`, `type`, `refs`, `pubkey`,
  `content_hash`, `record_hash`, `signature` are copied verbatim from the
  origin chain record. They reproduce the exact fields the origin signed.
- `body` is the record's content. For a **full** record it is the original
  content (so `content_hash` re-verifies). For a **redacted** record (see
  §5) it is a summary projection and `content_hash` does NOT cover it.
- `exposure` and `epistemic_class` are the origin record's `_meta` values,
  surfaced at top level so a verifier/importer doesn't have to parse `_meta`.
- `redacted` is true iff the body is a summary projection rather than full
  content.
- `summary_commitment` is the origin's signature over the summary (see §5);
  empty string for non-redacted records.

---

## 4. Export: how a capsule is built

1. **Select** candidate records (all records, or a caller-supplied index
   list), then filter by exposure, type, salience, timestamp window, and
   tags (see §6).
2. **Exposure-gate** each candidate (§5). `private` and `quarantine` records
   are dropped. `summary` records are reduced to a summary projection and
   marked `redacted`. `shared` and `public` records are included in full.
3. For each **redacted** record, sign a **summary commitment** (§5.1).
4. Compute the **Merkle root** over the included records' `record_hash`
   values, in chain order (by `index`), as 32-byte leaves (Bitcoin-style:
   duplicate the last leaf if a layer has an odd count).
5. Build the header and compute `capsule_id` (§7).

An export with an empty selection MUST fail rather than emit an empty capsule.

---

## 5. Exposure gating

`exposure` controls what may cross the agent's boundary — the read side of the
protected-zone membrane:

| exposure | export behavior |
|----------|-----------------|
| `private` | never exported |
| `quarantine` | never exported (untrusted input the chain remembered) |
| `summary` | exported as a summary projection, `redacted=true` |
| `shared` | exported in full |
| `public` | exported in full |

### 5.1 Summary projection and commitment

For a `summary`-exposed record, the body is reduced to a projection: keep only
`title` and/or `summary` string fields if present (else a `"summary":
"(summary withheld)"` placeholder), plus the `_meta` block. The full content is
discarded.

Because the original content is gone, `content_hash` cannot re-verify the
summary. Instead, the origin signs a **summary commitment** over canonical
JSON of:

```json
{
  "kind": "ct-capsule-summary-commitment-v1",
  "origin_record_hash": "<the record's record_hash>",
  "summary": <the summary body>
}
```

The signature (hex) is stored in `summary_commitment`. Binding the summary to
the origin `record_hash` prevents lifting a commitment from one record onto
another. A verifier MUST require a valid commitment on every redacted record
(§8 step 6).

---

## 6. Selection filters (export options)

All optional; combined with logical AND:

- **indices**: consider exactly these record indices (in chain order) instead
  of the whole chain.
- **type_filter**: keep only records of this `type`.
- **min_salience**: drop records whose `_meta.salience` is below this float.
- **after_ms / before_ms**: keep records whose `timestamp` (ms since epoch)
  is in the half-open window `[after_ms, before_ms)`. Either bound may be
  omitted.
- **tags**: keep only records whose content carries a `tags` list intersecting
  the requested tags. Records with no `tags` field are dropped when a tag
  filter is active.

---

## 7. `capsule_id`

```
capsule_id = hex( sha256( canonical_json({
  "header": <the header object>,
  "record_hashes": [ <each record's record_hash, in order> ]
}) ) )
```

The id binds the header (including the Merkle root) to the ordered record
hashes. Any change to the header or to the set/order of records changes the id.

---

## 8. Verification algorithm

A verifier MUST perform all of the following, and reject the capsule wholesale
on the first failure (no partial trust):

1. `header.capsule_format_version` is understood (== 2 for this spec).
2. `header.origin_pubkey` is present and non-empty.
3. The capsule contains at least one record; each parses into a CapsuleRecord.
4. For each record:
   a. `pubkey` equals `header.origin_pubkey`.
   b. If **not** redacted: `sha256(canonical_json(body))` equals
      `content_hash`; and `sha256(canonical_json(signing_payload))` equals
      `record_hash`, where `signing_payload` is the object
      `{index, prior_hash, timestamp, type, content: body, refs, pubkey,
      content_hash}` (the exact fields and order the chain signs — see
      `chain.py`).
   c. The Ed25519 `signature` verifies against `pubkey` over the bytes
      `record_hash` (hex-decoded). This holds for redacted records too —
      provenance is independent of body redaction.
   d. If **redacted**: `summary_commitment` is present and verifies (Ed25519)
      against `pubkey` over the canonical-JSON commitment message of §5.1.
      A missing or invalid commitment is a failure.
5. The Merkle root recomputed over the records' `record_hash` leaves (chain
   order) equals `header.merkle_root`.
6. The recomputed `capsule_id` (§7) equals the stated `capsule_id`.

On success, a redacted record's **summary** is authentic (origin-signed) but
**incomplete** (the full content was withheld). A reader knows the summary,
not the underlying record.

---

## 9. Import and trust model

Importing a capsule means ingesting another party's signed claims. The
importer MUST:

1. **Verify first** (§8) and import nothing if verification fails.
2. Append each origin record as a **new** local record of type
   `imported_capsule` (append-only — never rewrite or delete; the local
   chain's own verification is unaffected). The local record is authored and
   signed by the importing agent's key; the origin provenance is preserved
   inside the content:
   ```json
   {
     "capsule_id": "...",
     "origin_pubkey": "...",
     "origin_index": 7,
     "origin_record_hash": "...",
     "origin_signature": "...",
     "origin_type": "observation",
     "origin_redacted": false,
     "imported_body": { ... },
     "_meta": { ... }
   }
   ```
3. Record cautious metadata: `source = "peer_agent"`, `epistemic_class`
   demoted to no stronger than `inferred` (a stronger origin class is lowered;
   a weaker one — `speculative`, `disputed` — is preserved), `exposure =
   private` (imported memory does not re-export onward without a fresh
   decision), and low default salience/confidence.
4. **Deduplicate** by `capsule_id`: re-importing an already-imported capsule
   is a no-op.

Imported memory must never be presented as the importing agent's own
first-person history. It is an attributed third-party claim.

---

## 10. Reference API (`capsule.py`)

- `export_capsule(chain, *, indices=, type_filter=, min_salience=, after_ms=,
  before_ms=, tags=, title=, note=) -> dict`
- `write_capsule(capsule, path)` / `read_capsule(path) -> dict`
- `verify_capsule(capsule) -> (ok: bool, message: str)`
- `import_capsule(chain, capsule, *, build_meta_fn, skip_if_imported=True)
  -> dict`
- `already_imported(chain, capsule_id) -> bool`

REPL: `/export-capsule <path>`, `/import-capsule <path>`.
Webapp: `GET /api/capsule/export`, `POST /api/capsule/import` (both require a
session token; import enforces the same verify-before-import gate).
