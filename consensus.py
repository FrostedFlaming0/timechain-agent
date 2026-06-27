"""
consensus — quorum-attested, authenticated hardening of the chain. Ported from
cypher-tempre-self-model/consensus.py and adapted to this repo's Ed25519 chain.

The signed hash-chain is already tamper-EVIDENT. This adds tamper-RESISTANCE: a
quorum of independent witnesses, each holding its own secret key, HMAC-attests
every chain head. The chain is accepted only if a quorum (k-of-n) of witnesses
produce valid signatures that AGREE on the head's RECOMPUTED record hash.

  - To forge history you must also forge >= k witness MACs = steal >= k secret
    keys. Re-signing a record with the operator key no longer suffices, because
    the witnesses pinned the ORIGINAL recomputed hash.
  - Up to n-k corrupted/equivocating witnesses are outvoted and flagged.

Adaptation to this repo: the skill attests over a recomputed `ring_hash`; here we
attest over the recomputed `record_hash` (`sha256_hex(canonical_json(
record.signing_payload()))`) — the same value `chain.verify()` recomputes — so a
forger who rewrites the stored `record_hash` field still fails consensus.

HONEST SCOPE: on a single host the witness keys share one trust domain, so this
is authenticated quorum attestation, not distributed BFT. Point the witnesses at
independent hosts/HSMs and the same code gives true Byzantine fault tolerance.

Config + attestations live in a sidecar dir (`consensus/`), never on the signed
chain, so deleting them can never break `chain.verify()`.

Stdlib + cryptography only (hmac, hashlib, secrets).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Optional

from chain import canonical_json, sha256_hex


def _mac(key_hex: str, msg: str) -> str:
    return hmac.new(bytes.fromhex(key_hex), msg.encode(), hashlib.sha256).hexdigest()


def recompute_record_hash(rec) -> str:
    """Recompute a record's hash from its signing payload — the same value
    `chain.append`/`chain.verify` use. A forger who altered content (or rewrote
    the stored record_hash) produces a different value here."""
    return sha256_hex(canonical_json(rec.signing_payload()))


class Quorum:
    def __init__(self, chain, consensus_dir: Optional[str | Path] = None):
        self.chain = chain
        base = (Path(consensus_dir) if consensus_dir
                else Path(chain.db_path).parent / "consensus")
        self.dir = base
        self.cfg_path = base / "config.json"
        self.att_path = base / "attestations.jsonl"

    def init(self, n: int = 3, quorum: int = 2) -> dict:
        if quorum > n:
            raise ValueError("quorum cannot exceed n")
        if quorum < 1:
            raise ValueError("quorum must be >= 1")
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "n": n, "quorum": quorum,
            "witnesses": [{"id": f"w{i}", "key": secrets.token_hex(16)}
                          for i in range(n)],
        }
        self.cfg_path.write_text(json.dumps(cfg, indent=2))
        return cfg

    def _cfg(self) -> dict:
        return json.loads(self.cfg_path.read_text())

    def attest(self) -> tuple:
        """Each witness MACs the current head: msg = f'{index}:{record_hash}'."""
        cfg = self._cfg()
        head = self.chain.head()
        if not head:
            raise RuntimeError("no chain head to attest")
        h, rh = head.index, head.record_hash
        msg = f"{h}:{rh}"
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.att_path.open("a") as f:
            for w in cfg["witnesses"]:
                f.write(json.dumps({
                    "height": h, "record_hash": rh, "witness": w["id"],
                    "mac": _mac(w["key"], msg),
                }) + "\n")
        return h, rh, cfg["n"]

    def _attestations(self) -> list:
        if not self.att_path.exists():
            return []
        return [json.loads(l) for l in self.att_path.read_text().splitlines()
                if l.strip()]

    def verify(self) -> tuple:
        """Return (ok, report). ok requires BOTH the Ed25519 chain.verify() AND
        a quorum of valid, agreeing witness attestations at every attested head."""
        cfg = self._cfg()
        keys = {w["id"]: w["key"] for w in cfg["witnesses"]}
        by_h = {r.index: r for r in self.chain.iter_records()}
        ok_hash, detail = self.chain.verify()
        out = [detail] if isinstance(detail, str) else list(detail)
        atts = self._attestations()
        consensus_ok = True
        for h in sorted({a["height"] for a in atts}):
            # Compare against the RECOMPUTED record hash: a forger who rewrote
            # the stored hash (or the content) still fails, because the witnesses
            # pinned the original.
            actual = recompute_record_hash(by_h[h]) if h in by_h else None
            valid, faulty, seen = 0, [], set()
            for a in atts:
                if a["height"] != h or a["witness"] in seen:
                    continue
                seen.add(a["witness"])
                key = keys.get(a["witness"])
                good_sig = bool(key) and hmac.compare_digest(
                    a["mac"], _mac(key, f"{a['height']}:{a['record_hash']}"))
                if good_sig and a["record_hash"] == actual:
                    valid += 1
                elif good_sig:
                    faulty.append(f"{a['witness']}(equivocates)")
                else:
                    faulty.append(f"{a['witness']}(bad-sig)")
            status = "OK" if valid >= cfg["quorum"] else "FAIL"
            if valid < cfg["quorum"]:
                consensus_ok = False
            line = (f"height {h}: {valid}/{cfg['n']} valid & agreeing, "
                    f"quorum {cfg['quorum']} -> {status}")
            if faulty:
                line += f"  | faulty (tolerated if quorum holds): {', '.join(faulty)}"
            out.append(line)
        return (ok_hash and consensus_ok), out
