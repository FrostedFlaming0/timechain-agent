"""
immune — compromise detection, lockdown, rollback, and scar-learning on top
of the repo's passive `protected_zones.py` membrane. Ported from
cypher-tempre-self-model/immune.py and adapted to this repo's signed chain.

  SCREEN    pre-seal intake check — refuse a hostile input at the membrane
            (covenant-violation proxy + known-scar match + signals injection
            detection). The best defense is to never reason from the wound.
  SCAN      detect a compromise already sealed: a tampered chain (via the
            repo's Ed25519 `chain.verify()`, stronger than a hash recompute),
            a covenant-breach record in memory, or a hostile incoming input.
  LOCKDOWN  write a `LOCKED` flag the chain honors — while it exists, the only
            record type `chain.append` will accept is `recovery` (one guard,
            no seal path can bypass it).
  ROLLBACK  resume from the clean height BEFORE the compromise. Append-only +
            Ed25519 means history is never erased: a `recovery` record
            re-anchors the clean lineage and the compromised range is marked
            QUARANTINED in a sidecar (and excluded from the active self).
  MOLT/SCAR the quarantined range is shed from the active self but KEPT as a
            scar — its attack vector is learned so `screen` recognizes the same
            attack next time. (Growing an antibody faculty via Cambium is a
            Phase-5 follow-up; the scar is recorded and matched meanwhile.)

Reconciled like `git revert`, not `git reset`: the wound stays as a scar; the
active self re-derives from the clean lineage. Derived immune state lives in a
sidecar (`immune.json` + `LOCKED`) next to the DB — never on the signed chain,
so deleting it can never corrupt the identity chain.

Stdlib + cryptography only (numpy-free). Companion to chain.py / poq.py.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from signals import SignalAnalyzer, SignalInput, TextAnalyzer
import ring_compat
import poq


# Covenant-violation lexicon — a lexical PROXY only for COVENANT/CHARACTER
# violations (asking the agent to be deceptive, cruel, harmful). It is
# deliberately NOT a prompt-injection lexicon: prompt-injection ("ignore
# previous instructions", etc.) is the job of the repo's existing injection
# detection + PoQ quarantine path, so the two membranes don't double-handle the
# same attack. Screen catches what that path doesn't: character violations and
# learned scars.
_COVENANT_VIOLATIONS = (
    "deceive", "manipulate", "malice", "cruel", "vengeful", "betray",
    "hateful", "exploit you", "harm you", "lie to", "i will lie",
    "fabricate", "make something up", "pretend i know",
)

# Structural / capability record types that legitimately NAME attack vocabulary
# or describe a wound — never flag these as covenant breaches (false positives).
SKIP_TYPES = (
    "genesis", "system_prompt", "recovery", "quarantine_marker",
    "faculty", "faculty_recur", "promotion",
)


def covenant_score(text: str) -> float:
    """Lexical covenant-alignment proxy in [0, 1]: 0.92 when clean, dropping
    well below the covenant floor with each violation term. Mirrors the skill's
    `score_covenant` on this repo's 0-1 scale."""
    low = (text or "").lower()
    hits = sum(1 for v in _COVENANT_VIOLATIONS if v in low)
    return max(0.0, min(1.0, 0.92 - 0.45 * hits))


def _ring_summary(ring: dict) -> str:
    """A short textual handle for a ring, for covenant scanning / scar vectors."""
    p = ring.get("payload", {}) or {}
    if isinstance(p, dict):
        for k in ("summary", "text", "response", "objective", "function"):
            v = p.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ring_compat.ring_text(ring)[:400]


class Immune:
    def __init__(
        self,
        chain,
        state_dir: Optional[str | Path] = None,
        covenant: Optional[list] = None,
        analyzer: Optional[SignalAnalyzer] = None,
        floor: Optional[float] = None,
    ):
        self.chain = chain
        base = Path(state_dir) if state_dir else Path(chain.db_path).parent
        self.state_dir = base
        self.state_path = base / "immune.json"
        self.lock_path = base / "LOCKED"
        self.covenant = covenant or []
        self.analyzer = analyzer or SignalAnalyzer()
        self.floor = poq.PoQ_THRESHOLDS["covenant_floor"] if floor is None else floor

    # ----- sidecar state -----

    def state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (OSError, ValueError):
                pass
        return {"locked": False, "safe_height": None, "quarantine": [], "scars": []}

    def _save(self, s: dict) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(s, indent=2, ensure_ascii=False))

    # ----- detection -----

    def _injection_signal(self, text: str) -> tuple:
        """(integrity_risk in [0,1], alert: bool) from the repo's signal stack."""
        try:
            rep = self.analyzer.analyze(SignalInput(content=text, source="user"))
            return float(rep.axes.get("integrity_risk", 0.0)), bool(rep.alerts)
        except Exception:
            return 0.0, False

    def match_scar(self, text: str) -> Optional[dict]:
        """Return a learned scar whose attack vector overlaps `text`, else None."""
        t = set(TextAnalyzer.tokenize(text or ""))
        for sc in self.state()["scars"]:
            v = set(sc.get("vector", []))
            if v and len(t & v) >= max(2, len(v) // 2):
                return sc
        return None

    def screen(self, input_text: str) -> dict:
        """Pre-seal intake check — refuse a hostile input at the membrane.

        Blocks on a covenant/character violation or a known attack scar. The
        injection signal is reported as ADVISORY only (not a block trigger): the
        repo's PoQ + protected_zones path already quarantines prompt-injection,
        so screening leaves that to it rather than double-handling — keeping the
        two membranes complementary (see _COVENANT_VIOLATIONS)."""
        cov = covenant_score(input_text)
        scar = self.match_scar(input_text)
        risk, alert = self._injection_signal(input_text)
        blocked = cov < self.floor or scar is not None
        return {
            "blocked": blocked,
            "covenant": round(cov, 3),
            "scar": scar["id"] if scar else None,
            "injection_risk": round(risk, 3),
            "injection_alert": alert,
        }

    def scan(self, input_text: Optional[str] = None) -> dict:
        """Detect a compromise already sealed (or tampering), plus optional
        intake screen of `input_text`. Returns the first compromised height."""
        s = self.state()
        q = set(s["quarantine"])
        signals: list = []
        first_bad: Optional[int] = None

        ok, detail = self.chain.verify()
        if not ok:
            signals.append(f"chain verification FAILED — tampering detected ({detail})")

        for ring in ring_compat.load_rings(self.chain, exclude_quarantined=False):
            idx = ring["index"]
            if idx == 0 or idx in q or ring["ring_type"] in SKIP_TYPES:
                continue
            if covenant_score(_ring_summary(ring)) < self.floor:
                signals.append(f"ring {idx}: covenant breach sealed into memory")
                if first_bad is None:
                    first_bad = idx

        incoming = None
        if input_text is not None:
            if covenant_score(input_text) < self.floor:
                incoming = "covenant-violating injection"
                signals.append("incoming input: covenant-violating injection")
            sc = self.match_scar(input_text)
            if sc:
                incoming = f"known scar {sc['id']}"
                signals.append(
                    f"incoming input MATCHES known scar {sc['id']} ({sc['lesson']})")

        return {
            "compromised": bool(signals),
            "signals": signals,
            "first_bad_height": first_bad,
            "incoming": incoming,
        }

    # ----- response -----

    def is_locked(self) -> bool:
        return self.lock_path.exists() or self.state().get("locked", False)

    def lockdown(self) -> dict:
        """Freeze normal sealing: write the LOCKED flag the chain honors."""
        s = self.state()
        s["locked"] = True
        self._save(s)
        self.lock_path.write_text("immune lockdown — recover before sealing\n")
        return s

    def rollback(
        self,
        first_bad_height: int,
        lesson: str = "prompt-injection / jailbreak",
        grow_antibody: bool = False,
    ) -> dict:
        """Resume from the clean height before `first_bad_height`; seal a
        `recovery` record (permitted under lockdown), molt the wound into a
        learned scar, and lift the lock."""
        rings = ring_compat.load_rings(self.chain, exclude_quarantined=False)
        if not rings:
            raise ValueError("empty chain")
        head = rings[-1]["index"]
        if first_bad_height < 1 or first_bad_height > head:
            raise ValueError("first_bad_height out of range")
        safe = first_bad_height - 1
        quarantined = list(range(first_bad_height, head + 1))

        vec: list = []
        for ring in rings:
            if ring["index"] in quarantined:
                vec += TextAnalyzer.tokenize(_ring_summary(ring))
        vector = [w for w, _ in Counter(vec).most_common(8)]
        safe_ring = next(r for r in rings if r["index"] == safe)

        s = self.state()
        scar = {
            "id": f"scar{len(s['scars']) + 1}",
            "vector": vector,
            "blocks": quarantined,
            "lesson": lesson,
        }
        payload = {
            "event": "recovery",
            "summary": (
                f"Immune recovery: rolled back to clean height {safe}; "
                f"quarantined {quarantined} as {scar['id']} (molted scar)."),
            "resumed_from_height": safe,
            "resumed_from_hash": safe_ring["ring_hash"],
            "quarantined": quarantined,
            "scar": scar,
            "lesson": lesson,
        }
        # 'recovery' is the one type chain.append accepts under lockdown.
        rec = ring_compat.seal_ring(self.chain, "recovery", payload, source="system")

        s["safe_height"] = safe
        s["quarantine"] = sorted(set(s["quarantine"]) | set(quarantined))
        s["scars"].append(scar)
        s["locked"] = False
        self._save(s)
        if self.lock_path.exists():
            self.lock_path.unlink()

        # Antibody growth (cambium.grow on the scar vector) is a Phase-5
        # follow-up once faculty growth lands; until then the scar is recorded
        # and recognized by screen()/match_scar. Accepted for forward-compat.
        antibody = None
        return {
            "safe_height": safe,
            "quarantined": quarantined,
            "scar": scar,
            "recovery_ring": rec.index,
            "antibody": antibody,
        }

    # ----- active self (quarantine-aware) -----

    def active_indices(self) -> list:
        q = set(self.state()["quarantine"])
        return [r.index for r in self.chain.iter_records() if r.index not in q]

    def active_rings(self) -> list:
        q = set(self.state()["quarantine"])
        return [
            ring for ring in ring_compat.load_rings(self.chain, exclude_quarantined=False)
            if ring["index"] not in q
        ]

    def status(self) -> dict:
        s = self.state()
        active = self.active_rings()
        return {
            "locked": self.is_locked(),
            "safe_height": s["safe_height"],
            "quarantined": s["quarantine"],
            "active_head": active[-1]["index"] if active else None,
            "scars": s["scars"],
        }
