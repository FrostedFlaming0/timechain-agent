"""
faculties — data-faculty registries + endogenous growth (Cambium). Ported from
cypher-tempre-self-model/cambium.py and adapted to this repo.

Two kinds of faculty coexist (plan I):
  - CODE faculties  — the executable detectors in `signals.py` (real regex/
                      lexicon logic: injection scan, coherence, artifacts). Kept
                      as-is; they do work descriptive proxies cannot.
  - DATA faculties  — descriptive entries (`faculties/{modalities,senses}.json`)
                      used for relevance/overlap scoring: gap detection here,
                      perspective ranking in chronosynaptic, recall labeling.

`load_corpus` unifies both (data faculties + a bridge over the signals registry
names) so a query can match either kind.

GROWTH (Cambium, plan J): when an input's dissonance exceeds the floor, the
existing faculties don't cover it, so the agent grows a new one:
  DETECT dissonance -> PROPOSE (fuse two close faculties, else sprout from the
  uncovered terms) -> SPAWN into the emergent registry (`faculties/emergent.json`)
  -> seal a `faculty` record. On the 3rd recurrence of the same gap the faculty
  is PROMOTED into the canonical data registry and a `promotion` record is sealed.

The growth core (detect_gap / infer_kind / propose / match_emergent / faculty_poq)
is storage-independent and ports verbatim; only sealing/promotion writes are
re-pointed at `Chain` via `ring_compat`. Scores are kept 0-255 internally and
normalized to 0-1 only when written to `_meta.poq`. Numpy-free.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from signals import (
    MODALITY_REGISTRY, SENSE_REGISTRY,
    _MODALITY_ROUTING_TRIGGERS, _SENSE_ROUTING_TRIGGERS, TextAnalyzer,
)
import ring_compat


DISSONANCE_FLOOR = 150     # below this, existing faculties cover the input
SPROUT_DISSONANCE = 210    # at/above this the gap is too foreign to fuse -> sprout
PROMOTE_AT = 3             # recurrence count that triggers promotion to canonical
REASON_VERBS = {"analyze", "plan", "compute", "design", "solve", "debug",
                "optimize", "prove", "derive", "decide", "evaluate",
                "calculate", "reason", "refactor"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def short(text: str, n: int = 70) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def clamp(x) -> int:
    return int(max(0, min(255, round(x))))


def tokens(text: str) -> list:
    return [w for w in TextAnalyzer.tokenize(text or "")
            if w not in TextAnalyzer.STOPWORDS and len(w) > 1]


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def coverage(a: set, b: set) -> float:
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _detector_name(fn) -> str:
    nm = getattr(fn, "modality_name", None)
    if isinstance(nm, str) and nm:
        return nm
    raw = getattr(fn, "__name__", "")
    return raw[2:] if raw.startswith(("m_", "s_")) else raw


# --------------------------------------------------------------------------- #
# Faculty corpus (data faculties + signals bridge)
# --------------------------------------------------------------------------- #

def signals_corpus() -> list:
    """Bridge: present the repo's executable signal detectors as descriptive
    faculties so a query can match a code faculty as well as a data one."""
    out = []
    for kind, reg, trig in (
        ("modality", MODALITY_REGISTRY, _MODALITY_ROUTING_TRIGGERS),
        ("sense", SENSE_REGISTRY, _SENSE_ROUTING_TRIGGERS),
    ):
        for fn in reg:
            name = _detector_name(fn)
            t = list(trig.get(name) or [])
            out.append({
                "kind": kind, "id": f"sig:{name}", "name": name,
                "function": " ".join(t) or name, "category": "signal",
                "tokens": set(TextAnalyzer.tokenize(name.replace("_", " "))) | set(t),
            })
    return out


def load_data_corpus(faculty_dir: str | Path) -> list:
    out = []
    for kind, fname, key in (("modality", "modalities.json", "modalities"),
                             ("sense", "senses.json", "senses")):
        p = Path(faculty_dir) / fname
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for f in data.get(key, []):
            out.append({
                "kind": kind, "id": f"data:{f['id']}", "name": f["name"],
                "function": f["function"], "category": f.get("category", "general"),
                "tokens": set(tokens(f["name"] + " " + f["function"])),
            })
    return out


def load_corpus(faculty_dir: str | Path, include_signals: bool = True) -> list:
    corpus = load_data_corpus(faculty_dir)
    if include_signals:
        corpus += signals_corpus()
    return corpus


# --------------------------------------------------------------------------- #
# Gap detection + proposal (ported verbatim)
# --------------------------------------------------------------------------- #

def detect_gap(corpus: list, input_text: str, context: str = "") -> dict:
    toks = set(tokens(f"{input_text} {context}"))
    if not toks:
        return {"dissonance": 0, "coverage_ratio": 1.0, "uncovered": [],
                "top_activated": [], "_acts": [], "input_tokens": []}
    activations, covered = [], set()
    for f in corpus:
        inter = toks & f["tokens"]
        if inter:
            activations.append((len(inter), f))
            covered |= inter
    uncovered = sorted(toks - covered, key=lambda w: (-len(w), w))
    coverage_ratio = len(covered) / len(toks)
    dissonance = clamp((1 - coverage_ratio) * 255)
    activations.sort(key=lambda x: -x[0])
    top = [{"kind": f["kind"], "id": f["id"], "name": f["name"], "matched": n}
           for n, f in activations[:5]]
    return {"dissonance": dissonance, "coverage_ratio": round(coverage_ratio, 3),
            "uncovered": uncovered, "top_activated": top, "_acts": activations,
            "input_tokens": sorted(toks)}


def infer_kind(input_text: str) -> str:
    return "modality" if set(tokens(input_text)) & REASON_VERBS else "sense"


def propose(gap: dict, input_text: str, mode: str = "auto", kind_override=None) -> dict:
    acts = gap["_acts"]
    can_fuse = len(acts) >= 2 and acts[0][0] >= 2 and acts[1][0] >= 2
    do_fuse = ((mode == "fuse" and can_fuse)
               or (mode == "auto" and can_fuse and gap["dissonance"] < SPROUT_DISSONANCE))
    if do_fuse:
        a, b = acts[0][1], acts[1][1]
        kind = kind_override or ("sense" if a["kind"] == "sense" and b["kind"] == "sense"
                                 else "modality")
        return {
            "kind": kind,
            "name": f"{a['name']} × {b['name']} Fusion",
            "function": (f"Fused faculty applying {a['name']} ({short(a['function'], 40)}) "
                         f"together with {b['name']} ({short(b['function'], 40)}) when an "
                         f"input requires both at once."),
            "category": a["category"],
            "origin": f"fusion({a['id']}+{b['id']})",
            "parents": [a["id"], b["id"]],
            "seed_terms": [],
        }
    seed = [w for w in gap["uncovered"] if len(w) >= 4][:6] or gap["uncovered"][:6]
    kind = kind_override or infer_kind(input_text)
    label = "-".join(w.capitalize() for w in seed[:2]) if seed else "Novel"
    suffix = "Sensing" if kind == "sense" else "Reasoning"
    if kind == "sense":
        function = (f"Detect and tag the presence of {', '.join(seed)} in input — "
                    f"a perceptual gap the existing senses did not cover.")
        category = "structural"
    else:
        function = (f"Reason about and resolve problems involving {', '.join(seed)} — "
                    f"a reasoning gap the existing modalities did not cover.")
        category = "knowledge"
    return {"kind": kind, "name": f"{label} {suffix}", "function": function,
            "category": category, "origin": "sprout", "parents": [], "seed_terms": seed}


# --------------------------------------------------------------------------- #
# Emergent store + PoQ for faculty records
# --------------------------------------------------------------------------- #

def load_emergent(faculty_dir: str | Path) -> dict:
    p = Path(faculty_dir) / "emergent.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"registry": "emergent", "faculties": []}


def save_emergent(faculty_dir: str | Path, data: dict) -> None:
    Path(faculty_dir).mkdir(parents=True, exist_ok=True)
    (Path(faculty_dir) / "emergent.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False))


def match_emergent(data: dict, prop: dict):
    for e in data["faculties"]:
        if e["name"] == prop["name"]:
            return e
        if prop["parents"] and e.get("parents") == prop["parents"]:
            return e
        if (prop["seed_terms"] and e.get("seed_terms")
                and jaccard(set(prop["seed_terms"]), set(e["seed_terms"])) >= 0.5):
            return e
    return None


def faculty_poq_255(gap: dict, function: str) -> dict:
    return {
        "coherence": 205,
        "relevance": clamp(255 - gap["dissonance"]),
        "novelty": clamp(150 + gap["dissonance"] * 0.4),
        "consistency": 220,
        "depth": clamp(120 + len(set(tokens(function))) * 5),
        "covenant": 235,
    }


def _poq01(d255: dict) -> dict:
    """Normalize a 0-255 score dict to the repo's 0-1 _meta.poq scale."""
    return {k: round(v / 255.0, 4) for k, v in d255.items()}


# --------------------------------------------------------------------------- #
# The growth engine (seals re-pointed at Chain via ring_compat)
# --------------------------------------------------------------------------- #

class FacultyGarden:
    def __init__(self, chain, faculty_dir: str | Path, include_signals: bool = True):
        self.chain = chain
        self.dir = Path(faculty_dir)
        self.include_signals = include_signals

    def corpus(self) -> list:
        return load_corpus(self.dir, self.include_signals)

    def promote(self, e: dict, difficulty: int = 0):
        fname = "modalities.json" if e["kind"] == "modality" else "senses.json"
        key = "modalities" if e["kind"] == "modality" else "senses"
        p = self.dir / fname
        data = json.loads(p.read_text()) if p.exists() else {key: []}
        new_id = (max((it["id"] for it in data[key]), default=0) + 1)
        data[key].append({
            "id": new_id, "name": e["name"],
            "function": e["function"], "category": e["category"],
        })
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        e["promoted_to_id"] = new_id
        payload = {
            "event": "faculty_promotion", "emergent": e["eid"], "name": e["name"],
            "kind": e["kind"], "promoted_to_id": new_id,
            "recurrence": e["recurrence"], "registry": fname,
        }
        poq = _poq01({"coherence": 210, "relevance": 205, "novelty": 175,
                      "consistency": 220, "depth": 205, "covenant": 255})
        return ring_compat.seal_ring(self.chain, "promotion", payload,
                                     source="assistant", poq=poq, difficulty=difficulty)

    def grow(self, input_text: str, context: str = "", mode: str = "auto",
             kind_override=None, difficulty: int = 0):
        corpus = self.corpus()
        gap = detect_gap(corpus, input_text, context)
        result = {"gap": gap, "grew": False}
        if gap["dissonance"] <= DISSONANCE_FLOOR:
            result["action"] = "covered"
            result["reason"] = (f"dissonance {gap['dissonance']} <= floor "
                                f"{DISSONANCE_FLOOR}: existing faculties cover this; no growth.")
            return result, None

        prop = propose(gap, input_text, mode=mode, kind_override=kind_override)
        data = load_emergent(self.dir)
        existing = match_emergent(data, prop)

        if existing:
            existing["recurrence"] += 1
            existing.setdefault("history", []).append(
                {"ts": now_iso(), "dissonance": gap["dissonance"],
                 "context": short(input_text, 120)})
            if existing["recurrence"] >= PROMOTE_AT and existing["status"] == "emergent":
                rec = self.promote(existing, difficulty=difficulty)
                existing["status"] = "promoted"
                save_emergent(self.dir, data)
                result.update(grew=True, action="promoted", faculty=existing)
                return result, rec
            save_emergent(self.dir, data)
            payload = {"event": "faculty_recurrence", "emergent": existing["eid"],
                       "name": existing["name"], "recurrence": existing["recurrence"],
                       "dissonance": gap["dissonance"], "trigger": short(input_text, 200)}
            rec = ring_compat.seal_ring(
                self.chain, "faculty_recur", payload, source="assistant",
                poq=_poq01(faculty_poq_255(gap, existing["function"])),
                difficulty=difficulty)
            result.update(grew=True, action="recurrence", faculty=existing)
            return result, rec

        eid = f"E{len(data['faculties']) + 1}"
        fac = {"eid": eid, "kind": prop["kind"], "name": prop["name"],
               "function": prop["function"], "category": prop["category"],
               "origin": prop["origin"], "parents": prop["parents"],
               "seed_terms": prop["seed_terms"], "status": "emergent",
               "recurrence": 1, "born_at": now_iso(), "promoted_to_id": None,
               "history": [{"ts": now_iso(), "dissonance": gap["dissonance"],
                            "context": short(input_text, 120)}]}
        payload = {"event": "faculty_birth", "emergent": eid, "kind": fac["kind"],
                   "name": fac["name"], "function": fac["function"],
                   "category": fac["category"], "origin": fac["origin"],
                   "parents": fac["parents"], "seed_terms": fac["seed_terms"],
                   "dissonance": gap["dissonance"], "trigger": short(input_text, 200)}
        rec = ring_compat.seal_ring(
            self.chain, "faculty", payload, source="assistant",
            poq=_poq01(faculty_poq_255(gap, fac["function"])), difficulty=difficulty)
        fac["born_ring"] = rec.record_hash
        data["faculties"].append(fac)
        save_emergent(self.dir, data)
        result.update(grew=True, action="born", faculty=fac)
        return result, rec
