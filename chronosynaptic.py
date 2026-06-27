"""
chronosynaptic — single-pass parallel-self MCTS, sealed into the chain. Ported
from cypher-tempre-self-model/chronosynaptic.py onto this repo's substrate.

NOT a subagent fan-out. One in-process reasoning pass forks many *perspectives
of itself* — each a lens drawn from the repo's own faculty registry
(`signals.MODALITY_REGISTRY` / `SENSE_REGISTRY`) — and runs MCTS over them:

    SELECT   descend the tree of perspective-paths by UCT.
    EXPAND   adopt an untried perspective -> a new candidate stance.
    SIMULATE roll out that perspective's FUTURE (greedy continuations) to
             estimate the highest truth reachable from it.
    BACKPROP flow the value up the path.

Every node is scored by the repo's PoQ gate (`poq.PoQEvaluator`) against unified
data: PAST (grounding/continuity vs the chain), TRAINING (the model's own
judgment via the `external_scores` seam), and FUTURE (the rollout values).

COLLAPSE: the single highest-truth path is sealed as one `synthesis` record;
the rejected forks are recorded in its payload (the chain witnesses the collapse)
but are not sealed — they fall away.

EXPLICIT NOTES MODE (preferred for audits): the model does the semantic work —
perspective summaries + scores — and asks this module to collapse them. The
winner is sealed; every rejected perspective is preserved in the same payload.

The MCTS (Node, select, _uct, expand, simulate, backprop, search, best_path) is
storage-independent and ported verbatim; only the faculty source, the PoQ
valuation, and sealing are re-pointed at the repo. Numpy-free (signals + poq +
chain only).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from signals import (
    MODALITY_REGISTRY, SENSE_REGISTRY,
    _MODALITY_ROUTING_TRIGGERS, _SENSE_ROUTING_TRIGGERS, TextAnalyzer,
)
from poq import PoQEvaluator
import ring_compat


def _detector_name(fn) -> str:
    """A detector's name, matching signals.py's convention: a sprouted
    detector's `.modality_name`, else `__name__` with the `m_`/`s_` prefix
    stripped (`m_intent` -> `intent`)."""
    nm = getattr(fn, "modality_name", None)
    if isinstance(nm, str) and nm:
        return nm
    raw = getattr(fn, "__name__", "")
    if raw.startswith(("m_", "s_")):
        return raw[2:]
    return raw


def _toks(text: str) -> set:
    return set(TextAnalyzer.tokenize(text or ""))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def short(s: str, n: int = 48) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def load_faculties() -> list:
    """Build the perspective pool from the repo's signal detectors. Each
    detector contributes a faculty-lens with its name + routing-trigger words as
    the overlap tokens."""
    fac = []
    for kind, registry, triggers in (
        ("modality", MODALITY_REGISTRY, _MODALITY_ROUTING_TRIGGERS),
        ("sense", SENSE_REGISTRY, _SENSE_ROUTING_TRIGGERS),
    ):
        for fn in registry:
            name = _detector_name(fn)
            trig = list(triggers.get(name) or [])
            toks = _toks(name.replace("_", " ")) | set(trig)
            fac.append({
                "kind": kind, "id": name, "name": name,
                "function": " ".join(sorted(trig)) or name,
                "category": kind, "tokens": toks,
            })
    return fac


def frame(perspective: dict, query: str) -> str:
    """The stance contributed by adopting one perspective (a faculty-lens)."""
    foc = ", ".join(sorted(_toks(query) & perspective["tokens"])[:4]) or perspective["category"]
    return (f"[{perspective['name']}] {perspective['category']} reading "
            f"focusing on {foc} via {short(perspective['function'])}")


# --------------------------------------------------------------------------- #
# Explicit perspective-notes (the preferred, model-judged path)
# --------------------------------------------------------------------------- #

PRESERVED_NOTE_FIELDS = {
    "assumptions", "confidence", "evidence", "findings", "notes",
    "open_questions", "recommendations", "risks", "severity", "verdict",
}
KNOWN_NOTE_FIELDS = PRESERVED_NOTE_FIELDS | {
    "brightness", "chosen", "kind", "name", "score", "scores", "selected",
    "summary", "synthesis", "value",
}


def note_scores(note: dict, index: int) -> dict:
    """Pull a scores dict (any dimension names) or a scalar score/value/
    brightness from a perspective note. Scale-agnostic — the winner is chosen by
    relative value, so 0-1 or 0-255 both work."""
    raw = note.get("scores")
    if isinstance(raw, dict) and raw:
        scores = {k: float(v) for k, v in raw.items()
                  if isinstance(v, (int, float))}
        if scores:
            return scores
    scalar = note.get("score", note.get("value", note.get("brightness")))
    if scalar is not None:
        try:
            return {"score": float(scalar)}
        except (TypeError, ValueError):
            raise ValueError(f"perspective {index}: score must be a number")
    raise ValueError(f"perspective {index}: provide score/value/brightness or a scores object")


def normalize_perspective_note(note: dict, index: int) -> dict:
    if not isinstance(note, dict):
        raise ValueError(f"perspective {index}: expected an object")
    summary = (note.get("summary") or note.get("synthesis") or "").strip()
    if not summary:
        raise ValueError(f"perspective {index}: summary is required")
    scores = note_scores(note, index)
    value = round(sum(scores.values()) / len(scores), 3)
    out = {
        "index": index,
        "name": str(note.get("name") or f"Perspective {index}"),
        "kind": str(note.get("kind") or "explicit"),
        "summary": summary,
        "scores": scores,
        "value": value,
        "chosen_hint": bool(note.get("chosen") or note.get("selected")),
    }
    for field in sorted(PRESERVED_NOTE_FIELDS):
        if field in note:
            out[field] = note[field]
    details = {k: v for k, v in note.items() if k not in KNOWN_NOTE_FIELDS}
    if details:
        out["details"] = details
    return out


def public_perspective(perspective: dict, decision=None) -> dict:
    out = {k: v for k, v in perspective.items() if k != "chosen_hint"}
    if decision:
        out["decision"] = decision
    return out


def choose_explicit_perspective(perspectives: list, winner=None) -> dict:
    if winner:
        winner_s = str(winner).strip()
        matches = []
        if winner_s.isdigit():
            wanted = int(winner_s)
            matches = [p for p in perspectives if p["index"] == wanted]
        if not matches:
            wanted = winner_s.lower()
            matches = [p for p in perspectives if p["name"].lower() == wanted]
        if len(matches) != 1:
            raise ValueError(f"winner {winner!r} did not match exactly one perspective")
        return matches[0]
    hinted = [p for p in perspectives if p.get("chosen_hint")]
    if len(hinted) > 1:
        raise ValueError("multiple perspectives marked chosen/selected; pass winner=")
    if hinted:
        return hinted[0]
    return max(perspectives, key=lambda p: (p["value"], -p["index"]))


# --------------------------------------------------------------------------- #
# MCTS
# --------------------------------------------------------------------------- #

class Node:
    __slots__ = ("parent", "depth", "perspective", "path", "children",
                 "untried", "N", "W", "poq")

    def __init__(self, parent, depth, perspective, path, untried):
        self.parent = parent
        self.depth = depth
        self.perspective = perspective   # the faculty-lens adopted (None at root)
        self.path = path                 # list of perspectives root..this
        self.children = []
        self.untried = untried           # perspectives not yet expanded here
        self.N = 0
        self.W = 0.0
        self.poq = None                  # immediate PoQ verdict at this node

    def q(self) -> float:
        return self.W / self.N if self.N else 0.0


class ChronosynapticTree:
    def __init__(self, chain, iterations=16, forks=4, max_depth=2, c=1.2,
                 evaluator: Optional[PoQEvaluator] = None, faculties=None,
                 chain_text_cap=50):
        self.chain = chain
        self.evaluator = evaluator or PoQEvaluator()
        self.faculties = faculties if faculties is not None else load_faculties()
        self.iterations = iterations
        self.forks = forks
        self.max_depth = max_depth
        self.c = c
        rings = ring_compat.load_rings(chain, exclude_quarantined=False)
        self._chain_texts = [ring_compat.ring_text(r) for r in rings][-chain_text_cap:]
        self._covenant = None
        if rings and isinstance(rings[0].get("payload"), dict):
            cov = rings[0]["payload"].get("covenant")
            if isinstance(cov, list):
                self._covenant = cov

    # ---- perspective selection (relevance to the query, exploration via UCT) ----
    def rank(self, query, context, k, exclude_ids=()):
        q = _toks(f"{query} {context}")
        pool = [f for f in self.faculties if (f["kind"], f["id"]) not in exclude_ids]
        pool.sort(key=lambda f: len(q & f["tokens"]) + jaccard(q, f["tokens"]),
                  reverse=True)
        return pool[:k]

    def _used(self, path):
        return {(p["kind"], p["id"]) for p in path}

    # ---- PoQ valuation against unified data (past chain + training seam) ----
    def value(self, path, query, context, external=None):
        text = self.compose(path, query)
        res = self.evaluator.evaluate(
            query, text, retrieved_texts=self._chain_texts,
            covenant=self._covenant, external_scores=external)
        verdict = {
            "brightness": res.brightness,
            "decision": res.verdict,
            "scores": res.dimensions,
            "grounding": res.grounding,
        }
        return verdict, text

    def compose(self, path, query):
        return ("Synthesis of self-perspectives — "
                + " ; ".join(frame(p, query) for p in path))

    # ---- MCTS phases ----
    def select(self, root):
        node = root
        while True:
            if node.depth >= self.max_depth:
                return node
            if node.untried:
                return node
            if not node.children:
                return node
            node = max(node.children, key=lambda ch: self._uct(ch))

    def _uct(self, child):
        if child.N == 0:
            return float("inf")
        return child.q() + self.c * math.sqrt(math.log(child.parent.N) / child.N)

    def expand(self, node, query, context):
        p = node.untried.pop(0)
        path = node.path + [p]
        verdict, _ = self.value(path, query, context)
        nxt_depth = node.depth + 1
        untried = (self.rank(query, context, self.forks, self._used(path))
                   if nxt_depth < self.max_depth else [])
        child = Node(node, nxt_depth, p, path, untried)
        child.poq = verdict
        node.children.append(child)
        return child

    def simulate(self, node, query, context):
        """Roll out the FUTURE: greedily extend to max_depth, choosing the
        continuation perspective with the highest PoQ brightness."""
        path = list(node.path)
        depth = node.depth
        while depth < self.max_depth:
            pool = self.rank(query, context, self.forks, self._used(path))
            if not pool:
                break
            best, best_b = None, -1.0
            for f in pool:
                verdict, _ = self.value(path + [f], query, context)
                if verdict["brightness"] > best_b:
                    best_b, best = verdict["brightness"], f
            path.append(best)
            depth += 1
        verdict, _ = self.value(path, query, context)
        return verdict["brightness"]   # already 0-1 on this repo's scale

    def backprop(self, node, value):
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent

    # ---- the single-pass search ----
    def search(self, query, context=""):
        root = Node(None, 0, None, [], self.rank(query, context, self.forks))
        for _ in range(self.iterations):
            node = self.select(root)
            if node.depth < self.max_depth and node.untried:
                node = self.expand(node, query, context)
            value = self.simulate(node, query, context)
            self.backprop(node, value)
        return root

    def best_path(self, root):
        node, chosen = root, []
        while node.children:
            node = max(node.children, key=lambda ch: (ch.N, ch.q()))
            chosen.append(node)
        return chosen

    def collapse_and_seal(self, root, query, context="", do_seal=True):
        chosen = self.best_path(root)
        if not chosen:
            return None, None
        leaf = chosen[-1]
        synthesis = self.compose(leaf.path, query)
        forks_report = sorted(
            [{"perspective": ch.perspective["name"], "kind": ch.perspective["kind"],
              "visits": ch.N, "value": round(ch.q(), 3)} for ch in root.children],
            key=lambda d: d["visits"], reverse=True)
        payload = {
            "event": "chronosynaptic_collapse",
            "query": query,
            "chosen_path": [p["name"] for p in leaf.path],
            "synthesis": synthesis,
            "considered_forks": forks_report,
            "collapsed_from": len(root.children),
            "sealed_one_of": len(root.children),
        }
        rec = None
        if do_seal:
            rec = ring_compat.seal_ring(
                self.chain, "synthesis", payload, source="assistant",
                poq=leaf.poq["scores"] if leaf.poq else None)
        return {"chosen": chosen, "leaf": leaf, "forks": forks_report,
                "synthesis": synthesis}, rec

    def collapse_explicit_notes(self, notes, query=None, context=None, winner=None,
                                do_seal=True):
        if isinstance(notes, list):
            top, raw_perspectives = {}, notes
        elif isinstance(notes, dict):
            top = notes
            raw_perspectives = notes.get("perspectives") or notes.get("forks")
        else:
            raise ValueError("notes must be an object or a list of perspectives")
        if not isinstance(raw_perspectives, list) or not raw_perspectives:
            raise ValueError("notes must contain a non-empty perspectives list")

        query = query or top.get("query")
        if not query:
            raise ValueError("query is required in notes or via query=")
        context = top.get("context", "") if context is None else context

        perspectives = [normalize_perspective_note(n, i)
                        for i, n in enumerate(raw_perspectives, start=1)]
        chosen = choose_explicit_perspective(perspectives, winner=winner)
        synthesis = top.get("synthesis") or chosen["summary"]
        rejected = [p for p in perspectives if p["index"] != chosen["index"]]
        forks_report = [
            {"perspective": p["name"], "kind": p["kind"], "value": p["value"],
             "decision": "sealed" if p["index"] == chosen["index"] else "rejected"}
            for p in sorted(perspectives, key=lambda i: i["value"], reverse=True)
        ]
        payload = {
            "event": "chronosynaptic_explicit_collapse",
            "mode": "explicit-perspective-notes",
            "query": query,
            "context": context,
            "chosen_path": [chosen["name"]],
            "chosen_perspective": public_perspective(chosen, decision="sealed"),
            "synthesis": synthesis,
            "considered_forks": forks_report,
            "perspectives": [
                public_perspective(p, decision="sealed" if p["index"] == chosen["index"] else "rejected")
                for p in perspectives
            ],
            "rejected_perspectives": [public_perspective(p, decision="rejected") for p in rejected],
            "collapsed_from": len(perspectives),
            "sealed_one_of": 1,
            "score_basis": "model-supplied explicit perspective notes",
        }
        for field in ("audit_id", "scope", "repo", "commit", "source"):
            if field in top:
                payload[field] = top[field]
        rec = None
        if do_seal:
            rec = ring_compat.seal_ring(self.chain, "synthesis", payload,
                                        source="assistant", poq=chosen["scores"])
        return {"chosen": chosen, "forks": forks_report, "rejected": rejected,
                "synthesis": synthesis, "payload": payload}, rec
