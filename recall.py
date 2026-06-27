"""
recall — self-labeling + relevance-realization (index -> fetch) over the chain.
Ported from cypher-tempre-self-model/recall.py, re-based on this repo's
`signals.py` faculty stack and `Chain` (via `ring_compat`).

As the chain (or a Continuum task chain) outgrows the context window, the agent
cannot reread everything. Recall gives it a queryable, verifiable map:

  SELF-LABEL  `label(content)` runs content through the repo's SignalAnalyzer;
              the senses/modalities that FIRE become the block's labels, with
              salient keywords, identifier-like entities, and a salience score.
              Continuum seals these at ingest, so recall reads them instantly.
  INDEX       `index()` renders the compact MAP OF MEMORY — one line per record
              (idx | type | summary | keywords | entities | senses/modalities).
              It is mostly a *renderer* over labels the chain already holds.
  FETCH       `fetch(ids)` pulls the full content of the blocks the MODEL chose,
              budget-bounded. The MODEL is the relevance judge; the labels are
              the map it reads, never a string-match arbiter.
  RETRIEVE    `retrieve(query)` is ONLY a cheap pre-filter for chains too large
              to index in context. It delegates to the repo's numpy retriever
              and never decides relevance on its own.
  VERIFY      `verify_source(idx, repo)` re-checks a Continuum source block
              against the live file (reuses the Phase-1 source-verify logic).

The label/index/fetch core is numpy-free (signals + chain only). `retrieve`
imports the numpy retriever lazily, so this module loads in minimal envs.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from signals import SignalAnalyzer, SignalInput, TextAnalyzer
import ring_compat


ENTITY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _content_tokens(text: str) -> list:
    return [w for w in TextAnalyzer.tokenize(text) if w not in TextAnalyzer.STOPWORDS]


def entities(text: str, cap: int = 12) -> list:
    """Identifier-like tokens: snake_case, CamelCase, dotted, or digit-bearing."""
    ents = set()
    for w in ENTITY_RE.findall(text or ""):
        core = w.strip(".")
        if len(core) > 2 and (
            ("_" in core)
            or any(c.isupper() for c in core[1:])
            or ("." in core)
            or any(c.isdigit() for c in core)
        ):
            ents.add(core)
    return sorted(ents)[:cap]


def keywords(text: str, k: int = 10) -> list:
    return [w for w, _ in Counter(_content_tokens(text)).most_common(k)]


def _strings(obj) -> list:
    out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for key, v in obj.items():
            if key in ("labels", "state", "_meta", "poq_verdict"):
                continue
            out += _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _strings(v)
    return out


def block_text(ring: dict) -> str:
    """The block's DISTINCTIVE content. For a Continuum block that's the source
    chunk (`data.content`); for everything else it's the payload text minus the
    repeating boilerplate (labels / rolling state / metadata)."""
    payload = ring.get("payload", {})
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict) and "content" in data:
        return str(data.get("content") or "")
    return " ".join(_strings(payload))


def excerpt(text: str, words: int = 24) -> str:
    parts = (text or "").split()
    return " ".join(parts[:words])


def _names(items) -> list:
    """Normalize a labels list that may hold strings or {'name': ...} dicts."""
    out = []
    for it in items or []:
        if isinstance(it, dict):
            name = it.get("name")
            if name:
                out.append(name)
        elif isinstance(it, str):
            out.append(it)
    return out


class Recall:
    def __init__(self, chain, analyzer: Optional[SignalAnalyzer] = None,
                 retriever=None):
        self.chain = chain
        self.analyzer = analyzer or SignalAnalyzer()
        # Optional repo Retriever for the cheap pre-filter path (retrieve()).
        # None keeps the dependency-free index()/fetch() path fully usable.
        self.retriever = retriever

    # ----- self-labeling -----

    def label(self, content: str, context: str = "") -> dict:
        """Self-label content: which senses/modalities fire (via signals), plus
        salient keywords, identifier entities, and a salience score in [0, 1]."""
        try:
            rep = self.analyzer.analyze(SignalInput(content=content, source="assistant"))
            senses = list(rep.activated_senses())[:5]
            mods = list(rep.activated_modalities())[:5]
        except Exception:
            senses, mods = [], []
        ents = entities(content)
        toks = TextAnalyzer.tokenize(content)
        variety = len(set(toks)) / max(len(toks), 1)
        salience = max(0.0, min(1.0, 0.2 + 0.04 * len(ents) + 0.4 * variety))
        return {
            "senses": senses,
            "modalities": mods,
            "keywords": keywords(content),
            "entities": ents,
            "salience": round(salience, 3),
        }

    def block_labels(self, ring: dict) -> dict:
        """Prefer labels sealed into the block (Continuum) or the record's
        `_meta` activation lists; only recompute if neither is present."""
        payload = ring.get("payload", {}) if isinstance(ring, dict) else {}
        lab = payload.get("labels")
        if isinstance(lab, dict) and lab:
            return lab
        meta = payload.get("_meta", {}) if isinstance(payload, dict) else {}
        senses = meta.get("senses_activated") if isinstance(meta, dict) else None
        mods = meta.get("modalities_activated") if isinstance(meta, dict) else None
        if senses or mods:
            text = block_text(ring)
            return {
                "senses": _names(senses), "modalities": _names(mods),
                "keywords": keywords(text, 6), "entities": entities(text),
                "salience": None,
            }
        return self.label(block_text(ring))

    # ----- index -> fetch (the model is the relevance judge) -----

    def index(self, limit: Optional[int] = None) -> list:
        """The compact MAP OF MEMORY: one entry per record with its handles."""
        rings = ring_compat.load_rings(self.chain, exclude_quarantined=False)
        out = []
        for r in rings:
            lab = self.block_labels(r)
            out.append({
                "index": r["index"],
                "type": r["ring_type"],
                "summary": excerpt(block_text(r), 24)[:240],
                "keywords": (lab.get("keywords") or [])[:6],
                "entities": (lab.get("entities") or [])[:6],
                "senses": _names(lab.get("senses"))[:3],
                "modalities": _names(lab.get("modalities"))[:3],
                "salience": lab.get("salience"),
            })
        if limit:
            out = out[-limit:]
        return out

    def fetch(self, indices, budget_tokens: int = 2000) -> list:
        """Pull full content for the chosen indices, budget-bounded by approx
        tokens. The last block is truncated rather than dropped so the budget is
        a soft ceiling, not a silent cap."""
        out = []
        spent = 0
        for idx in indices:
            rec = self._get(idx)
            if rec is None:
                continue
            text = block_text(ring_compat.record_to_ring(rec))
            t = approx_tokens(text)
            if out and spent + t > budget_tokens:
                remaining = max(0, budget_tokens - spent)
                if remaining > 0:
                    out.append({"index": idx, "type": rec.type,
                                "content": text[:remaining * 4] + " …[truncated]",
                                "truncated": True})
                break
            spent += t
            out.append({"index": idx, "type": rec.type, "content": text,
                        "truncated": False})
        return out

    def _get(self, index: int):
        getter = getattr(self.chain, "get", None)
        if callable(getter):
            return getter(index)
        return next(iter(self.chain.iter_records(start=index, end=index + 1)), None)

    # ----- cheap pre-filter (delegates to the numpy retriever) -----

    def retrieve(self, query: str, k: int = 8, n_recent: int = 10) -> list:
        """A cheap PRE-FILTER only — narrows candidates for chains too large to
        index in context; never the arbiter of relevance. Delegates to the
        repo's `retrieval.Retriever` (passed at construction), then returns
        compact briefs the MODEL judges via index()/fetch(). Quarantined records
        are filtered out, matching the agent's own retrieval path."""
        if self.retriever is None:
            raise RuntimeError(
                "recall.retrieve needs a Retriever — construct "
                "Recall(chain, retriever=<Retriever>). The dependency-free "
                "index()/fetch() model-as-judge path needs no retriever."
            )
        records = self.retriever.build_context(
            query=query, k_semantic=k, n_recent=n_recent)
        import protected_zones
        records = protected_zones.filter_quarantined(records)
        out = []
        for r in records:
            ring = ring_compat.record_to_ring(r)
            out.append({
                "index": r.index,
                "type": r.type,
                "excerpt": excerpt(block_text(ring), 30)[:200],
            })
        return out

    # ----- source verification for Continuum blocks (reuses Phase-1 A) -----

    def verify_source(self, ring_index: int, repo: Optional[str | Path] = None) -> dict:
        """Re-check a Continuum source block against the live file. Mirrors
        source_verify.verify_file_record but keys off the block's `data`
        coordinates (relative_path + file_content_hash + git_commit)."""
        import source_verify
        rec = self._get(ring_index)
        if rec is None:
            return {"ring_index": ring_index, "verdict": "missing-ring"}
        ring = ring_compat.record_to_ring(rec)
        data = ring.get("payload", {}).get("data") or {}
        rel = data.get("relative_path") or data.get("item")
        stored = data.get("file_content_hash")
        result = {"ring_index": ring_index, "relative_path": rel}
        if not rel:
            result["verdict"] = "no-source-path"
            return result
        live = Path(rel)
        if not live.is_absolute() and repo is not None:
            live = Path(repo) / rel
        if not live.is_file():
            result["verdict"] = "missing-source-file"
            result["source_path"] = str(live)
            return result
        live_sha = hashlib.sha256(live.read_bytes()).hexdigest()
        # Continuum stores the SHA of the (possibly redacted) full file text via
        # sha256_text; compare against the raw-bytes sha only when they match
        # form. We compare the stored file_content_hash to a re-hash of the live
        # file's text to detect drift.
        live_text_sha = hashlib.sha256(
            live.read_text(errors="replace").encode("utf-8")).hexdigest()
        result["source_path"] = str(live)
        result["stored_file_hash"] = stored
        result["live_file_hash"] = live_text_sha
        if stored and live_text_sha != stored and live_sha != stored:
            result["verdict"] = "source-mismatch"
            return result
        stored_commit = data.get("git_commit")
        if stored_commit:
            git = source_verify.current_git_info(live)
            result["stored_git_commit"] = stored_commit
            result["live_git_commit"] = git.get("commit")
            if git.get("dirty"):
                result["verdict"] = "dirty-worktree"
                return result
            if git.get("commit") and git["commit"] != stored_commit:
                result["verdict"] = "revision-drift"
                return result
        result["verdict"] = "verified"
        return result
