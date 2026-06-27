"""
audit — compute a read-only audit snapshot of a chain for the web dashboard.

Returns a single JSON-able dict: top-line metrics, domain context (which subject
areas the chain has accumulated), the faculty surface (modality/sense categories),
a ring list, and the blockspace (content-addressed blobs). Pure read; never
mutates the chain. Numpy-free (uses signals' tokenizer + ring_compat only).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from signals import TextAnalyzer
import ring_compat
import recall


# Domain taxonomy: name -> trigger keywords. A record "hits" a domain when its
# text contains any of these words. Mirrors the audit dashboard's Domain Context
# — a coarse, deterministic read of what subject areas the chain has touched.
DOMAINS = {
    "Code & Software": {"code", "function", "class", "commit", "git", "repo",
                        "test", "source", "module", "refactor", "bug", "api",
                        "import", "compile"},
    "Timechain & Self-Model": {"timechain", "ring", "chain", "continuum", "poq",
                               "chronosynaptic", "sense", "senses", "modality",
                               "modalities", "faculty", "immune", "recall",
                               "genesis", "covenant"},
    "Security & Audit": {"audit", "verify", "verification", "risk", "consensus",
                         "tamper", "compliance", "security", "signature",
                         "attack", "injection", "quarantine", "scar"},
    "Blockchain & Web3": {"hash", "block", "bitcoin", "wallet", "transaction",
                          "merkle", "ledger", "ed25519"},
    "Data Science & Statistics": {"data", "json", "model", "dataset",
                                  "distribution", "regression", "quantile",
                                  "csv", "vector", "embedding", "embeddings"},
    "Documents & Writing": {"summary", "report", "document", "note", "draft",
                            "readme", "changelog", "docs"},
    "Operations & Packaging": {"selftest", "version", "install", "deploy",
                               "release", "build", "package", "pip", "pytest"},
    "Research & Knowledge": {"source", "finding", "evidence", "analysis",
                             "research", "paper", "hypothesis", "study"},
    "Finance & Markets": {"token", "nasdaq", "drawdown", "return", "trading",
                          "cagr", "volume", "price", "market", "portfolio"},
    "Biology & Genomics": {"gene", "dna", "entropy", "protein", "genome", "cell"},
}


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _content_tokens(text: str) -> list:
    return [w for w in TextAnalyzer.tokenize(text or "")
            if w not in TextAnalyzer.STOPWORDS and len(w) > 1]


def _load(path: Path, key: str) -> list:
    try:
        return json.loads(Path(path).read_text()).get(key, [])
    except Exception:
        return []


def _categories(items: list) -> list:
    total = len(items)
    counts = Counter(it.get("category", "other") for it in items)
    return [{"category": cat, "count": n, "total": total}
            for cat, n in counts.most_common()]


def _blobs(blob_dir: Optional[Path], chain) -> list:
    if not blob_dir:
        return []
    bd = Path(blob_dir)
    if not bd.is_dir():
        return []
    names = {}
    for r in chain.iter_records():
        if r.type == "file" and isinstance(r.content, dict):
            sha = r.content.get("blob_sha256")
            if sha:
                names[sha] = r.content.get("filename") or sha
    out = []
    for p in sorted(bd.iterdir()):
        if p.is_file():
            out.append({"name": names.get(p.name, p.name[:16] + "…"),
                        "hash": p.name, "size": p.stat().st_size})
    return out


def compute(chain, faculty_dir, blob_dir=None, integrity: Optional[bool] = None,
            ring_limit: int = 250) -> dict:
    """Build the audit snapshot dict. `integrity` is the result of a prior
    chain.verify() (pass it in so the endpoint doesn't verify twice)."""
    records = list(chain.iter_records())
    texts = {r.index: recall.block_text(ring_compat.record_to_ring(r))
             for r in records}
    rtoks = {idx: _content_tokens(t) for idx, t in texts.items()}

    fdir = Path(faculty_dir)
    mods = _load(fdir / "modalities.json", "modalities")
    senses = _load(fdir / "senses.json", "senses")
    emergent = _load(fdir / "emergent.json", "faculties")

    # Domain context.
    domains = []
    for name, kws in DOMAINS.items():
        hit_rings = 0
        toks = 0
        kw_counter: Counter = Counter()
        for r in records:
            hits = [w for w in rtoks[r.index] if w in kws]
            if hits:
                hit_rings += 1
                toks += approx_tokens(texts[r.index])
                kw_counter.update(hits)
        if hit_rings:
            domains.append({
                "name": name,
                "rings": hit_rings,
                "est_tokens": toks,
                "keywords": [{"word": w, "count": c}
                             for w, c in kw_counter.most_common(8)],
            })
    domains.sort(key=lambda d: d["rings"], reverse=True)
    max_rings = max((d["rings"] for d in domains), default=1)
    for d in domains:
        d["bar"] = round(d["rings"] / max_rings, 3) if max_rings else 0.0

    # Ring list (newest first, bounded).
    ring_list = []
    for r in records:
        content = r.content if isinstance(r.content, dict) else {}
        meta = content.get("_meta", {}) if isinstance(content, dict) else {}
        poq = meta.get("poq") if isinstance(meta, dict) else None
        brightness = poq.get("brightness") if isinstance(poq, dict) else None
        summary = texts[r.index].strip().replace("\n", " ")
        ring_list.append({
            "index": r.index,
            "type": r.type,
            "brightness": brightness,
            "summary": summary[:260],
            "keywords": [w for w, _ in Counter(rtoks[r.index]).most_common(8)],
        })
    ring_list = list(reversed(ring_list[-ring_limit:]))

    return {
        "metrics": {
            "rings": len(records),
            "modalities": len(mods),
            "senses": len(senses),
            "blockspace": len(_blobs(blob_dir, chain)),
            "context_tokens": sum(approx_tokens(t) for t in texts.values()),
            "integrity": ("PASS" if integrity else "FAIL") if integrity is not None else "—",
        },
        "domains": domains,
        "faculties": {
            "modalities_total": len(mods),
            "senses_total": len(senses),
            "emergent": len(emergent),
            "modalities": _categories(mods),
            "senses": _categories(senses),
        },
        "rings": ring_list,
        "blockspace": _blobs(blob_dir, chain),
    }
