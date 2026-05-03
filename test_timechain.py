"""
test_timechain — pytest suite for the timechain agent.

Run with:
    pip install pytest
    pytest test_timechain.py -v

Tests are grouped by concern:
  - Chain integrity: signing, hash linkage, tamper detection
  - Merkle batching and inclusion proofs
  - Retrieval: semantic search, salience, ancestry
  - Agent behavior: turns, reflection, revision, drift detection
  - Time formatting: humanize_delta correctness
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

try:
    import pytest
except ImportError:
    # Allow the test module to import without pytest, so a standalone runner
    # can introspect the test classes. The actual test suite still needs
    # pytest to run normally.
    class _PytestStub:
        class fixture:
            def __init__(self, *a, **kw): pass
            def __call__(self, fn): return fn
        class mark:
            @staticmethod
            def parametrize(*a, **kw):
                def decorator(fn): return fn
                return decorator
        @staticmethod
        def raises(exc, match=None):
            class _Ctx:
                def __enter__(self): return self
                def __exit__(self, et, ev, tb):
                    return et is not None and issubclass(et, exc)
            return _Ctx()
    pytest = _PytestStub()

from chain import (
    Chain,
    GENESIS_PRIOR_HASH,
    canonical_json,
    load_or_create_key,
    merkle_proof,
    merkle_root,
    sha256,
    sha256_hex,
    verify_inclusion,
    verify_merkle_proof,
)
from retrieval import EmbeddingIndex, HashingEmbedder, Retriever
from agent import Agent, MockLLM, _humanize_delta, _format_absolute_time


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workdir():
    d = Path(tempfile.mkdtemp(prefix="timechain-test-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def chain(workdir):
    key = load_or_create_key(workdir / "operator.key")
    c = Chain(workdir / "chain.sqlite", key)
    yield c
    c.close()


@pytest.fixture
def index(workdir):
    embedder = HashingEmbedder(dim=64)
    idx = EmbeddingIndex(workdir / "embed.sqlite", embedder, dim=64)
    yield idx
    idx.close()


@pytest.fixture
def agent(chain, index):
    retriever = Retriever(chain, index)
    return Agent(chain, retriever, MockLLM(), system_prompt="test prompt")


# ---------------------------------------------------------------------------
# Chain integrity
# ---------------------------------------------------------------------------

class TestChainIntegrity:
    def test_first_record_links_to_genesis_zeros(self, chain):
        rec = chain.append("test", {"v": 1})
        assert rec.index == 0
        assert rec.prior_hash == GENESIS_PRIOR_HASH

    def test_subsequent_records_link_to_predecessor(self, chain):
        a = chain.append("test", {"n": 1})
        b = chain.append("test", {"n": 2})
        c = chain.append("test", {"n": 3})
        assert b.prior_hash == a.record_hash
        assert c.prior_hash == b.record_hash
        assert b.index == 1
        assert c.index == 2

    def test_signature_round_trip(self, chain):
        rec = chain.append("test", {"v": 42})
        ok, msg = chain.verify(expected_pubkey=chain.pubkey_hex)
        assert ok, msg

    def test_empty_chain_verifies(self, chain):
        ok, msg = chain.verify()
        assert ok

    def test_long_chain_verifies(self, chain):
        for i in range(50):
            chain.append("test", {"i": i, "data": f"record-{i}"})
        ok, msg = chain.verify(expected_pubkey=chain.pubkey_hex)
        assert ok, msg
        assert chain.length() == 50

    def test_unicode_content_round_trip(self, chain):
        for s in ["héllo", "日本語", "🌊 wave", "Ω alpha β"]:
            chain.append("unicode", {"text": s})
        ok, msg = chain.verify()
        assert ok, msg
        # Confirm content reads back identically
        for i, s in enumerate(["héllo", "日本語", "🌊 wave", "Ω alpha β"]):
            rec = chain.get(i)
            assert rec.content["text"] == s


# ---------------------------------------------------------------------------
# Tamper detection — the cryptographic heart of the system
# ---------------------------------------------------------------------------

class TestTamperDetection:
    def test_detects_content_modification(self, chain, workdir):
        chain.append("test", {"v": 1})
        chain.append("test", {"v": 2})
        chain.append("test", {"v": 3})
        chain.close()

        # Reach into SQLite and modify the content of record 1
        conn = sqlite3.connect(workdir / "chain.sqlite")
        conn.execute("UPDATE records SET content_json = ? WHERE idx = 1",
                     ('{"v": 999}',))
        conn.commit()
        conn.close()

        # Reopen and verify
        key = load_or_create_key(workdir / "operator.key")
        chain2 = Chain(workdir / "chain.sqlite", key)
        ok, msg = chain2.verify(expected_pubkey=chain2.pubkey_hex)
        chain2.close()
        assert not ok
        assert "index 1" in msg

    def test_detects_signature_modification(self, chain, workdir):
        chain.append("test", {"v": 1})
        chain.append("test", {"v": 2})
        chain.close()

        # Flip a single hex character in the signature
        conn = sqlite3.connect(workdir / "chain.sqlite")
        cur = conn.execute("SELECT signature FROM records WHERE idx = 1")
        sig = cur.fetchone()[0]
        bad_sig = ("0" if sig[0] != "0" else "1") + sig[1:]
        conn.execute("UPDATE records SET signature = ? WHERE idx = 1", (bad_sig,))
        conn.commit()
        conn.close()

        key = load_or_create_key(workdir / "operator.key")
        chain2 = Chain(workdir / "chain.sqlite", key)
        ok, msg = chain2.verify()
        chain2.close()
        assert not ok
        assert "signature" in msg.lower() or "index 1" in msg

    def test_detects_prior_hash_break(self, chain, workdir):
        chain.append("test", {"v": 1})
        chain.append("test", {"v": 2})
        chain.append("test", {"v": 3})
        chain.close()

        # Replace prior_hash on record 2 with a wrong-but-valid-looking value
        conn = sqlite3.connect(workdir / "chain.sqlite")
        conn.execute(
            "UPDATE records SET prior_hash = ? WHERE idx = 2",
            ("a" * 64,),
        )
        conn.commit()
        conn.close()

        key = load_or_create_key(workdir / "operator.key")
        chain2 = Chain(workdir / "chain.sqlite", key)
        ok, msg = chain2.verify()
        chain2.close()
        assert not ok

    def test_detects_record_deletion(self, chain, workdir):
        for i in range(5):
            chain.append("test", {"i": i})
        chain.close()

        # Delete record 2 — creates an index gap
        conn = sqlite3.connect(workdir / "chain.sqlite")
        conn.execute("DELETE FROM records WHERE idx = 2")
        conn.commit()
        conn.close()

        key = load_or_create_key(workdir / "operator.key")
        chain2 = Chain(workdir / "chain.sqlite", key)
        ok, msg = chain2.verify()
        chain2.close()
        assert not ok
        assert "index" in msg.lower()


# ---------------------------------------------------------------------------
# Merkle batching and inclusion proofs
# ---------------------------------------------------------------------------

class TestMerkle:
    def test_root_of_single_leaf(self):
        leaf = sha256(b"x")
        assert merkle_root([leaf]) == leaf

    def test_root_is_deterministic(self):
        leaves = [sha256(f"r{i}".encode()) for i in range(7)]
        assert merkle_root(leaves) == merkle_root(leaves)

    def test_root_changes_with_any_leaf_change(self):
        leaves = [sha256(f"r{i}".encode()) for i in range(8)]
        original = merkle_root(leaves)
        leaves[3] = sha256(b"different")
        assert merkle_root(leaves) != original

    def test_proof_verifies(self):
        leaves = [sha256(f"r{i}".encode()) for i in range(11)]  # odd count
        root = merkle_root(leaves)
        for target in range(len(leaves)):
            proof = merkle_proof(leaves, target)
            assert verify_merkle_proof(leaves[target], proof, root), \
                f"proof failed for leaf {target}"

    def test_proof_fails_for_wrong_leaf(self):
        leaves = [sha256(f"r{i}".encode()) for i in range(8)]
        root = merkle_root(leaves)
        proof = merkle_proof(leaves, 3)
        wrong = sha256(b"not in the set")
        assert not verify_merkle_proof(wrong, proof, root)

    def test_chain_batch_seal_and_inclusion(self, chain):
        for i in range(10):
            chain.append("test", {"i": i})
        batch = chain.seal_batch(batch_size=10)
        assert batch is not None
        assert batch["first_idx"] == 0
        assert batch["last_idx"] == 9

        # Inclusion proof should verify
        proof = chain.inclusion_proof(5)
        assert proof is not None
        assert verify_inclusion(
            proof["record_hash"],
            proof["proof"],
            proof["merkle_root"],
        )

    def test_seal_returns_none_on_empty(self, chain):
        # No records yet
        batch = chain.seal_batch()
        assert batch is None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_search_returns_results(self, chain, index):
        chain.append("note", {"text": "the cat sat on the mat"})
        chain.append("note", {"text": "the dog ran in the park"})
        chain.append("note", {"text": "fish swim in the sea"})
        index.index_chain(chain)

        results = index.search("cat", k=2)
        assert len(results) >= 1
        # Top hit should be the cat record (index 0)
        assert results[0][0] == 0

    def test_salience_boosts_reflections(self, chain, index):
        # Add a generic observation and a reflection on similar content
        chain.append("observation", {"text": "we discussed weather"})
        chain.append("reflection", {"text": "weather"})
        index.index_chain(chain)

        retriever = Retriever(chain, index)
        # Reflection should rank higher than observation due to salience boost,
        # even when semantic similarity is similar
        hits = retriever.hybrid("weather", k=2)
        assert hits[0].record.type == "reflection"

    def test_recency_includes_latest(self, chain, index):
        for i in range(10):
            chain.append("test", {"text": f"record {i}"})
        index.index_chain(chain)

        retriever = Retriever(chain, index)
        hits = retriever.recent(n=3)
        # Recent should give us the last 3 (indices 9, 8, 7 in some order)
        indices = sorted(h.record.index for h in hits)
        assert indices == [7, 8, 9]

    def test_ancestry_walk(self, chain, index):
        a = chain.append("test", {"n": 1})
        b = chain.append("test", {"n": 2}, refs=[a.record_hash])
        c = chain.append("test", {"n": 3}, refs=[b.record_hash])
        index.index_chain(chain)

        retriever = Retriever(chain, index)
        ancestors = retriever.ancestry(c.record_hash, depth=3)
        ancestor_indices = {h.record.index for h in ancestors}
        assert {0, 1, 2}.issubset(ancestor_indices)


# ---------------------------------------------------------------------------
# Agent behavior
# ---------------------------------------------------------------------------

class TestAgent:
    def test_genesis_writes_record_zero(self, agent, index):
        rec = agent.commit_genesis(["be honest"])
        index.index_record(rec)
        assert rec.index == 0
        assert rec.type == "genesis"
        assert rec.content["commitments"] == ["be honest"]

    def test_genesis_cannot_be_committed_twice(self, agent, index):
        agent.commit_genesis(["a"])
        with pytest.raises(RuntimeError):
            agent.commit_genesis(["b"])

    def test_turn_writes_two_records(self, agent, index):
        agent.commit_genesis(["be honest"])
        before = agent.chain.length()
        turn = agent.turn("hello there", retrieve_k=3)
        index.index_record(turn.observation_record)
        index.index_record(turn.response_record)
        after = agent.chain.length()
        assert after == before + 2
        assert turn.observation_record.type == "observation"
        assert turn.response_record.type == "response"

    def test_response_refs_include_observation(self, agent, index):
        agent.commit_genesis(["be honest"])
        turn = agent.turn("hello", retrieve_k=3)
        assert turn.observation_record.record_hash in turn.response_record.refs

    def test_log_system_prompt_writes_once_then_dedupes(self, agent, index):
        agent.commit_genesis(["be honest"])
        rec1 = agent.log_system_prompt()
        assert rec1 is not None
        # Calling again with same prompt should return None
        rec2 = agent.log_system_prompt()
        assert rec2 is None

    def test_log_system_prompt_writes_again_on_change(self, chain, index):
        retriever = Retriever(chain, index)
        agent_a = Agent(chain, retriever, MockLLM(), system_prompt="prompt A")
        agent_a.commit_genesis(["be honest"])
        agent_a.log_system_prompt()

        # New agent with new prompt against same chain
        agent_b = Agent(chain, retriever, MockLLM(), system_prompt="prompt B")
        rec = agent_b.log_system_prompt()
        assert rec is not None
        assert rec.content["text"] == "prompt B"

    def test_reflect_needs_history(self, agent, index):
        agent.commit_genesis(["be honest"])
        # Not enough substantive records yet
        assert agent.reflect(window=10) is None

    def test_reflect_writes_record(self, agent, index):
        agent.commit_genesis(["be honest"])
        for msg in ["hi", "test 1", "test 2", "test 3"]:
            t = agent.turn(msg)
            index.index_record(t.observation_record)
            index.index_record(t.response_record)
        rec = agent.reflect(window=20)
        assert rec is not None
        assert rec.type == "reflection"
        assert "text" in rec.content
        # Reflection refs all records it considered
        assert len(rec.refs) > 0

    def test_revise_creates_new_record_originalstays(self, agent, index):
        agent.commit_genesis(["be honest"])
        t = agent.turn("hello")
        index.index_record(t.observation_record)
        index.index_record(t.response_record)
        original_idx = t.response_record.index
        original_content = agent.chain.get(original_idx).content

        rev = agent.revise(original_idx, "actually I meant something else")
        index.index_record(rev)

        # Original record content unchanged
        assert agent.chain.get(original_idx).content == original_content
        # Revision record exists with proper linkage
        assert rev.type == "revision"
        assert rev.content["revises_index"] == original_idx
        assert rev.content["revises_hash"] == agent.chain.get(original_idx).record_hash

    def test_revise_returns_none_for_missing_index(self, agent):
        agent.commit_genesis(["be honest"])
        assert agent.revise(9999, "no such record") is None

    def test_chain_verifies_after_full_workflow(self, agent, index):
        agent.commit_genesis(["be honest"])
        agent.log_system_prompt()
        for msg in ["hi", "test 1", "test 2", "test 3", "test 4"]:
            t = agent.turn(msg)
            index.index_record(t.observation_record)
            index.index_record(t.response_record)
        agent.reflect(window=20)
        agent.revise(2, "correction")
        ok, msg = agent.chain.verify(expected_pubkey=agent.chain.pubkey_hex)
        assert ok, msg


# ---------------------------------------------------------------------------
# File ingestion
# ---------------------------------------------------------------------------

class TestFileIngestion:
    def test_text_file_ingest(self, workdir, chain, index):
        from agent import Agent
        retriever = Retriever(chain, index)
        blob_dir = workdir / "blobs"
        agent = Agent(chain, retriever, MockLLM(), system_prompt="t", blob_dir=blob_dir)

        # Create a text file
        f = workdir / "hello.txt"
        f.write_text("hello world\nthis is a test file\nline three", encoding="utf-8")

        rec = agent.ingest_file(f)
        index.index_record(rec)

        assert rec.type == "file"
        assert rec.content["filename"] == "hello.txt"
        assert rec.content["kind"] == "document"
        assert "hello world" in rec.content["extracted_text"]
        assert (blob_dir / rec.content["blob_sha256"]).exists()

    def test_csv_file_ingest(self, workdir, chain, index):
        from agent import Agent
        retriever = Retriever(chain, index)
        blob_dir = workdir / "blobs"
        agent = Agent(chain, retriever, MockLLM(), system_prompt="t", blob_dir=blob_dir)

        f = workdir / "data.csv"
        f.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n", encoding="utf-8")

        rec = agent.ingest_file(f)
        index.index_record(rec)
        assert rec.content["kind"] == "spreadsheet"
        assert "Alice" in rec.content["extracted_text"]
        assert "30" in rec.content["extracted_text"]

    def test_code_file_ingest(self, workdir, chain, index):
        from agent import Agent
        retriever = Retriever(chain, index)
        blob_dir = workdir / "blobs"
        agent = Agent(chain, retriever, MockLLM(), system_prompt="t", blob_dir=blob_dir)

        f = workdir / "test.py"
        f.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

        rec = agent.ingest_file(f)
        assert rec.content["kind"] == "code"
        assert "def hello" in rec.content["extracted_text"]

    def test_unsupported_extension_rejected(self, workdir, chain, index):
        from agent import Agent
        retriever = Retriever(chain, index)
        blob_dir = workdir / "blobs"
        agent = Agent(chain, retriever, MockLLM(), system_prompt="t", blob_dir=blob_dir)

        f = workdir / "binary.exe"
        f.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError):
            agent.ingest_file(f)

    def test_missing_file_raises(self, workdir, chain, index):
        from agent import Agent
        retriever = Retriever(chain, index)
        blob_dir = workdir / "blobs"
        agent = Agent(chain, retriever, MockLLM(), system_prompt="t", blob_dir=blob_dir)
        with pytest.raises(FileNotFoundError):
            agent.ingest_file(workdir / "does_not_exist.txt")

    def test_blob_sha_matches_content(self, workdir, chain, index):
        from agent import Agent
        from file_ingest import verify_blob
        retriever = Retriever(chain, index)
        blob_dir = workdir / "blobs"
        agent = Agent(chain, retriever, MockLLM(), system_prompt="t", blob_dir=blob_dir)

        f = workdir / "x.txt"
        f.write_text("integrity test content", encoding="utf-8")
        rec = agent.ingest_file(f)
        assert verify_blob(rec.content, blob_dir)

        # Tamper with the blob — verify_blob should detect it
        blob_path = blob_dir / rec.content["blob_sha256"]
        blob_path.write_bytes(b"tampered content")
        assert not verify_blob(rec.content, blob_dir)

    def test_chain_verifies_after_file_ingest(self, workdir, chain, index):
        from agent import Agent
        retriever = Retriever(chain, index)
        blob_dir = workdir / "blobs"
        agent = Agent(chain, retriever, MockLLM(), system_prompt="t", blob_dir=blob_dir)

        agent.commit_genesis(["x"])
        f = workdir / "doc.md"
        f.write_text("# Title\n\nSome content here.", encoding="utf-8")
        agent.ingest_file(f)
        agent.turn("tell me about the file")

        ok, msg = chain.verify(expected_pubkey=chain.pubkey_hex)
        assert ok, msg

    def test_ingest_without_blob_dir_raises(self, chain, index):
        from agent import Agent
        retriever = Retriever(chain, index)
        agent = Agent(chain, retriever, MockLLM())  # no blob_dir
        with pytest.raises(RuntimeError, match="blob_dir"):
            agent.ingest_file("/tmp/anything.txt")

class TestGenesisDrift:
    def test_no_drift_returns_none(self, agent, index):
        commitments = ["a", "b", "c"]
        agent.commit_genesis(commitments)
        assert agent.check_genesis_drift(commitments) is None

    def test_drift_detected_when_commitments_differ(self, agent, index):
        agent.commit_genesis(["a", "b"])
        drift = agent.check_genesis_drift(["a", "b", "c"])
        assert drift is not None
        assert drift["status"] == "drift"
        assert drift["stored"] == ["a", "b"]
        assert drift["configured"] == ["a", "b", "c"]

    def test_drift_check_safe_on_empty_chain(self, agent):
        # No genesis yet
        assert agent.check_genesis_drift(["whatever"]) is None


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

class TestTimeFormatting:
    @pytest.mark.parametrize("seconds,expected", [
        (1, "just now"),
        (4, "just now"),
        (5, "5 seconds ago"),
        (59, "59 seconds ago"),
        (60, "1 minute ago"),
        (120, "2 minutes ago"),
        (3600, "1 hour ago"),
        (7200, "2 hours ago"),
        (86400, "1 day ago"),
        (172800, "2 days ago"),
        (604800, "1 week ago"),
        (2592000, "1 month ago"),
        (31536000, "1 year ago"),
    ])
    def test_humanize_delta(self, seconds, expected):
        assert _humanize_delta(seconds) == expected

    def test_format_absolute_time_format(self):
        ts = 1735689600000  # 2025-01-01 00:00:00 UTC
        formatted = _format_absolute_time(ts)
        assert "2025-01-01" in formatted
        assert "UTC" in formatted


# ---------------------------------------------------------------------------
# Token-budget truncation
# ---------------------------------------------------------------------------

class TestContextBudget:
    def test_no_truncation_under_budget(self, chain, index):
        retriever = Retriever(chain, index)
        agent = Agent(chain, retriever, MockLLM(), context_char_budget=10_000)
        agent.commit_genesis(["x"])
        for i in range(5):
            t = agent.turn(f"msg {i}")
            index.index_record(t.observation_record)
            index.index_record(t.response_record)
        # Build a normal prompt — should include all small records
        prompt = agent._format_prompt("ping", agent.retriever.build_context("ping"))
        assert "omitted" not in prompt

    def test_truncation_kicks_in_with_tight_budget(self, chain, index):
        retriever = Retriever(chain, index)
        # Very tight budget — most records should get dropped
        agent = Agent(chain, retriever, MockLLM(), context_char_budget=400)
        agent.commit_genesis(["x"])
        for i in range(20):
            t = agent.turn(f"long message number {i} with lots of text padding " * 5)
            index.index_record(t.observation_record)
            index.index_record(t.response_record)
        prompt = agent._format_prompt(
            "ping",
            agent.retriever.build_context("ping", k_semantic=15, n_recent=15),
        )
        assert "omitted" in prompt

    def test_truncation_keeps_higher_priority_types(self, chain, index):
        retriever = Retriever(chain, index)
        agent = Agent(chain, retriever, MockLLM(), context_char_budget=500)
        agent.commit_genesis(["important commitment"])
        # Add a reflection and many observations
        for i in range(10):
            t = agent.turn(f"long observation {i} " * 10)
            index.index_record(t.observation_record)
            index.index_record(t.response_record)
        refl = agent.reflect(window=10)
        if refl:
            index.index_record(refl)

        # Fetch candidates including all types
        ctx = agent.retriever.build_context("anything", k_semantic=30, n_recent=30)
        # Verify higher-priority types survive truncation when budget is tight
        # by checking what _truncate_to_budget does directly:
        kept, dropped = agent._truncate_to_budget(ctx, fixed_overhead_chars=200)
        if dropped > 0:
            # Among kept, no observation should outrank a dropped reflection
            kept_types = {r.type for r in kept}
            # Genesis or reflection (high priority) should be present if any was retrieved
            high_priority_in_ctx = any(r.type in ("genesis", "reflection", "revision")
                                       for r in ctx)
            if high_priority_in_ctx:
                assert any(r.type in ("genesis", "reflection", "revision") for r in kept)
