"""
run_tests — standalone runner for test_timechain.py without requiring pytest.

This is a fallback for environments where pip can't reach PyPI. For normal
development, install pytest and use it directly:

    pip install pytest
    pytest test_timechain.py -v

This runner gives you the same coverage minus the parametrized cases.
"""

from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import test_timechain as t
from chain import Chain, load_or_create_key
from retrieval import EmbeddingIndex, HashingEmbedder, Retriever
from agent import Agent, MockLLM


def build_fixtures(param_names: list[str], workdir: Path) -> dict:
    """Recreate the pytest fixtures by hand."""
    out = {}
    if "workdir" in param_names:
        out["workdir"] = workdir
    if "chain" in param_names or "agent" in param_names:
        key = load_or_create_key(workdir / "operator.key")
        out["chain"] = Chain(workdir / "chain.sqlite", key)
    if "index" in param_names or "agent" in param_names:
        embedder = HashingEmbedder(dim=64)
        out["index"] = EmbeddingIndex(workdir / "embed.sqlite", embedder, dim=64)
    if "agent" in param_names:
        retriever = Retriever(out["chain"], out["index"])
        out["agent"] = Agent(out["chain"], retriever, MockLLM(), system_prompt="test prompt")
    # Filter to just the requested params
    return {k: v for k, v in out.items() if k in param_names}


def run_one(method, instance):
    """Run a single test method, building fixtures as needed.

    Returns (status, detail). status is True (passed), False (failed), or
    the string "skip" when the test called pytest.skip().
    """
    sig = inspect.signature(method)
    workdir = Path(tempfile.mkdtemp(prefix="ttest-"))
    fixtures = {}
    try:
        fixtures = build_fixtures(list(sig.parameters), workdir)
        method(**fixtures)
        return True, ""
    except BaseException as e:
        # pytest.skip() raises _pytest.outcomes.Skipped, which derives from
        # BaseException (NOT Exception), so a bare `except Exception` lets it
        # escape and crash the harness. Treat it as a skip instead. Matching
        # on the class name keeps this runner pytest-import-free.
        if type(e).__name__ == "Skipped":
            return "skip", str(e)
        # Let genuine interpreter-level signals through.
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        return False, traceback.format_exc()
    finally:
        for v in fixtures.values():
            if hasattr(v, "close"):
                try:
                    v.close()
                except Exception:
                    pass
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    classes = [
        getattr(t, n)
        for n in dir(t)
        if n.startswith("Test") and inspect.isclass(getattr(t, n))
    ]
    passed, failed, skipped = 0, 0, 0
    failures: list[tuple[str, str]] = []

    for cls in classes:
        instance = cls()
        method_names = [m for m in dir(cls) if m.startswith("test_")]
        for name in method_names:
            method = getattr(instance, name)
            full_name = f"{cls.__name__}.{name}"
            # Skip parametrized tests — they need pytest's runner.
            # Detect by checking if the method requires non-fixture params.
            sig = inspect.signature(getattr(cls, name))
            known_fixtures = {"self", "workdir", "chain", "index", "agent"}
            non_fixture_params = [
                p for p in sig.parameters if p not in known_fixtures
            ]
            if non_fixture_params:
                skipped += 1
                print(f"  SKIP {full_name} (parametrized — needs pytest)")
                continue
            ok, err = run_one(method, instance)
            if ok == "skip":
                skipped += 1
                print(f"  SKIP {full_name} ({err})")
            elif ok:
                passed += 1
                print(f"  ok   {full_name}")
            else:
                failed += 1
                failures.append((full_name, err))
                print(f"  FAIL {full_name}")

    print()
    print(f"results: {passed} passed, {failed} failed, {skipped} skipped")
    if failures:
        print()
        for name, tb in failures:
            print(f"=== {name} ===")
            print(tb)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
