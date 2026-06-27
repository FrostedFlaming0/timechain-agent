# pytest configuration.
#
# test_cypher_port.py and test_cypher_integration.py are STANDALONE suites, run
# via `python3 test_cypher_port.py` / `python3 test_cypher_integration.py`. Their
# `test_*` functions take a positional `workdir` (not a pytest fixture) and
# report via a `check()` helper that does not raise — so pytest must not collect
# them: it would misread `workdir` as a missing fixture (erroring), and even if
# it didn't, a failing `check()` would not fail the test. The pytest suite is
# test_timechain.py.
collect_ignore = [
    "test_cypher_port.py",
    "test_cypher_integration.py",
]
