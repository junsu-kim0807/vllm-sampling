#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run tests for hierarchical verification config and spec decode cost breakdown.

Usage (from repo root, with vllm dev deps and torch installed):

  # Recommended: run pytest
  pytest tests/v1/spec_decode/test_hierarchical_verification_config.py -v

  # Or run this script (uses pytest if available, else runs checks manually)
  python scripts/run_hierarchical_verification_tests.py
"""

import sys
from pathlib import Path

# Add project root so "vllm" and "tests" are importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_with_pytest() -> bool:
    """Run the test file with pytest. Returns True if pytest ran successfully."""
    try:
        import pytest
    except ImportError:
        return False
    return pytest.main([
        "-v",
        str(REPO_ROOT / "tests" / "v1" / "spec_decode" / "test_hierarchical_verification_config.py"),
    ]) == 0


def run_standalone() -> bool:
    """Run the same test logic without pytest."""
    from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
    from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingStats

    def make_ngram_config(
        hierarchical_verification: bool = False,
        compress_method: str = "random",
        compression_ratio: float = 0.5,
    ) -> SpeculativeConfig:
        target_model_config = ModelConfig("facebook/opt-125m", trust_remote_code=True)
        return SpeculativeConfig(
            method="ngram",
            prompt_lookup_min=1,
            prompt_lookup_max=1,
            num_speculative_tokens=2,
            target_model_config=target_model_config,
            target_parallel_config=ParallelConfig(),
            hierarchical_verification=hierarchical_verification,
            compress_method=compress_method,
            compression_ratio=compression_ratio,
        )

    ok = True

    # Config defaults
    config = make_ngram_config()
    assert config.hierarchical_verification is False, "default hierarchical_verification"
    assert config.compress_method == "random", "default compress_method"
    assert config.compression_ratio == 0.5, "default compression_ratio"
    print("  [OK] Config defaults")

    # Config explicit
    config = make_ngram_config(
        hierarchical_verification=True, compress_method="random", compression_ratio=0.25
    )
    assert config.hierarchical_verification is True
    assert config.compression_ratio == 0.25
    print("  [OK] Config explicit hierarchical_verification")

    # Config hash
    c1 = make_ngram_config(hierarchical_verification=False)
    c2 = make_ngram_config(hierarchical_verification=True)
    assert c1.compute_hash() != c2.compute_hash(), "hash differs by hierarchical_verification"
    print("  [OK] Config compute_hash")

    # SpecDecodingStats cost breakdown
    stats = SpecDecodingStats.new(num_spec_tokens=4)
    assert stats.draft_time_sec == 0.0 and stats.full_verification_time_sec == 0.0
    print("  [OK] SpecDecodingStats.new zero cost breakdown")

    stats.observe_draft(num_draft_tokens=4, num_accepted_tokens=2)
    assert stats.num_drafts == 1 and stats.num_accepted_tokens == 2
    assert stats.draft_time_sec == 0.0
    print("  [OK] SpecDecodingStats.observe_draft unchanged")

    stats2 = SpecDecodingStats.new(num_spec_tokens=4)
    stats2.observe_draft_with_cost_breakdown(
        num_draft_tokens=4,
        num_accepted_tokens=2,
        draft_time_sec=0.01,
        full_verification_time_sec=0.008,
    )
    assert stats2.draft_time_sec == 0.01 and stats2.full_verification_time_sec == 0.008
    print("  [OK] SpecDecodingStats.observe_draft_with_cost_breakdown")

    # SpecDecodingLogging
    logging = SpecDecodingLogging()
    st = SpecDecodingStats.new(num_spec_tokens=2)
    st.observe_draft_with_cost_breakdown(
        num_draft_tokens=2, num_accepted_tokens=1,
        draft_time_sec=0.01, full_verification_time_sec=0.02,
    )
    logging.observe(st)
    assert len(logging.draft_time_sec) == 1 and logging.draft_time_sec[0] == 0.01
    logging.log(log_fn=lambda *a, **k: None)
    assert len(logging.draft_time_sec) == 0
    print("  [OK] SpecDecodingLogging cost breakdown")

    return ok


def main() -> int:
    print("Hierarchical verification config & cost breakdown tests")
    print("-" * 50)
    if run_with_pytest():
        print("Done (pytest).")
        return 0
    print("pytest not found, running standalone checks...")
    try:
        if run_standalone():
            print("-" * 50)
            print("Done (standalone).")
            return 0
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
