# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Policy and wiring checks for standalone hierarchical_verification + frontier."""

import ast
from pathlib import Path


def _func_body(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    return ""


def test_vllm_as_plain_draft_sets_prompt_lookup_for_replace_validation() -> None:
    """HV (and similar) __post_init__ leaves prompt_lookup_* at 0; replace(draft_model) must override."""
    root = Path(__file__).resolve().parents[3]
    text = (root / "vllm" / "v1" / "spec_decode" / "adaptive_cascade.py").read_text(
        encoding="utf-8"
    )
    body = _func_body(text, "_vllm_as_plain_draft")
    assert "prompt_lookup_min=1" in body
    assert "prompt_lookup_max=1" in body


def test_run_hv_draft_step_does_not_call_adaptive_build_prefix_conditioned_inputs() -> None:
    """Legacy draft HV path must use runner-local ``hv_step_packing``, not adaptive_cascade."""
    root = Path(__file__).resolve().parents[3]
    text = (root / "vllm" / "v1" / "worker" / "gpu_model_runner.py").read_text(
        encoding="utf-8"
    )
    body = _func_body(text, "_run_hv_draft_step")
    assert "hv_build_prefix_conditioned_inputs(" in body
    # Do not substring-match "_build_prefix_conditioned_inputs(" — that would match
    # hv_build_prefix_conditioned_inputs(.
    assert "adaptive_cascade._build_prefix_conditioned_inputs(" not in body


def test_run_hv_draft_step_from_frontier_has_no_prefix_packing() -> None:
    """Standalone HV draft frontier path must not call ``hv_build_prefix_conditioned_inputs``."""
    root = Path(__file__).resolve().parents[3]
    text = (root / "vllm" / "v1" / "worker" / "gpu_model_runner.py").read_text(
        encoding="utf-8"
    )
    body = _func_body(text, "_run_hv_draft_step_from_frontier")
    assert "hv_build_prefix_conditioned_inputs(" not in body
    assert "_prepare_draft_metadata" in body


def test_speculative_verify_args_hierarchical_has_no_self_cache_config() -> None:
    """SpeculativeConfig has no cache_config; hierarchical block must not reference it."""
    root = Path(__file__).resolve().parents[3]
    lines = (root / "vllm" / "config" / "speculative.py").read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == 'if self.method == "hierarchical_verification":':
            start = i
            break
    assert start is not None
    base_indent = len(lines[start]) - len(lines[start].lstrip(" "))
    block_lines: list[str] = []
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            block_lines.append(line)
            continue
        ind = len(line) - len(line.lstrip(" "))
        if ind <= base_indent and line.lstrip().startswith("if self.method"):
            break
        block_lines.append(line)
    block = "\n".join(block_lines)
    assert "self.cache_config" not in block


def test_hierarchical_verification_module_skips_adaptive_propose_chunk_import() -> None:
    """Standalone proposer must not depend on adaptive_cascade._propose_chunk_from_prefix."""
    root = Path(__file__).resolve().parents[3]
    text = (root / "vllm" / "v1" / "spec_decode" / "hierarchical_verification.py").read_text(
        encoding="utf-8"
    )
    assert "_propose_chunk_from_prefix" not in text


def test_hv_propose_prefill_fallback_pads_to_hv_cap_width() -> None:
    """No scheduled spec slots: plain draft returns width L; pad to ``self.k`` for runner CPU copy."""
    root = Path(__file__).resolve().parents[3]
    text = (root / "vllm" / "v1" / "spec_decode" / "hierarchical_verification.py").read_text(
        encoding="utf-8"
    )
    body = _func_body(text, "propose")
    assert "not so.scheduled_spec_decode_tokens" in body
    assert "if w < self.k:" in body
    assert "PLACEHOLDER_TOKEN_ID" in body
    assert "torch.cat([out, pad], dim=-1)" in body


def test_create_engine_config_rejects_hv_with_mamba_align() -> None:
    """Engine-level gate: HV + intermediate frontier + mamba align is rejected early."""
    root = Path(__file__).resolve().parents[3]
    text = (root / "vllm" / "engine" / "arg_utils.py").read_text(encoding="utf-8")
    assert 'speculative_config.method == "hierarchical_verification"' in text
    assert "cache_config.mamba_cache_mode == \"align\"" in text
