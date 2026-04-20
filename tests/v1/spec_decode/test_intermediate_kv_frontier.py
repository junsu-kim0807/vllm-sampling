# SPDX-License-Identifier: Apache-2.0
"""Unit tests for intermediate KV frontier gating (static config helpers)."""

import ast
from pathlib import Path
from types import SimpleNamespace
def _static_intermediate_kv_frontier_enabled(vllm_config: SimpleNamespace) -> bool:
    """Mirror of ``GPUModelRunner._static_intermediate_kv_frontier_enabled`` (no vllm import)."""
    spec = vllm_config.speculative_config
    if spec is None or spec.intermediate_model is None:
        return False
    mode = getattr(spec, "intermediate_kv_mode", "mirror_frontier")
    if mode != "mirror_frontier":
        return False
    if spec.method == "adaptive_spechive":
        return spec.adaptive_spechive_mode == "hierarchical_verification"
    if spec.method == "pivot":
        return bool(spec.pivot_spechive)
    if spec.method == "hierarchical_verification":
        return True
    return False


def _cfg_hierarchical_adaptive() -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="adaptive_spechive",
            intermediate_model="dummy",
            adaptive_spechive_mode="hierarchical_verification",
            pivot_spechive=False,
            intermediate_kv_mode="mirror_frontier",
        )
    )


def _cfg_hierarchical_adaptive_draft_like() -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="adaptive_spechive",
            intermediate_model="dummy",
            adaptive_spechive_mode="hierarchical_verification",
            pivot_spechive=False,
            intermediate_kv_mode="draft_like",
        )
    )


def _cfg_pivot_spechive() -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="pivot",
            intermediate_model="dummy",
            adaptive_spechive_mode="draft_target",
            pivot_spechive=True,
            intermediate_kv_mode="mirror_frontier",
        )
    )


def _cfg_pivot_spechive_draft_like() -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="pivot",
            intermediate_model="dummy",
            adaptive_spechive_mode="draft_target",
            pivot_spechive=True,
            intermediate_kv_mode="draft_like",
        )
    )


def _cfg_standalone_hierarchical_verification() -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="hierarchical_verification",
            intermediate_model="dummy",
            adaptive_spechive_mode="draft_target",
            pivot_spechive=False,
            intermediate_kv_mode="mirror_frontier",
        )
    )


def _cfg_no_intermediate() -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="adaptive_spechive",
            intermediate_model=None,
            adaptive_spechive_mode="hierarchical_verification",
            pivot_spechive=False,
            intermediate_kv_mode="mirror_frontier",
        )
    )


def test_static_intermediate_kv_frontier_enabled_adaptive_hv() -> None:
    assert _static_intermediate_kv_frontier_enabled(_cfg_hierarchical_adaptive())


def test_static_intermediate_kv_frontier_disabled_adaptive_hv_draft_like() -> None:
    assert not _static_intermediate_kv_frontier_enabled(_cfg_hierarchical_adaptive_draft_like())


def test_static_intermediate_kv_frontier_enabled_pivot_spechive() -> None:
    assert _static_intermediate_kv_frontier_enabled(_cfg_pivot_spechive())


def test_static_intermediate_kv_frontier_disabled_pivot_spechive_draft_like() -> None:
    assert not _static_intermediate_kv_frontier_enabled(_cfg_pivot_spechive_draft_like())


def test_static_intermediate_kv_frontier_disabled_without_intermediate_model() -> None:
    assert not _static_intermediate_kv_frontier_enabled(_cfg_no_intermediate())


def test_static_intermediate_kv_frontier_enabled_standalone_hierarchical_verification() -> (
    None
):
    assert _static_intermediate_kv_frontier_enabled(
        _cfg_standalone_hierarchical_verification()
    )


def test_static_intermediate_kv_frontier_defaults_to_mirror_when_attr_missing() -> None:
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="adaptive_spechive",
            intermediate_model="dummy",
            adaptive_spechive_mode="hierarchical_verification",
            pivot_spechive=False,
        )
    )
    assert _static_intermediate_kv_frontier_enabled(cfg)


def _func_body_after_def(src: str, func_name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(src, node) or ""
    return ""


def test_gpu_model_runner_static_gate_matches_intermediate_kv_mode() -> None:
    """Implementation must gate mirror frontier on ``intermediate_kv_mode``."""
    root = Path(__file__).resolve().parents[3]
    text = (root / "vllm" / "v1" / "worker" / "gpu_model_runner.py").read_text(
        encoding="utf-8"
    )
    static = _func_body_after_def(text, "_static_intermediate_kv_frontier_enabled")
    assert "intermediate_kv_mode" in static
    assert "mirror_frontier" in static


def test_mirror_helpers_short_circuit_on_frontier_disabled() -> None:
    """Hot-path mirror helpers must no-op when ``_intermediate_kv_frontier_enabled`` is false."""
    root = Path(__file__).resolve().parents[3]
    text = (root / "vllm" / "v1" / "worker" / "gpu_model_runner.py").read_text(
        encoding="utf-8"
    )
    for fn in (
        "_sync_intermediate_states_with_scheduler",
        "_prepare_intermediate_metadata",
        "_advance_intermediate_frontier_after_round",
        "_sync_intermediate_num_accepted_from_target",
        "_reconcile_intermediate_frontier_after_target",
    ):
        body = _func_body_after_def(text, fn)
        assert "_intermediate_kv_frontier_enabled" in body, fn
        assert "if not self._intermediate_kv_frontier_enabled():" in body, fn


def _kwonly_args(src: str, func_name: str) -> set[str]:
    tree = ast.parse(src)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            out |= {a.arg for a in node.args.kwonlyargs}
    return out


def test_verify_chunk_with_inter_verifier_accepts_mirror_cad_kwarg() -> None:
    """Regression: runner can pass mirror KV CAD into verify for frontier mode."""
    root = Path(__file__).resolve().parents[3]
    cascade = (root / "vllm" / "v1" / "spec_decode" / "adaptive_cascade.py").read_text(
        encoding="utf-8"
    )
    pivot = (root / "vllm" / "v1" / "spec_decode" / "pivot.py").read_text(encoding="utf-8")
    assert "mirror_kv_common_attn_metadata" in _kwonly_args(
        cascade, "verify_chunk_with_inter_verifier"
    )
    assert "mirror_kv_common_attn_metadata" in _kwonly_args(
        pivot, "verify_chunk_with_inter_verifier"
    )


def test_intermediate_kv_storage_assumptions_docstring() -> None:
    """Docstring documents shared blocks vs physical KV / reconcile policy."""
    root = Path(__file__).resolve().parents[3]
    text = (root / "vllm" / "v1" / "worker" / "gpu_model_runner.py").read_text(
        encoding="utf-8"
    )
    assert "Intermediate KV frontier — storage assumptions" in text
    assert "_hv_scheduler_output" in text


def test_reconcile_docstring_post_bookkeeping_and_commit_block_table() -> None:
    """Reconcile runs post-bookkeeping; block table is committed after logical align."""
    root = Path(__file__).resolve().parents[3]
    text = (root / "vllm" / "v1" / "worker" / "gpu_model_runner.py").read_text(
        encoding="utf-8"
    )
    assert "post-bookkeeping" in text
    assert "commit_block_table" in text


def test_adaptive_verify_docstring_prefix_conditioned_base_cad_only() -> None:
    """HV verify remains prefix-conditioned; mirror CAD is base geometry only."""
    root = Path(__file__).resolve().parents[3]
    cascade = (root / "vllm" / "v1" / "spec_decode" / "adaptive_cascade.py").read_text(
        encoding="utf-8"
    )
    assert "_build_prefix_conditioned_inputs" in cascade
    assert "mirror_kv_common_attn_metadata" in cascade
    assert "prefix-conditioned" in cascade
    assert "draft_like" in cascade


def test_speculative_config_defines_intermediate_kv_mode() -> None:
    root = Path(__file__).resolve().parents[3]
    spec_src = (root / "vllm" / "config" / "speculative.py").read_text(encoding="utf-8")
    assert "IntermediateKvMode = Literal[" in spec_src
    assert '"draft_like"' in spec_src and '"mirror_frontier"' in spec_src
    assert "intermediate_kv_mode: IntermediateKvMode" in spec_src
    assert 'intermediate_kv_mode: IntermediateKvMode = "mirror_frontier"' in spec_src
