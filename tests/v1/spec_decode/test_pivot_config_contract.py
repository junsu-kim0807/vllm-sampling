# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import pytest

from vllm.config import SpeculativeConfig


@dataclass
class _FakeHFConfig:
    model_type: str = "llama"
    architectures: list[str] = field(default_factory=lambda: ["LlamaForCausalLM"])


@dataclass
class _FakeModelConfig:
    vocab_size: int
    hf_config: _FakeHFConfig

    def get_vocab_size(self) -> int:
        return self.vocab_size


def _make_spec(
    *,
    method: str = "pivot",
    num_speculative_tokens: int = 5,
    parallel_drafting: bool = False,
    target_vocab: int = 32000,
    draft_vocab: int = 32000,
    intermediate_vocab: int = 32000,
    intermediate_model_type: str = "llama",
    intermediate_arch: list[str] | None = None,
    pivot_proposal_engine: str | None = None,
    pivot_verification_pipeline: str | None = None,
    pivot_hidden_state_source: str | None = None,
    pivot_spechive: bool = False,
    pivot_spechive_num_rounds: int = 1,
    pivot_use_eagle_tree: bool = False,
    pivot_family_collapse_policy: str = "max_accept_len",
) -> SpeculativeConfig:
    spec = object.__new__(SpeculativeConfig)
    spec.method = method
    spec.num_speculative_tokens = num_speculative_tokens
    spec.parallel_drafting = parallel_drafting
    spec.target_model_config = _FakeModelConfig(
        vocab_size=target_vocab,
        hf_config=_FakeHFConfig(model_type="llama"),
    )
    spec.draft_model_config = _FakeModelConfig(
        vocab_size=draft_vocab,
        hf_config=_FakeHFConfig(model_type="llama"),
    )
    spec.intermediate_model_config = _FakeModelConfig(
        vocab_size=intermediate_vocab,
        hf_config=_FakeHFConfig(
            model_type=intermediate_model_type,
            architectures=intermediate_arch or ["LlamaForCausalLM"],
        ),
    )
    spec.adaptive_spechive_mode = "draft_target"
    spec.adaptive_spechive_num_rounds = 1
    spec.pivot_topk_selection = 5
    spec.pivot_expansion_pct = 0.2
    spec.pivot_spechive = pivot_spechive
    spec.pivot_spechive_num_rounds = pivot_spechive_num_rounds
    spec.pivot_proposal_engine = pivot_proposal_engine
    spec.pivot_verification_pipeline = pivot_verification_pipeline
    spec.pivot_hidden_state_source = pivot_hidden_state_source
    spec.pivot_use_eagle_tree = pivot_use_eagle_tree
    spec.pivot_family_collapse_policy = pivot_family_collapse_policy
    spec.magicdec = False
    spec.magicdec_method = "streaming"
    spec.magicdec_kv_budget = 256
    spec.intermediate_model = "fake/intermediate"
    spec.speculative_token_tree = str([(i + 1) * (0,) for i in range(num_speculative_tokens)])
    return spec


def test_pivot_slot_budget_follows_verified_width_contract() -> None:
    spec = _make_spec(num_speculative_tokens=7, parallel_drafting=False)
    assert spec.uses_draft_model()
    # One extra slot per request follows draft-model contract, independent of
    # internal I/D staging.
    assert spec.max_num_new_slots_for_drafting == 1
    assert spec.runner_num_speculative_tokens() == 7


def test_pivot_eagle_head_does_not_report_draft_model_slot_contract() -> None:
    spec = _make_spec(
        pivot_proposal_engine="eagle3_head",
        pivot_verification_pipeline="target_only",
        pivot_hidden_state_source="target",
    )
    assert not spec.uses_draft_model()
    assert spec.uses_model_based_drafter()
    assert spec.uses_gpu_sampled_tokens_for_drafting()
    assert spec.max_num_new_slots_for_drafting == 0


def test_pivot_eagle_head_requires_aux_hidden_outputs() -> None:
    spec = _make_spec(
        pivot_proposal_engine="eagle3_head",
        pivot_verification_pipeline="intermediate_then_target",
        pivot_hidden_state_source="intermediate",
    )
    assert spec.requires_aux_hidden_state_outputs()


def test_pivot_mode_updates_compute_hash() -> None:
    spec_a = _make_spec(
        pivot_proposal_engine="draft_model",
        pivot_verification_pipeline="target_only",
    )
    spec_b = _make_spec(
        pivot_proposal_engine="eagle3_head",
        pivot_verification_pipeline="target_only",
        pivot_hidden_state_source="target",
    )
    assert spec_a.compute_hash() != spec_b.compute_hash()


def test_pivot_tree_mode_updates_compute_hash() -> None:
    spec_a = _make_spec(
        pivot_proposal_engine="eagle3_head",
        pivot_verification_pipeline="target_only",
        pivot_hidden_state_source="target",
        pivot_use_eagle_tree=False,
    )
    spec_b = _make_spec(
        pivot_proposal_engine="eagle3_head",
        pivot_verification_pipeline="target_only",
        pivot_hidden_state_source="target",
        pivot_use_eagle_tree=True,
    )
    assert spec_a.compute_hash() != spec_b.compute_hash()


def test_pivot_runner_width_uses_intermediate_pipeline_contract() -> None:
    spec = _make_spec(
        num_speculative_tokens=4,
        pivot_proposal_engine="draft_model",
        pivot_verification_pipeline="intermediate_then_target",
        pivot_spechive_num_rounds=3,
    )
    assert spec.runner_num_speculative_tokens() == 4 + 3 * (4 + 1)


def test_pivot_tree_runner_width_uses_tree_pipeline_contract() -> None:
    spec = _make_spec(
        num_speculative_tokens=4,
        pivot_proposal_engine="eagle3_head",
        pivot_verification_pipeline="intermediate_tree_then_target_tree",
        pivot_hidden_state_source="intermediate",
        pivot_use_eagle_tree=True,
        pivot_spechive_num_rounds=3,
    )
    assert spec.runner_num_speculative_tokens() == 4 + 3 * (4 + 1)


def test_pivot_draft_engine_uses_model_based_gpu_tokens() -> None:
    spec = _make_spec(
        pivot_proposal_engine="draft_model",
        pivot_verification_pipeline="target_only",
    )
    assert spec.uses_model_based_drafter()
    assert spec.uses_gpu_sampled_tokens_for_drafting()


def test_pivot_parallel_drafting_slot_math_matches_existing_rule() -> None:
    spec = _make_spec(num_speculative_tokens=6, parallel_drafting=True)
    # Existing rule: (num_spec_tokens - 1) masked slots + one draft-model slot.
    assert spec.max_num_new_slots_for_drafting == 6


def test_pivot_requires_draft_vocab_match_with_target() -> None:
    spec = _make_spec(draft_vocab=12345)
    with pytest.raises(ValueError, match="Target and draft model should have the same"):
        spec.verify_equal_vocab_size_if_draft_model()


def test_pivot_allows_intermediate_vocab_mismatch_when_shared_head_declared() -> None:
    spec = _make_spec(
        intermediate_vocab=12345,
        intermediate_model_type="eagle",
        intermediate_arch=["EagleLlamaForCausalLM"],
    )
    # Should not raise: intermediate declares shared output semantics.
    spec.verify_equal_vocab_size_if_draft_model()


def test_pivot_rejects_intermediate_vocab_mismatch_without_shared_head() -> None:
    spec = _make_spec(
        intermediate_vocab=12345,
        intermediate_model_type="llama",
        intermediate_arch=["LlamaForCausalLM"],
    )
    with pytest.raises(ValueError, match="pivot requires intermediate output compatibility"):
        spec.verify_equal_vocab_size_if_draft_model()


def test_pivot_tree_mode_requires_eagle3_head() -> None:
    spec = _make_spec(
        pivot_proposal_engine="draft_model",
        pivot_verification_pipeline="target_only",
        pivot_use_eagle_tree=True,
    )
    with pytest.raises(ValueError, match="pivot_use_eagle_tree=True requires"):
        spec._verify_args()
