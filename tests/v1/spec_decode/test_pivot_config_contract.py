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
    return spec


def test_pivot_slot_budget_follows_verified_width_contract() -> None:
    spec = _make_spec(num_speculative_tokens=7, parallel_drafting=False)
    assert spec.uses_draft_model()
    # One extra slot per request follows draft-model contract, independent of
    # internal I/D staging.
    assert spec.max_num_new_slots_for_drafting == 1
    assert spec.runner_num_speculative_tokens() == 7


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
