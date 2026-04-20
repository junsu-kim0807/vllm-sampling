# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for standalone ``method=hierarchical_verification`` speculative decoding."""

import torch

from vllm.config import SpeculativeConfig
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.hierarchical_verification import HierarchicalVerificationProposer
from vllm.v1.spec_decode.hybrid_bundle_utils import _build_hybrid_bundle_from_rows
from vllm.v1.spec_decode.spec_stage_runtime import HybridProposalBundle


def _hv_spec(*, L: int = 4, R: int = 2) -> SpeculativeConfig:
    """Minimal ``SpeculativeConfig`` instance for HV numeric helpers."""
    s = object.__new__(SpeculativeConfig)
    s.method = "hierarchical_verification"
    s.num_speculative_tokens = L
    s.num_hv_rounds = R
    return s


def test_hv_max_spec_len_and_runner_widths() -> None:
    s = _hv_spec(L=4, R=2)
    assert s.hv_chunk_len() == 4
    assert s.hv_max_spec_len() == 2 * (4 + 1) + 4
    assert s.runner_num_speculative_tokens() == s.hv_max_spec_len()
    assert s.runner_num_partial_speculative_tokens() == 2 * 4


def test_hv_r1_l4() -> None:
    s = _hv_spec(L=4, R=1)
    assert s.hv_max_spec_len() == 1 * (4 + 1) + 4
    assert s.runner_num_partial_speculative_tokens() == 4


def test_uses_draft_model_and_model_based() -> None:
    s = _hv_spec()
    assert s.uses_draft_model()
    assert s.uses_model_based_drafter()
    assert s.uses_gpu_sampled_tokens_for_drafting()


def test_build_hybrid_bundle_max_spec_len_matches_pad_width() -> None:
    B, K_final = 2, 13
    rows = torch.full((B, K_final), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    rows[0, :5] = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    rows[1, :3] = torch.tensor([10, 11, 12], dtype=torch.int32)
    bundle = _build_hybrid_bundle_from_rows(
        rows,
        mode="hierarchical_verification",
        draft_probs=None,
    )
    assert bundle.max_spec_len == K_final
    assert bundle.num_draft_tokens == [5, 3]
    assert int(bundle.cu_num_draft_tokens[-1].item()) == 8


def test_take_discard_pending_hv_bundle() -> None:
    p = object.__new__(HierarchicalVerificationProposer)
    p._pending_hybrid_bundle = None
    assert p.take_pending_hv_bundle() is None
    assert p.take_pending_spechive_bundle() is None
    stub = HybridProposalBundle(
        draft_token_ids=torch.tensor([1], dtype=torch.int32),
        draft_probs=None,
        num_draft_tokens=[1],
        cu_num_draft_tokens=torch.tensor([1], dtype=torch.int32),
        max_spec_len=1,
        mode="hierarchical_verification",
    )
    p._pending_hybrid_bundle = stub
    assert p.take_pending_hv_bundle() is stub
    assert p._pending_hybrid_bundle is None
    p._pending_hybrid_bundle = stub
    p.discard_pending_hv_state()
    assert p._pending_hybrid_bundle is None
    p._pending_hybrid_bundle = stub
    p.discard_pending_hierarchical_verification_state()
    assert p._pending_hybrid_bundle is None
