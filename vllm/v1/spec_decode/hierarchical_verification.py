# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone hierarchical verification: D proposes L, I verifies, full batch each round."""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.adaptive_cascade import (
    IntermediateDraftModelProposer,
    _collect_emitted_probs_from_sampled,
    _propose_chunk_from_prefix,
    _vllm_as_plain_draft,
    _vllm_hierarchical_verification_chunk,
)
from vllm.v1.spec_decode.draft_model import PinnedDraftNamespaceDraftModelProposer
from vllm.v1.spec_decode.spec_stage_ops import run_verify_stage
from vllm.v1.spec_decode.spec_stage_runtime import (
    DitRoundDecision,
    DitRoundProposal,
    DitRoundVerification,
    HybridProposalBundle,
)
def hv_run_inter_verification_acceptance(
    *,
    proposal: DitRoundProposal,
    verification: DitRoundVerification,
    sampling_metadata: SamplingMetadata,
    rejection_sampler,
    vocab_size: int,
    use_draft_probs: bool,
) -> DitRoundDecision:
    """Token-level intermediate acceptance via ``run_verify_stage``."""
    batch_size = int(proposal.tokens.shape[0])
    chunk_len = int(proposal.tokens.shape[1])
    draft_flat = proposal.tokens.reshape(-1).to(torch.int32)
    num_draft_tokens = [chunk_len] * batch_size
    cu_num_draft_tokens = torch.cumsum(
        torch.tensor(num_draft_tokens, dtype=torch.int32, device=draft_flat.device),
        dim=0,
    ).to(torch.int32)
    draft_probs_flat = (
        proposal.probs.reshape(-1, vocab_size)
        if proposal.probs is not None
        else None
    )
    emitted_rows, stage_out, target_probs_flat, bonus_probs = run_verify_stage(
        rejection_sampler,
        draft_token_ids_flat=draft_flat,
        draft_probs_flat=draft_probs_flat,
        num_draft_tokens=num_draft_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        max_spec_len=chunk_len,
        verifier_logits_flat=verification.logits_flat,
        bonus_logits=verification.bonus_logits,
        sampling_metadata=sampling_metadata,
        vocab_size=vocab_size,
        include_processed_probs=use_draft_probs,
    )
    emitted_prob_rows: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    if use_draft_probs and target_probs_flat.numel() > 0:
        per_req_probs = target_probs_flat.view(batch_size, chunk_len, vocab_size)
        emitted_prob_rows = _collect_emitted_probs_from_sampled(
            stage_out.sampled_token_ids,
            per_req_probs,
            bonus_probs,
            vocab_size=vocab_size,
        )
    return DitRoundDecision(
        emitted_rows=emitted_rows,
        emitted_prob_rows=emitted_prob_rows,
        accepted_lens=[len(r) for r in emitted_rows],
    )


class HierarchicalVerificationProposer:
    """Full-batch hierarchical D→I rounds; single authoritative target verify on runner."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: object | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner
        spec = vllm_config.speculative_config
        assert spec is not None
        self.l = spec.hv_chunk_len()
        self.r = int(spec.num_hv_rounds)
        self.k = spec.hv_max_spec_len()
        self.draft = PinnedDraftNamespaceDraftModelProposer(
            _vllm_as_plain_draft(vllm_config), device, runner
        )
        self.inter = IntermediateDraftModelProposer(
            _vllm_hierarchical_verification_chunk(
                vllm_config, length=self.l, intermediate=True
            ),
            device,
            runner,
        )
        self._pending_hybrid_bundle: HybridProposalBundle | None = None

    def __getattr__(self, name: str):
        """Delegate draft-slot bookkeeping (e.g. ``prepare_next_token_ids_padded``)."""
        if name.startswith("_") or name in (
            "propose",
            "load_model",
            "model",
            "intermediate_model",
            "initialize_attn_backend",
            "initialize_cudagraph_keys",
            "dummy_run",
            "validate_same_kv_cache_group",
            "clear_draft_probs",
            "propose_chunk_from_prefix",
            "verify_chunk_with_inter_verifier",
            "run_inter_verification_acceptance",
            "take_pending_hv_bundle",
            "discard_pending_hv_state",
            "take_pending_spechive_bundle",
            "discard_pending_hierarchical_verification_state",
            "kv_cache_gid",
            "supports_mm_inputs",
            "draft",
            "inter",
            "l",
            "r",
            "k",
            "vllm_config",
            "device",
            "runner",
        ):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        return getattr(self.draft, name)

    @property
    def model(self) -> nn.Module:
        return self.draft.model

    @property
    def intermediate_model(self) -> nn.Module | None:
        return self.inter.model

    @property
    def kv_cache_gid(self) -> int | None:
        return getattr(self.draft, "kv_cache_gid", None)

    @property
    def supports_mm_inputs(self) -> bool:
        return bool(getattr(self.draft, "supports_mm_inputs", False))

    def load_model(self, target_model: nn.Module) -> None:
        self.draft.load_model(target_model)
        self.inter.load_model(target_model)

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        self.draft.initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        self.inter.initialize_attn_backend(kv_cache_config, kernel_block_sizes)

    def initialize_cudagraph_keys(self, cudagraph_mode) -> None:
        self.draft.initialize_cudagraph_keys(cudagraph_mode)
        self.inter.initialize_cudagraph_keys(cudagraph_mode)

    def dummy_run(self, *args, **kwargs):
        self.draft.dummy_run(*args, **kwargs)
        self.inter.dummy_run(*args, **kwargs)

    def validate_same_kv_cache_group(self, kv_cache_config) -> None:
        self.draft.validate_same_kv_cache_group(kv_cache_config)
        self.inter.validate_same_kv_cache_group(kv_cache_config)

    def clear_draft_probs(self) -> None:
        if hasattr(self.draft, "clear_draft_probs"):
            self.draft.clear_draft_probs()

    def propose_chunk_from_prefix(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        chunk_len: int,
        sampling_metadata: SamplingMetadata,
        use_draft_probs: bool,
    ) -> DitRoundProposal:
        tokens, probs = _propose_chunk_from_prefix(
            self.draft,
            cad=base_common_attn_metadata,
            target_token_ids=base_target_token_ids,
            target_positions=base_target_positions,
            target_hidden_states=base_target_hidden_states,
            next_token_ids=base_next_token_ids,
            num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            chunk_len=chunk_len,
            sampling_metadata=sampling_metadata,
            use_draft_probs=use_draft_probs,
        )
        return DitRoundProposal(tokens=tokens, probs=probs)

    def verify_chunk_with_inter_verifier(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        candidate_tokens: torch.Tensor,
        mirror_kv_common_attn_metadata: CommonAttentionMetadata | None = None,
    ) -> DitRoundVerification:
        if mirror_kv_common_attn_metadata is not None:
            raise ValueError(
                "standalone hierarchical_verification does not use mirror_kv CAD"
            )
        assert self.runner is not None
        hv_pre = self.runner._prepare_hv_step(
            self.inter,
            base_common_attn_metadata,
            target_token_ids=base_target_token_ids,
            target_positions=base_target_positions,
            target_hidden_states=base_target_hidden_states,
            next_token_ids=base_next_token_ids,
            num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            candidate_tokens=candidate_tokens.to(torch.int32),
        )
        return self.runner._run_hv_verify_step(
            self.inter,
            hv_pre,
            num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
        )

    def run_inter_verification_acceptance(
        self,
        *,
        proposal: DitRoundProposal,
        verification: DitRoundVerification,
        sampling_metadata: SamplingMetadata,
        rejection_sampler,
        vocab_size: int,
        use_draft_probs: bool,
    ) -> DitRoundDecision:
        return hv_run_inter_verification_acceptance(
            proposal=proposal,
            verification=verification,
            sampling_metadata=sampling_metadata,
            rejection_sampler=rejection_sampler,
            vocab_size=vocab_size,
            use_draft_probs=use_draft_probs,
        )

    def take_pending_hv_bundle(self) -> HybridProposalBundle | None:
        b = self._pending_hybrid_bundle
        self._pending_hybrid_bundle = None
        return b

    def discard_pending_hv_state(self) -> None:
        self._pending_hybrid_bundle = None

    def take_pending_spechive_bundle(self) -> HybridProposalBundle | None:
        return self.take_pending_hv_bundle()

    def discard_pending_hierarchical_verification_state(self) -> None:
        self.discard_pending_hv_state()

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
        spec_decode_common_attn_metadata_by_gid: (
            dict[int, CommonAttentionMetadata] | None
        ) = None,
    ) -> torch.Tensor:
        if mm_embed_inputs is not None:
            raise NotImplementedError(
                "hierarchical_verification does not support multimodal draft inputs yet."
            )
        assert self.runner is not None
        out, bundle = self.runner.run_hv_rounds(
            drafter=self,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            spec_decode_cad_by_gid=spec_decode_common_attn_metadata_by_gid,
        )
        self._pending_hybrid_bundle = bundle
        self.clear_draft_probs()
        return out
