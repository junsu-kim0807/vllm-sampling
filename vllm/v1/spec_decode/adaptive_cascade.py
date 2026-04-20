# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive spechive speculative decoding: draft (D), intermediate (I), target (T).

``draft_target``: draft→target. ``inter_verification``: intermediate as draft→target.

``hierarchical_verification``: run multi-round D/I verification in round orchestration.
Each round drafts a chunk of ``L=num_speculative_tokens`` and
intermediate-verifies it into a logical emitted prefix. Right before final
bundle construction, DIT performs one additional draft-only tail proposal and
lets the standard target rejection-sampling path verify the whole bundle in one
shot. The runner keeps the default outer contract where each engine iteration
has one authoritative target verification.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig, get_layers_from_vllm_config, replace
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
)
from vllm.v1.spec_decode.draft_model import (
    DraftModelProposer,
    PinnedDraftNamespaceDraftModelProposer,
)
from vllm.v1.spec_decode.eagle import EagleProposer, SpecDecodeBaseProposer
from vllm.v1.spec_decode.spec_stage_ops import (
    run_verify_stage,
)
from vllm.v1.spec_decode.spec_stage_runtime import (
    DitRoundDecision,
    DitRoundProposal,
    DitRoundVerification,
    HybridProposalBundle,
    IntermediateRoundState,
    StagedHiddenStateBundle,
)
from vllm.v1.spec_decode.hybrid_bundle_utils import (
    _build_hybrid_bundle_from_rows,
    _flatten_prob_rows_for_output,
)
from vllm.v1.spec_decode.hv_step_packing import (
    build_prefix_conditioned_inputs as _hv_build_prefix_conditioned_inputs,
    slice_hv_verification_logits,
)
from vllm.v1.spec_decode.spec_stage_utils import clone_common_attn_metadata
from vllm.v1.spec_decode.utils import (
    create_vllm_config_for_draft_model,
)

logger = init_logger(__name__)


def _spechive_debug_enabled() -> bool:
    return (
        os.environ.get("VLLM_SPEC_SPECHIVE_DEBUG", "0") == "1"
        or os.environ.get("VLLM_SPEC_DIT_DEBUG", "0") == "1"
    )


def _spechive_debug_assert(cond: bool, code: str, *, detail: str = "") -> bool:
    if _spechive_debug_enabled() and not cond:
        logger.warning("SPECHIVE_DEBUG check failed: %s (%s)", code, detail)
    return cond


def _chain_spec_token_tree(num_tokens: int) -> str:
    return str([(i + 1) * (0,) for i in range(num_tokens)])


def _vllm_as_plain_draft(base: VllmConfig) -> VllmConfig:
    spec = base.speculative_config
    assert spec is not None
    new_spec = replace(spec, method="draft_model")
    return replace(base, speculative_config=new_spec)


def _vllm_intermediate_as_draft(base: VllmConfig) -> VllmConfig:
    spec = base.speculative_config
    assert spec is not None
    assert spec.intermediate_model_config is not None
    assert spec.intermediate_parallel_config is not None
    assert spec.intermediate_tensor_parallel_size is not None
    new_spec = replace(
        spec,
        method="draft_model",
        draft_model_config=spec.intermediate_model_config,
        draft_parallel_config=spec.intermediate_parallel_config,
        draft_tensor_parallel_size=spec.intermediate_tensor_parallel_size,
    )
    return replace(base, speculative_config=new_spec)


def _vllm_hierarchical_verification_chunk(
    base: VllmConfig, *, length: int, intermediate: bool
) -> VllmConfig:
    spec = base.speculative_config
    assert spec is not None
    tree = _chain_spec_token_tree(length)
    if intermediate:
        assert spec.intermediate_model_config is not None
        assert spec.intermediate_parallel_config is not None
        assert spec.intermediate_tensor_parallel_size is not None
        new_spec = replace(
            spec,
            method="draft_model",
            draft_model_config=spec.intermediate_model_config,
            draft_parallel_config=spec.intermediate_parallel_config,
            draft_tensor_parallel_size=spec.intermediate_tensor_parallel_size,
            num_speculative_tokens=length,
            speculative_token_tree=tree,
        )
    else:
        new_spec = replace(
            spec,
            method="draft_model",
            num_speculative_tokens=length,
            speculative_token_tree=tree,
        )
    return replace(base, speculative_config=new_spec)


def _vllm_hierarchical_eagle_chunk(
    base: VllmConfig, *, length: int, intermediate: bool
) -> VllmConfig:
    """VllmConfig slice for hierarchical spechive with Eagle3 draft (D) or I draft."""
    spec = base.speculative_config
    assert spec is not None
    tree = _chain_spec_token_tree(length)
    if intermediate:
        assert spec.intermediate_model_config is not None
        assert spec.intermediate_parallel_config is not None
        assert spec.intermediate_tensor_parallel_size is not None
        new_spec = replace(
            spec,
            method="draft_model",
            draft_model_config=spec.intermediate_model_config,
            draft_parallel_config=spec.intermediate_parallel_config,
            draft_tensor_parallel_size=spec.intermediate_tensor_parallel_size,
            num_speculative_tokens=length,
            speculative_token_tree=tree,
        )
    else:
        new_spec = replace(
            spec,
            method="eagle3",
            num_speculative_tokens=length,
            speculative_token_tree=tree,
            prompt_lookup_min=1,
            prompt_lookup_max=1,
        )
    return replace(base, speculative_config=new_spec)


def _hv_clone_cad(cad: CommonAttentionMetadata) -> CommonAttentionMetadata:
    return clone_common_attn_metadata(cad)


class IntermediateDraftModelProposer(DraftModelProposer):
    """Loads the intermediate verifier weights under a distinct module prefix."""

    _INTERMEDIATE_PREFIX = "intermediate_model."

    @override
    def _get_model(self) -> nn.Module:
        from vllm.compilation.backends import set_model_tag

        temp_vllm_config = create_vllm_config_for_draft_model(
            self.vllm_config,
            compile_cache_namespace="intermediate_model",
        )
        with set_model_tag("intermediate_model"):
            model = get_model(
                vllm_config=temp_vllm_config,
                prefix="intermediate_model",
            )
        return model

    @override
    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        intermediate_attn_layer_names = {
            name
            for name in all_attn_layers.keys()
            if name.startswith(self._INTERMEDIATE_PREFIX)
        }
        if not intermediate_attn_layer_names:
            sample = sorted(all_attn_layers.keys())[:16]
            raise RuntimeError(
                "Failed to discover intermediate_model attention layers in the "
                "static forward context after loading the intermediate verifier. "
                f"sample_layer_names={sample}"
            )
        self._draft_attn_layer_names = intermediate_attn_layer_names

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config,
        kernel_block_sizes=None,
    ) -> None:
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        discovered = {
            layer_name
            for group in self.draft_attn_groups
            for layer_name in group.layer_names
        }
        missing = self._draft_attn_layer_names - discovered
        if missing:
            raise RuntimeError(
                "Intermediate verifier KV/attention backend initialization missed "
                "layers bound in load_model() "
                f"(sample): {sorted(missing)[:32]}"
            )


def _build_prefix_conditioned_inputs(
    proposer: SpecDecodeBaseProposer,
    *,
    cad: CommonAttentionMetadata,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
    prefix_rows: list[list[int]],
    roll_rows: list[list[int]] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    CommonAttentionMetadata,
    list[int],
    list[int],
    list[int],
]:
    """Build proposer inputs conditioned on logical prefix rows per request."""
    block_size = proposer.draft_attn_groups[0].kv_cache_spec.block_size
    return _hv_build_prefix_conditioned_inputs(
        cad=cad,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        prefix_rows=prefix_rows,
        roll_rows=roll_rows,
        block_size=block_size,
    )


def _canonicalize_reused_prefix_frontier(
    proposer: SpecDecodeBaseProposer,
    *,
    cad: CommonAttentionMetadata,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
) -> tuple[
    CommonAttentionMetadata,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Normalize reused prefix-prefab inputs back to the pre-SIFP layout.

    Reused staged prefabs must present a frontier where token/position/hidden
    tensors and CAD all describe the same flat query layout. When an older or
    partially patched producer leaks a post-`set_inputs_first_pass` CAD/hidden
    pair together with pre-SIFP tokens/positions, rebuild the CAD/hidden view
    expected by `_build_prefix_conditioned_inputs`.
    """
    tok_n = int(target_token_ids.shape[0])
    pos_n = (
        int(target_positions.shape[0])
        if target_positions.dim() == 1
        else int(target_positions.shape[1])
    )
    hid_n = int(target_hidden_states.shape[0])
    qsl_n = int(cad.query_start_loc[-1].item())
    if qsl_n == tok_n and pos_n == tok_n and hid_n == tok_n:
        return cad, target_token_ids, target_positions, target_hidden_states, next_token_ids

    extra_slots = int(getattr(proposer, "net_num_new_slots_per_request", 0))
    batch_size = int(cad.batch_size())
    expected_post = tok_n + batch_size * extra_slots
    if (
        extra_slots <= 0
        or batch_size <= 0
        or qsl_n != expected_post
        or hid_n != expected_post
        or pos_n != tok_n
    ):
        return cad, target_token_ids, target_positions, target_hidden_states, next_token_ids

    qsl = cad.query_start_loc
    offsets = extra_slots * torch.arange(
        len(qsl), dtype=qsl.dtype, device=qsl.device
    )
    pre_qsl = qsl - offsets
    if int(pre_qsl[-1].item()) != tok_n:
        return cad, target_token_ids, target_positions, target_hidden_states, next_token_ids

    hidden_pieces: list[torch.Tensor] = []
    slot_pieces: list[torch.Tensor] = []
    max_query_len = 1
    for b in range(batch_size):
        dst_start = int(pre_qsl[b].item())
        dst_end = int(pre_qsl[b + 1].item())
        orig_len = dst_end - dst_start
        src_start = int(qsl[b].item())
        src_end = src_start + orig_len
        hidden_pieces.append(target_hidden_states[src_start:src_end])
        slot_pieces.append(cad.slot_mapping[src_start:src_end])
        max_query_len = max(max_query_len, orig_len)

    pre_hidden = torch.cat(hidden_pieces, dim=0)
    pre_slot = torch.cat(slot_pieces, dim=0)
    pre_seq_lens = cad.seq_lens - extra_slots
    pre_cad = cad.replace(
        query_start_loc=pre_qsl,
        query_start_loc_cpu=pre_qsl.detach().cpu(),
        seq_lens=pre_seq_lens,
        num_actual_tokens=tok_n,
        max_query_len=max_query_len,
        max_seq_len=int(pre_seq_lens.max().item()),
        slot_mapping=pre_slot,
        _seq_lens_cpu=pre_seq_lens.detach().cpu()
        if cad._seq_lens_cpu is not None
        else None,
        _num_computed_tokens_cpu=None,
        _num_computed_tokens_cache=None,
    )
    _spechive_debug_assert(
        int(pre_cad.query_start_loc[-1].item()) == tok_n
        and int(pre_hidden.shape[0]) == tok_n,
        "check_reused_prefab_frontier_canonicalization",
        detail=(
            f"tok_n={tok_n}, pos_n={pos_n}, hid_n={hid_n}, "
            f"post_qsl_end={qsl_n}, extra_slots={extra_slots}"
        ),
    )
    return pre_cad, target_token_ids, target_positions, pre_hidden, next_token_ids


def _forward_prefix_conditioned_logits(
    proposer: SpecDecodeBaseProposer,
    *,
    cad: CommonAttentionMetadata,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
    num_rejected_tokens_gpu: torch.Tensor | None,
    prefix_rows: list[list[int]],
    roll_rows: list[list[int]] | None = None,
) -> tuple[torch.Tensor, CommonAttentionMetadata, list[int], list[int], list[int]]:
    """Run one prefix-conditioned forward and return query-row logits."""
    (
        cad,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
    ) = _canonicalize_reused_prefix_frontier(
        proposer,
        cad=cad,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
    )
    (
        pref_toks,
        pref_pos,
        pref_hidden,
        pref_next,
        pref_cad,
        base_query_lens,
        prefix_lens,
        roll_lens,
    ) = _build_prefix_conditioned_inputs(
        proposer,
        cad=_hv_clone_cad(cad),
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        prefix_rows=prefix_rows,
        roll_rows=roll_rows,
    )
    num_tokens, token_indices_to_sample, pref_cad = proposer.set_inputs_first_pass(
        target_token_ids=pref_toks,
        next_token_ids=pref_next,
        target_positions=pref_pos,
        target_hidden_states=pref_hidden,
        token_indices_to_sample=None,
        cad=pref_cad,
        num_rejected_tokens_gpu=num_rejected_tokens_gpu,
    )
    assert token_indices_to_sample is not None

    per_layer_attn_metadata: dict[str, object] = {}
    attn_metadata = None
    for attn_group in proposer.draft_attn_groups:
        attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
            common_attn_metadata=pref_cad,
            draft_index=0,
        )
        for layer_name in attn_group.layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

    proposer._check_per_layer_attn_metadata_contract(per_layer_attn_metadata)

    if proposer.allowed_attn_types is not None and not isinstance(
        attn_metadata, proposer.allowed_attn_types
    ):
        raise ValueError(
            "adaptive_spechive hierarchical verification: unsupported attention metadata type "
            f"{type(attn_metadata)}; allowed: {proposer.allowed_attn_types}"
        )

    cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
        proposer._determine_batch_execution_and_padding(num_tokens)
    )
    if proposer.supports_mm_inputs:
        proposer.inputs_embeds[:num_tokens] = proposer.model.embed_input_ids(
            proposer.input_ids[:num_tokens],
        )
        input_ids = None
        inputs_embeds = proposer.inputs_embeds[:num_input_tokens]
    else:
        input_ids = proposer.input_ids[:num_input_tokens]
        inputs_embeds = None

    model_kwargs = {
        "input_ids": input_ids,
        "positions": proposer._get_positions(num_input_tokens),
        "inputs_embeds": inputs_embeds,
    }
    if proposer.pass_hidden_states_to_model:
        model_kwargs["hidden_states"] = proposer.hidden_states[:num_input_tokens]

    with set_forward_context(
        per_layer_attn_metadata,
        proposer.vllm_config,
        num_tokens=num_input_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        slot_mapping=proposer._get_slot_mapping(num_input_tokens, pref_cad.slot_mapping),
    ):
        ret_hidden_states = proposer.model(**model_kwargs)
        if proposer.model_returns_tuple():
            last_hidden_states, _ = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states
    all_logits = proposer.model.compute_logits(last_hidden_states).to(torch.float32)
    expected_query_rows = sum(
        base_query_lens[b] + prefix_lens[b] + roll_lens[b]
        for b in range(len(base_query_lens))
    )
    _spechive_debug_assert(
        int(num_tokens) == int(expected_query_rows),
        "check_hv_verify_total_query_rows",
        detail=f"num_tokens={num_tokens}, expected={expected_query_rows}",
    )
    _spechive_debug_assert(
        int(all_logits.shape[0]) >= int(num_tokens),
        "check_hv_verify_logits_rows_cover_queries",
        detail=f"logits_rows={all_logits.shape[0]}, num_tokens={num_tokens}",
    )
    return all_logits, pref_cad, base_query_lens, prefix_lens, roll_lens


def _propose_chunk_from_prefix(
    proposer: SpecDecodeBaseProposer,
    *,
    cad: CommonAttentionMetadata,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
    num_rejected_tokens_gpu: torch.Tensor | None,
    prefix_rows: list[list[int]],
    chunk_len: int,
    sampling_metadata: SamplingMetadata,
    use_draft_probs: bool,
    prefix_prefab: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CommonAttentionMetadata
    ]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample one chunk from one proposer call conditioned on prefix rows."""
    if prefix_prefab is not None:
        pref_toks, pref_pos, pref_hidden, pref_next, pref_cad = prefix_prefab
        pref_cad = _hv_clone_cad(pref_cad)
        (
            pref_cad,
            pref_toks,
            pref_pos,
            pref_hidden,
            pref_next,
        ) = _canonicalize_reused_prefix_frontier(
            proposer,
            cad=pref_cad,
            target_token_ids=pref_toks,
            target_positions=pref_pos,
            target_hidden_states=pref_hidden,
            next_token_ids=pref_next,
        )
    else:
        (
            pref_toks,
            pref_pos,
            pref_hidden,
            pref_next,
            pref_cad,
            _,
            _,
            _,
        ) = _build_prefix_conditioned_inputs(
            proposer,
            cad=_hv_clone_cad(cad),
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            prefix_rows=prefix_rows,
            roll_rows=None,
        )
    rows = proposer.propose(
        target_token_ids=pref_toks,
        target_positions=pref_pos,
        target_hidden_states=pref_hidden,
        next_token_ids=pref_next,
        token_indices_to_sample=None,
        common_attn_metadata=pref_cad,
        sampling_metadata=sampling_metadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        slot_mappings=None,
    ).to(torch.int32)
    rows = rows[:, :chunk_len]
    out_probs = None
    if use_draft_probs:
        probs_flat = getattr(proposer, "last_draft_probs_flat", None)
        if probs_flat is not None and probs_flat.numel() > 0:
            bsz = rows.shape[0]
            out_probs = probs_flat.view(bsz, -1, probs_flat.shape[-1])[:, : rows.shape[1]].to(
                torch.float32
            )
    return rows, out_probs


def _verify_chunk_with_prefix(
    inter: SpecDecodeBaseProposer,
    *,
    cad: CommonAttentionMetadata,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
    num_rejected_tokens_gpu: torch.Tensor | None,
    prefix_rows: list[list[int]],
    draft_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect verifier logits for full chunk from one prefix-conditioned forward."""
    roll_rows = [
        [int(tok) for tok in draft_tokens[b].tolist()]
        for b in range(draft_tokens.shape[0])
    ]
    all_logits, pref_cad, base_query_lens, prefix_lens, roll_lens = (
        _forward_prefix_conditioned_logits(
            inter,
            cad=cad,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            roll_rows=roll_rows,
        )
    )
    bsz = draft_tokens.shape[0]
    chunk_len = draft_tokens.shape[1]
    logits_flat, bonus_logits = slice_hv_verification_logits(
        all_logits,
        pref_cad,
        draft_tokens,
        base_query_lens,
        prefix_lens,
        roll_lens,
    )
    if _spechive_debug_enabled() and bsz <= 2 and chunk_len <= 4:
        per_row = logits_flat.view(bsz, chunk_len, -1)
        ref_steps: list[list[torch.Tensor]] = [[] for _ in range(bsz)]
        for j in range(chunk_len):
            partial_roll = [
                [int(tok) for tok in draft_tokens[b, :j].tolist()] for b in range(bsz)
            ]
            ref_logits, ref_cad, ref_base, ref_prefix, ref_roll = _forward_prefix_conditioned_logits(
                inter,
                cad=cad,
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                prefix_rows=prefix_rows,
                roll_rows=partial_roll,
            )
            ref_qsl = ref_cad.query_start_loc
            for b in range(bsz):
                ref_chain_len = ref_prefix[b] + ref_roll[b] + 1
                ref_start = int(ref_qsl[b].item()) + ref_base[b] - 1
                ref_idx = torch.arange(
                    ref_start,
                    ref_start + ref_chain_len,
                    dtype=torch.long,
                    device=ref_logits.device,
                )
                ref_chain = ref_logits[ref_idx]
                ref_steps[b].append(ref_chain[ref_prefix[b]])
        for b in range(bsz):
            for j in range(chunk_len):
                _spechive_debug_assert(
                    torch.allclose(
                        per_row[b, j], ref_steps[b][j], atol=1e-5, rtol=1e-4
                    ),
                    "check_hv_verify_slice_reference_allclose",
                    detail=f"req={b}, step={j}",
                )
    return logits_flat, bonus_logits


def verify_intermediate_chunk_with_prefix_prefab(
    inter: SpecDecodeBaseProposer,
    *,
    prefix_prefab: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CommonAttentionMetadata
    ],
    base_num_rejected_tokens_gpu: torch.Tensor | None,
    draft_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intermediate chunk verification using a reused prefix frontier (no prefix rebuild).

    Shared by adaptive spechive and pivot staged verification paths.
    """
    pt, pp, ph, pn, pcad = prefix_prefab
    pcad = _hv_clone_cad(pcad)
    empty_prefix = [[] for _ in range(int(pcad.batch_size()))]
    return _verify_chunk_with_prefix(
        inter,
        cad=pcad,
        target_token_ids=pt,
        target_positions=pp,
        target_hidden_states=ph,
        next_token_ids=pn,
        num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
        prefix_rows=empty_prefix,
        draft_tokens=draft_tokens.to(torch.int32),
    )


def _collect_emitted_probs_from_sampled(
    sampled: torch.Tensor,
    per_step_probs: torch.Tensor,
    bonus_probs: torch.Tensor,
    *,
    vocab_size: int,
) -> list[list[torch.Tensor]]:
    """Collect I-equivalent proposal probs for emitted tokens from sampled matrix."""
    bsz, Lp1 = sampled.shape
    out: list[list[torch.Tensor]] = [[] for _ in range(bsz)]
    for b in range(bsz):
        for c in range(Lp1):
            tok = int(sampled[b, c].item())
            if tok == PLACEHOLDER_TOKEN_ID or tok >= vocab_size:
                continue
            if c < per_step_probs.shape[1]:
                out[b].append(per_step_probs[b, c])
            else:
                out[b].append(bonus_probs[b])
    return out


class AdaptiveSpechiveProposer:
    """Routes to draft-model proposers for spechive stages."""

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
        self._mode = spec.adaptive_spechive_mode
        self._L = spec.num_speculative_tokens

        self._delegate: DraftModelProposer | EagleProposer
        self._inter_dit: IntermediateDraftModelProposer | None = None
        self._pending_hybrid_bundle: HybridProposalBundle | None = None
        self._use_staged_eagle = False
        self._staged_delegates: object | None = None

        if self._mode == "draft_target":
            self._delegate = PinnedDraftNamespaceDraftModelProposer(
                _vllm_as_plain_draft(vllm_config), device, runner
            )
        elif self._mode == "inter_verification":
            self._delegate = PinnedDraftNamespaceDraftModelProposer(
                _vllm_intermediate_as_draft(vllm_config), device, runner
            )
        else:
            assert self._mode == "hierarchical_verification"
            self._inter_dit = IntermediateDraftModelProposer(
                _vllm_hierarchical_verification_chunk(
                    vllm_config, length=self._L, intermediate=True
                ),
                device,
                runner,
            )
            if spec.adaptive_spechive_draft_uses_eagle3_head():
                from vllm.v1.spec_decode.staged_delegate_factory import (
                    build_pivot_staged_delegates,
                )

                self._delegate = EagleProposer(
                    _vllm_hierarchical_eagle_chunk(
                        vllm_config, length=self._L, intermediate=False
                    ),
                    device,
                    runner,
                )
                self._staged_delegates = build_pivot_staged_delegates(
                    proposal_engine_kind="eagle3_head",
                    draft_delegate=None,
                    eagle_delegate=self._delegate,
                    intermediate_delegate=self._inter_dit,
                    needs_intermediate_provider=True,
                )
                self._use_staged_eagle = True
            else:
                self._delegate = PinnedDraftNamespaceDraftModelProposer(
                    _vllm_hierarchical_verification_chunk(
                        vllm_config, length=self._L, intermediate=False
                    ),
                    device,
                    runner,
                )

    def __getattr__(self, name: str):
        if name.startswith("_") or name in (
            "propose",
            "load_model",
            "model",
            "intermediate_model",
            "initialize_attn_backend",
            "initialize_cudagraph_keys",
            "dummy_run",
            "validate_same_kv_cache_group",
        ):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        return getattr(self._delegate, name)

    @property
    def model(self) -> nn.Module:
        return self._delegate.model

    @property
    def intermediate_model(self) -> nn.Module | None:
        if self._inter_dit is None:
            return None
        return self._inter_dit.model

    def load_model(self, target_model: nn.Module) -> None:
        self._delegate.load_model(target_model)
        if self._inter_dit is not None:
            self._inter_dit.load_model(target_model)

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        self._delegate.initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        if self._inter_dit is not None:
            self._inter_dit.initialize_attn_backend(kv_cache_config, kernel_block_sizes)

    def initialize_cudagraph_keys(self, cudagraph_mode) -> None:
        self._delegate.initialize_cudagraph_keys(cudagraph_mode)
        if self._inter_dit is not None:
            self._inter_dit.initialize_cudagraph_keys(cudagraph_mode)

    def dummy_run(self, *args, **kwargs):
        self._delegate.dummy_run(*args, **kwargs)
        if self._inter_dit is not None:
            self._inter_dit.dummy_run(*args, **kwargs)

    def validate_same_kv_cache_group(self, kv_cache_config) -> None:
        self._delegate.validate_same_kv_cache_group(kv_cache_config)
        if self._inter_dit is not None:
            self._inter_dit.validate_same_kv_cache_group(kv_cache_config)

    def clear_draft_probs(self) -> None:
        if hasattr(self._delegate, "clear_draft_probs"):
            self._delegate.clear_draft_probs()

    def supports_staged_eagle_fastpath(self) -> bool:
        return self._use_staged_eagle and self._staged_delegates is not None

    def bootstrap_intermediate_round_state(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
    ) -> IntermediateRoundState:
        assert self._staged_delegates is not None
        provider = self._staged_delegates.hidden_state_provider
        assert provider is not None
        return provider.bootstrap_from_current_prefix(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
        )

    def refresh_intermediate_round_state(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        old_state: IntermediateRoundState | None,
    ) -> IntermediateRoundState:
        # Full frontier rebuild from the updated prefix. ``old_state`` is ignored
        # until incremental Eagle/KV advancement is implemented; the runner
        # still passes it so call sites can evolve without signature churn.
        # When ``GPUModelRunner`` enables the intermediate KV mirror, the runner
        # disables staged fastpath and skips calling this on the hot path.
        del old_state
        return self.bootstrap_intermediate_round_state(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
        )

    def _resolve_staged_hidden_bundle_for_adaptive(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        reuse_intermediate_state: IntermediateRoundState | None = None,
    ) -> StagedHiddenStateBundle | None:
        if not self._use_staged_eagle or self._staged_delegates is None:
            return None
        spec = self.vllm_config.speculative_config
        assert spec is not None
        src = spec.adaptive_spechive_hidden_state_source or "intermediate"
        if src == "target":
            return None
        if (
            reuse_intermediate_state is not None
            and reuse_intermediate_state.hidden_bundle is not None
        ):
            return reuse_intermediate_state.hidden_bundle
        rs = self.bootstrap_intermediate_round_state(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
        )
        assert rs.hidden_bundle is not None
        return rs.hidden_bundle

    def propose_chunk_from_intermediate_state(
        self,
        *,
        inter_state: IntermediateRoundState,
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
        assert self._staged_delegates is not None
        assert inter_state.hidden_bundle is not None
        from vllm.v1.spec_decode.staged_eagle import EagleHeadProposalEngine

        eng = self._staged_delegates.proposal_engine
        assert isinstance(eng, EagleHeadProposalEngine)
        rows, probs = eng.propose_from_hidden_states(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            hidden_bundle=inter_state.hidden_bundle,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            chunk_len=chunk_len,
            sampling_metadata=sampling_metadata,
            use_draft_probs=use_draft_probs,
        )
        return DitRoundProposal(tokens=rows, probs=probs)

    def verify_chunk_with_intermediate_state(
        self,
        *,
        inter_state: IntermediateRoundState,
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
        return self.verify_chunk_with_inter_verifier(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            candidate_tokens=candidate_tokens,
            reuse_intermediate_state=inter_state,
            mirror_kv_common_attn_metadata=mirror_kv_common_attn_metadata,
        )

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
        reuse_intermediate_state: IntermediateRoundState | None = None,
    ) -> DitRoundProposal:
        """Chunk proposal conditioned on base sampled token + logical prefix."""
        bundle = self._resolve_staged_hidden_bundle_for_adaptive(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            reuse_intermediate_state=reuse_intermediate_state,
        )
        if bundle is not None and self._staged_delegates is not None:
            from vllm.v1.spec_decode.staged_eagle import EagleHeadProposalEngine

            eng = self._staged_delegates.proposal_engine
            assert isinstance(eng, EagleHeadProposalEngine)
            tokens, probs = eng.propose_from_hidden_states(
                base_target_token_ids=base_target_token_ids,
                base_target_positions=base_target_positions,
                hidden_bundle=bundle,
                base_next_token_ids=base_next_token_ids,
                base_common_attn_metadata=base_common_attn_metadata,
                base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
                prefix_rows=prefix_rows,
                chunk_len=chunk_len,
                sampling_metadata=sampling_metadata,
                use_draft_probs=use_draft_probs,
            )
            return DitRoundProposal(tokens=tokens, probs=probs)
        tokens, probs = _propose_chunk_from_prefix(
            self._delegate,
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
        reuse_intermediate_state: IntermediateRoundState | None = None,
        mirror_kv_common_attn_metadata: CommonAttentionMetadata | None = None,
    ) -> DitRoundVerification:
        """Intermediate verification logits conditioned on logical prefix.

        HV still runs ``_verify_chunk_with_prefix`` → ``_forward_prefix_conditioned_logits``
        → ``_build_prefix_conditioned_inputs``: the logical prefix rows and draft
        roll are **synthetic extensions** on top of the per-request base slice from
        ``cad``.

        ``intermediate_kv_mode="draft_like"``: use ``base_common_attn_metadata`` only;
        pass ``mirror_kv_common_attn_metadata=None``.

        ``mirror_kv_common_attn_metadata`` (``intermediate_kv_mode="mirror_frontier"``):
        when set by ``GPUModelRunner._prepare_intermediate_metadata``, only that
        **base** ``cad`` (slot mapping, seq_lens, query_start_loc for the mirror
        ``InputBatch``) replaces ``base_common_attn_metadata``; prefix-conditioned
        packing is unchanged. Prefab fastpaths are skipped because mirror CAD disagrees
        with target-step prefab geometry.
        """
        assert self._inter_dit is not None
        spec = self.vllm_config.speculative_config
        assert spec is not None
        src = spec.adaptive_spechive_hidden_state_source or "intermediate"
        hb = None
        if reuse_intermediate_state is not None:
            hb = reuse_intermediate_state.hidden_bundle
        elif self._use_staged_eagle and src != "target":
            rs = self.bootstrap_intermediate_round_state(
                base_target_token_ids=base_target_token_ids,
                base_target_positions=base_target_positions,
                base_target_hidden_states=base_target_hidden_states,
                base_next_token_ids=base_next_token_ids,
                base_common_attn_metadata=base_common_attn_metadata,
                base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
                prefix_rows=prefix_rows,
            )
            hb = rs.hidden_bundle
        if mirror_kv_common_attn_metadata is not None:
            # Prefab bundles are built from target-step CAD; skip when using mirror CAD.
            hb = None
        if hb is not None and hb.prefix_prefab is not None and src != "target":
            logits_flat, bonus_logits = verify_intermediate_chunk_with_prefix_prefab(
                self._inter_dit,
                prefix_prefab=hb.prefix_prefab,
                base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
                draft_tokens=candidate_tokens,
            )
            return DitRoundVerification(
                logits_flat=logits_flat,
                bonus_logits=bonus_logits,
            )
        cad_verify = (
            mirror_kv_common_attn_metadata
            if mirror_kv_common_attn_metadata is not None
            else base_common_attn_metadata
        )
        logits_flat, bonus_logits = _verify_chunk_with_prefix(
            self._inter_dit,
            cad=cad_verify,
            target_token_ids=base_target_token_ids,
            target_positions=base_target_positions,
            target_hidden_states=base_target_hidden_states,
            next_token_ids=base_next_token_ids,
            num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            draft_tokens=candidate_tokens.to(torch.int32),
        )
        return DitRoundVerification(
            logits_flat=logits_flat,
            bonus_logits=bonus_logits,
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
        """Token-level acceptance wrapper; no authoritative state mutation."""
        batch_size = int(proposal.tokens.shape[0])
        chunk_len = int(proposal.tokens.shape[1])
        draft_flat = proposal.tokens.reshape(-1).to(torch.int32)
        num_draft_tokens = [chunk_len] * batch_size
        cu_num_draft_tokens = torch.cumsum(
            torch.tensor(
                num_draft_tokens, dtype=torch.int32, device=draft_flat.device
            ),
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

    def take_pending_spechive_bundle(self) -> HybridProposalBundle | None:
        bundle = self._pending_hybrid_bundle
        self._pending_hybrid_bundle = None
        return bundle

    def discard_pending_hierarchical_verification_state(self) -> None:
        """Drop pending hierarchical verification state when bundle is discarded."""
        self._pending_hybrid_bundle = None

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
    ) -> torch.Tensor:
        if self._mode != "hierarchical_verification":
            self.discard_pending_hierarchical_verification_state()
            rows = self._delegate.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                sampling_metadata=sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )
            self._pending_hybrid_bundle = _build_hybrid_bundle_from_rows(
                rows,
                mode=self._mode,
                draft_probs=getattr(self._delegate, "last_draft_probs_flat", None),
            )
            return rows

        assert self.runner is not None
        if mm_embed_inputs is not None:
            raise NotImplementedError(
                "adaptive_spechive hierarchical_verification does not support multimodal draft inputs yet "
                "(sub-batch MM gathering is unimplemented)."
            )
        out, bundle = self.runner.run_hierarchical_verification_rounds(
            drafter=self,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        self._pending_hybrid_bundle = bundle
        self.clear_draft_probs()
        return out
