# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import gc
import itertools
import json
import os
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, replace as dataclass_replace
from functools import reduce
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    SpeculativeConfig,
    VllmConfig,
    get_layers_from_vllm_config,
    update_config,
)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import (
    BatchDescriptor,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    XDRotaryEmbedding,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsXDRoPE,
    is_mixture_of_experts,
    supports_eagle3,
    supports_mrope,
    supports_multimodal_pruning,
    supports_realtime,
    supports_transcription,
    supports_xdrope,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    is_pooling_model,
    is_text_generation_model,
)
from vllm.model_executor.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import (
    BatchedTensorInputs,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.tracing import instrument
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.nvtx_pytorch_hooks import PytHooks
from vllm.utils.platform_utils import is_pin_memory_available, num_compute_units
from vllm.utils.torch_utils import (
    get_dtype_size,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    create_fast_prefill_custom_backend,
    get_dcp_local_seq_lens,
    reorder_batch_to_split_decodes_and_prefills,
)
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    SamplerOutput,
    SpecDecodeCostBreakdown,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    RejectionSampler,
    apply_sampling_constraints,
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.adaptive_spechive import AdaptiveSpechiveProposer
from vllm.v1.spec_decode.draft_model import (
    DraftModelProposer,
    PinnedDraftNamespaceDraftModelProposer,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from vllm.v1.spec_decode.adaptive_cascade import (
    IntermediateDraftModelProposer,
    _canonicalize_reused_prefix_frontier,
)
from vllm.v1.spec_decode.hierarchical_verification import HierarchicalVerificationProposer
from vllm.v1.spec_decode.hv_step_packing import (
    build_prefix_conditioned_inputs as hv_build_prefix_conditioned_inputs,
    gather_hv_verification_logits_from_spec_decode_metadata,
    slice_hv_verification_logits,
)
from vllm.v1.spec_decode.hybrid_bundle_utils import (
    _build_hybrid_bundle_from_rows,
    _flatten_prob_rows_for_output,
)
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.pivot import PivotProposer
from vllm.v1.spec_decode.spec_stage_ops import (
    collapse_family_paths_to_origin,
    collapse_family_tree_sampled_to_family_paths,
    collapse_pivot_expanded_sampled_to_origin,
    expand_intermediate_state_for_pivot_plan,
    validate_hybrid_bundle_draft_layout,
    validate_hierarchical_verification_tail_len_rowwise,
    get_accepted_draft_lens_from_sampled_tokens,
    get_target_verification_accepted_draft_prefix_lens,
    get_unselected_cleanup_rows,
    remap_hybrid_bundle_rows_for_metadata,
    sanitize_hybrid_bundle_for_metadata,
    select_pivot_expanded_rows_to_origin,
    validate_root_only_pivot_expansion,
)
from vllm.v1.spec_decode.spec_stage_runtime import (
    DitRoundDecision,
    DitRoundProposal,
    DitRoundVerification,
    HybridProposalBundle,
    IntermediateRoundState,
    PivotExpandedTreePlan,
    PivotExpansionPlan,
    RootTopKInfo,
    _split_flat_tokens_by_lengths,
    _split_probs_by_lengths,
    expand_hybrid_bundle_for_pivot_expansion,
    pivot_expansion_indices_fit_prepare_batch,
)
from vllm.v1.spec_decode.spec_stage_utils import (
    clone_common_attn_metadata,
    slice_sampling_metadata_for_subbatch,
)
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.spec_decode.tetris import apply_tetris
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.cp_utils import (
    check_attention_cp_compatibility,
    get_total_cp_world_size,
)
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ec_connector_model_runner_mixin import ECConnectorModelRunnerMixin
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    maybe_create_ubatch_slices,
    split_attn_metadata,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.workspace import lock_workspace

from .utils import (
    AttentionGroup,
    KVBlockZeroer,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    prepare_kernel_block_sizes,
    sanity_check_mm_encoder_outputs,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer

logger = init_logger(__name__)


def _pivot_origin_row_num_draft_tokens(
    expansion_plan: PivotExpansionPlan,
    spec_decode_metadata: SpecDecodeMetadata,
) -> list[int]:
    """Per-origin draft widths for post-collapse rows (one row per origin request)."""
    meta_nd = spec_decode_metadata.num_draft_tokens
    b_origin = (
        int(expansion_plan.origin_batch_size)
        if expansion_plan.origin_batch_size > 0
        else max(expansion_plan.expanded_to_origin) + 1
    )
    out: list[int] = []
    for o in range(b_origin):
        if (
            expansion_plan.origin_to_base_row is not None
            and o < len(expansion_plan.origin_to_base_row)
            and int(expansion_plan.origin_to_base_row[o]) >= 0
        ):
            j = int(expansion_plan.origin_to_base_row[o])
        else:
            j = next(
                idx
                for idx, org in enumerate(expansion_plan.expanded_to_origin)
                if org == o
            )
        out.append(int(meta_nd[j]))
    return out


def _post_collapse_num_draft_per_origin(
    expansion_plan: PivotExpansionPlan | None,
    tree_plan: PivotExpandedTreePlan | None,
    spec_decode_metadata: SpecDecodeMetadata,
) -> list[int] | None:
    """Per-origin ``num_draft_tokens`` for rows after family collapse (batch size B)."""
    if expansion_plan is not None:
        return _pivot_origin_row_num_draft_tokens(
            expansion_plan, spec_decode_metadata
        )
    if tree_plan is not None:
        meta_nd = spec_decode_metadata.num_draft_tokens
        b = int(tree_plan.origin_batch_size)
        out: list[int] = []
        for o in range(b):
            j = next(
                idx
                for idx, org in enumerate(tree_plan.expanded_to_origin)
                if org == o
            )
            if j >= len(meta_nd):
                return None
            out.append(int(meta_nd[j]))
        return out
    return None


def _pivot_mean_accepted_prefix_len(
    sampled_token_ids: torch.Tensor,
    num_draft_tokens: list[int] | None,
) -> tuple[float, int]:
    """Mean accepted draft-prefix length per row (``get_target_verification_...`` when aligned)."""
    nrows = int(sampled_token_ids.shape[0])
    if nrows == 0:
        return 0.0, 0
    if num_draft_tokens is not None and len(num_draft_tokens) == nrows:
        lens = get_target_verification_accepted_draft_prefix_lens(
            sampled_token_ids,
            num_draft_tokens,
            placeholder_token_id=PLACEHOLDER_TOKEN_ID,
        )
    else:
        lens = get_accepted_draft_lens_from_sampled_tokens(
            sampled_token_ids,
            placeholder_token_id=PLACEHOLDER_TOKEN_ID,
        )
    return (sum(lens) / len(lens), nrows)


def _pivot_plan_sm_indices(plan: PivotExpansionPlan) -> list[int]:
    """Sampling-metadata / list-index origins: always in ``[0, B)`` (never ``packed_to_origin``)."""
    if plan.packed_sm_origin is not None:
        return plan.packed_sm_origin
    return plan.expanded_to_origin


def _hybrid_bundle_row_req_ids_for_batch(
    input_batch,
    *,
    origin_batch_size: int,
    num_bundle_rows: int,
    pivot_expansion_plan: PivotExpansionPlan | None,
) -> tuple[str, ...] | None:
    """One scheduler request id per hybrid bundle row (propose-time batch order)."""
    if origin_batch_size <= 0 or num_bundle_rows <= 0:
        return None
    try:
        req = [str(input_batch.req_ids[i]) for i in range(origin_batch_size)]
    except (IndexError, TypeError):
        return None
    if pivot_expansion_plan is None:
        if num_bundle_rows != origin_batch_size:
            return None
        return tuple(req[b] for b in range(num_bundle_rows))
    sm = _pivot_plan_sm_indices(pivot_expansion_plan)
    if len(sm) < num_bundle_rows:
        return None
    out: list[str] = []
    for j in range(num_bundle_rows):
        o = int(sm[j])
        if o < 0 or o >= len(req):
            return None
        out.append(req[o])
    return tuple(out)


def _uses_pivot_linear_fixed_capacity_packing(
    speculative_config: SpeculativeConfig | None,
) -> bool:
    """Linear pivot fixed P layout (not eagle-tree legacy variable expansion)."""
    if speculative_config is None or speculative_config.method != "pivot":
        return False
    return (
        not speculative_config.pivot_use_eagle_tree
        and int(speculative_config.pivot_topk_selection) > 1
    )


def _expand_list_rows_by_pivot_plan(
    rows: list[list[Any]], plan: PivotExpansionPlan
) -> list[list[Any]]:
    return [list(rows[o]) for o in _pivot_plan_sm_indices(plan)]


def _collapse_draft_tensor_rows_for_scheduler(
    out_exp: torch.Tensor,
    plan: PivotExpansionPlan | None,
    batch_size: int,
) -> torch.Tensor:
    """Scheduler/drafter return value is origin-batch rows; bundle may be B'."""
    if plan is None:
        return out_exp
    if batch_size <= 0:
        return out_exp[:0]

    # Fixed-capacity pivot invariant: base rows are [0, B) in origin order.
    if (
        plan.uses_fixed_capacity_packing
        and plan.origin_to_base_row is not None
        and len(plan.origin_to_base_row) >= batch_size
        and all(int(plan.origin_to_base_row[o]) == o for o in range(batch_size))
    ):
        return out_exp[:batch_size]

    row_ids: list[int] = []
    if (
        plan.origin_to_base_row is not None
        and len(plan.origin_to_base_row) >= batch_size
    ):
        row_ids = [int(plan.origin_to_base_row[o]) for o in range(batch_size)]
    else:
        expanded_to_origin = plan.expanded_to_origin
        for o in range(batch_size):
            row_ids.append(
                next(idx for idx, orig in enumerate(expanded_to_origin) if int(orig) == o)
            )

    idx = torch.tensor(row_ids, dtype=torch.long, device=out_exp.device)
    return out_exp.index_select(0, idx)


def _clamp_hybrid_bundle_to_max_counts(
    bundle: HybridProposalBundle,
    max_per_row: np.ndarray,
) -> HybridProposalBundle:
    """Truncate expanded-row draft tokens to at most *max_per_row* each.

    When intermediate verification accepts different numbers of tokens for
    different expanded rows, the bundle's per-row counts can diverge from
    the scheduler's collapsed (origin-based) counts.  This helper rebuilds
    the flat token tensor so that every row has at most the origin's
    scheduled count, keeping the bundle and metadata in sync.
    """
    pb_ndt = bundle.num_draft_tokens
    clamped: list[int] = [
        min(int(pb_ndt[j]), int(max_per_row[j])) for j in range(len(pb_ndt))
    ]
    if clamped == list(pb_ndt):
        return bundle
    tok_rows = _split_flat_tokens_by_lengths(
        bundle.draft_token_ids, pb_ndt
    )
    new_flat_parts: list[torch.Tensor] = []
    for j, row in enumerate(tok_rows):
        c = clamped[j]
        if c > 0:
            new_flat_parts.append(row[:c])
    if new_flat_parts:
        new_flat = torch.cat(new_flat_parts, dim=0)
    else:
        new_flat = bundle.draft_token_ids.new_empty(
            (0,), dtype=bundle.draft_token_ids.dtype
        )
    device = new_flat.device
    cu = torch.cumsum(
        torch.tensor(clamped, dtype=torch.int32, device=device), dim=0
    ).to(torch.int32)
    new_probs = None
    if bundle.draft_probs is not None:
        prob_rows = _split_probs_by_lengths(bundle.draft_probs, pb_ndt)
        new_prob_parts: list[torch.Tensor] = []
        for j, row in enumerate(prob_rows):
            c = clamped[j]
            if c > 0:
                new_prob_parts.append(row[:c])
        if new_prob_parts:
            new_probs = torch.cat(new_prob_parts, dim=0)
    new_src = None
    if bundle.source_stage is not None:
        src_rows = _split_flat_tokens_by_lengths(
            bundle.source_stage, pb_ndt
        )
        new_src_parts: list[torch.Tensor] = []
        for j, row in enumerate(src_rows):
            c = clamped[j]
            if c > 0:
                new_src_parts.append(row[:c])
        if new_src_parts:
            new_src = torch.cat(new_src_parts, dim=0)
    return dataclass_replace(
        bundle,
        draft_token_ids=new_flat,
        draft_probs=new_probs,
        num_draft_tokens=clamped,
        cu_num_draft_tokens=cu,
        max_spec_len=max(clamped) if clamped else bundle.max_spec_len,
        source_stage=new_src,
    )


AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


@dataclass(frozen=True)
class AttentionMetadataBuildResult:
    """Result of ``_build_attention_metadata`` (single object return; no hidden runner state)."""

    attn_metadata: PerLayerAttnMetadata
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    spec_decode_common_attn_metadata_by_gid: dict[int, CommonAttentionMetadata]


@dataclass
class HvVerifyPrefabrication:
    """Packed intermediate-verify step inputs (no ``set_inputs_first_pass`` yet)."""

    pref_toks: torch.Tensor
    pref_pos: torch.Tensor
    pref_hidden: torch.Tensor
    pref_next: torch.Tensor
    pref_cad: CommonAttentionMetadata
    base_query_lens: list[int]
    prefix_lens: list[int]
    roll_lens: list[int]
    candidate_tokens: torch.Tensor


class IntermediateFrontierPrepareResult(NamedTuple):
    """Attention + spec metadata for one step using ``intermediate_input_batch`` buffers."""

    logits_indices: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    attn_metadata: PerLayerAttnMetadata
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    slot_mappings_by_layer: dict[str, torch.Tensor] | list[
        dict[str, torch.Tensor]
    ] | None
    spec_decode_common_attn_metadata_by_gid: (
        dict[int, CommonAttentionMetadata] | None
    ) = None


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: LogprobsTensors | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
        vocab_size: int,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._logprobs_tensors
        del self._sampled_token_ids
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            logprobs_lists = None
            if self._logprobs_tensors_cpu is not None:
                logprobs_lists = self._logprobs_tensors_cpu.tolists()
        else:
            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                self.sampled_token_ids_cpu,
                self.vocab_size,
                self._invalid_req_indices,
                logprobs_tensors=self._logprobs_tensors_cpu,
            )

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists
        return output


def _copy_pooler_output_to_cpu(
    raw_pooler_output: PoolerOutput, finished_mask: list[bool]
) -> list[torch.Tensor | None]:
    num_reqs = len(finished_mask)

    if isinstance(raw_pooler_output, torch.Tensor):
        if raw_pooler_output.shape[0] != num_reqs:
            raise ValueError(
                "Pooler output batch size does not match finished mask size: "
                f"{raw_pooler_output.shape[0]} != {num_reqs}."
            )

        num_finished = sum(finished_mask)
        if num_finished == 0:
            return [None] * num_reqs
        if num_finished == num_reqs:
            return list(raw_pooler_output.to("cpu", non_blocking=True))

        # partial finished
        finished_indices = [i for i, include in enumerate(finished_mask) if include]
        index_tensor = torch.tensor(
            finished_indices, device=raw_pooler_output.device, dtype=torch.long
        )
        finished_outputs = raw_pooler_output.index_select(0, index_tensor).to(
            "cpu", non_blocking=True
        )
        partial_pooler_output: list[torch.Tensor | None] = [None] * num_reqs
        for i, out in zip(finished_indices, finished_outputs):
            partial_pooler_output[i] = out
        return partial_pooler_output

    assert isinstance(raw_pooler_output, list)
    if len(raw_pooler_output) != num_reqs:
        raise ValueError(
            "Pooler output batch size does not match finished mask size: "
            f"{len(raw_pooler_output)} != {num_reqs}."
        )

    pooler_output: list[torch.Tensor | None] = [None] * num_reqs
    for i, (out, include) in enumerate(zip(raw_pooler_output, finished_mask)):
        if include and out is not None:
            pooler_output[i] = out.to("cpu", non_blocking=True)
    return pooler_output


class AsyncGPUPoolingModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        raw_pooler_output: PoolerOutput,
        finished_mask: list[bool],
        async_output_copy_stream: torch.cuda.Stream,
    ):
        self._model_runner_output = model_runner_output

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()

        # Keep a reference to the device tensors to avoid them being
        # deallocated until we finish copying it to the host.
        self._raw_pooler_output = raw_pooler_output

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self._model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
                raw_pooler_output=self._raw_pooler_output,
                finished_mask=finished_mask,
            )
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.
        This function blocks until the copy is finished.
        """
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._raw_pooler_output
        return self._model_runner_output


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None
    spec_decode_common_attn_metadata_by_gid: (
        dict[int, CommonAttentionMetadata] | None
    ) = None
    full_verification_time_sec: float = 0.0


class GPUModelRunner(
    LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
):
    @staticmethod
    def _static_intermediate_kv_frontier_enabled(vllm_config: VllmConfig) -> bool:
        spec = vllm_config.speculative_config
        if spec is None or spec.intermediate_model is None:
            return False
        if getattr(spec, "intermediate_kv_mode", "mirror_frontier") != "mirror_frontier":
            return False
        if spec.method == "adaptive_spechive":
            return spec.adaptive_spechive_mode == "hierarchical_verification"
        if spec.method == "pivot":
            return bool(spec.pivot_spechive)
        if spec.method == "hierarchical_verification":
            return True
        return False

    def _intermediate_kv_frontier_enabled(self) -> bool:
        if not self._static_intermediate_kv_frontier_enabled(self.vllm_config):
            return False
        if not get_pp_group().is_last_rank:
            return False
        return self.intermediate_input_batch is not None

    def _draft_kv_frontier_enabled(self) -> bool:
        """Standalone HV: draft mirror batch + metadata-direct draft proposals."""
        if not self._intermediate_kv_frontier_enabled():
            return False
        spec = self.speculative_config
        if spec is None or spec.method != "hierarchical_verification":
            return False
        return self.draft_input_batch is not None

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.offload_config = vllm_config.offload_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        from vllm.v1.spec_decode.profiler import (
            NullSpecDecodeProfiler,
            create_spec_decode_profiler,
        )
        self._spec_profiler = create_spec_decode_profiler(
            vllm_config, role="worker")
        self._current_profile_ctx = None
        self._null_profiler = NullSpecDecodeProfiler()
        self._deep_profiler = getattr(
            self._spec_profiler, "_deep_profiler", None)

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype

        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config
        )

        self.is_pooling_model = model_config.runner_type == "pooling"
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model
        )
        # This will be overridden in load_model()
        self.is_multimodal_pruning_enabled = False
        self.max_model_len = model_config.max_model_len

        # Always set to false after the first forward pass
        self.calculate_kv_scales = self.cache_config.calculate_kv_scales
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 1
        )

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = model_config.uses_alibi

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
        self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.uses_xdrope_dim = model_config.uses_xdrope_dim
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )

        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # Async scheduling
        self.use_async_scheduling = self.scheduler_config.async_scheduling

        # Sampler
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        self.eplb_state: EplbState | None = None
        # NOTE(yongji): flag to temporarily disable EPLB during scaling up/down
        self.eep_eplb_suppressed = False
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        # Initialize in initialize_kv_cache_tensors
        self.cross_layers_kv_cache: torch.Tensor | None = None
        self.cross_layers_attn_backend: type[AttentionBackend] | None = None
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            self.drafter: (
                NgramProposer  # noqa: F823
                | SuffixDecodingProposer
                | EagleProposer
                | DraftModelProposer
                | AdaptiveSpechiveProposer
                | HierarchicalVerificationProposer
                | PivotProposer
                | MedusaProposer
                | ExtractHiddenStatesProposer
            )
            if self.speculative_config.method == "ngram":
                from vllm.v1.spec_decode.ngram_proposer import NgramProposer

                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.method == "adaptive_spechive":
                self.drafter = AdaptiveSpechiveProposer(
                    vllm_config=self.vllm_config,
                    device=self.device,
                    runner=self,
                )
                if self.speculative_config.requires_aux_hidden_state_outputs():
                    self.use_aux_hidden_state_outputs = bool(
                        getattr(
                            self.drafter._delegate,
                            "eagle3_use_aux_hidden_state",
                            False,
                        )
                    )
            elif self.speculative_config.method == "pivot":
                self.drafter = PivotProposer(
                    vllm_config=self.vllm_config,
                    device=self.device,
                    runner=self,
                )
                if self.speculative_config.requires_aux_hidden_state_outputs():
                    aux_from_delegate = getattr(
                        self.drafter._main_delegate(),  # type: ignore[attr-defined]
                        "eagle3_use_aux_hidden_state",
                        False,
                    )
                    self.use_aux_hidden_state_outputs = bool(aux_from_delegate)
            elif self.speculative_config.method == "hierarchical_verification":
                self.drafter = HierarchicalVerificationProposer(
                    vllm_config=self.vllm_config,
                    device=self.device,
                    runner=self,
                )
            elif self.speculative_config.uses_draft_model():
                self.drafter = DraftModelProposer(
                    vllm_config=self.vllm_config,
                    device=self.device,
                    runner=self,
                )
            elif self.speculative_config.method == "suffix":
                self.drafter = SuffixDecodingProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
            elif self.speculative_config.method == "extract_hidden_states":
                self.drafter = ExtractHiddenStatesProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
                self.use_aux_hidden_state_outputs = True
            else:
                raise ValueError(
                    "Unknown speculative decoding method: "
                    f"{self.speculative_config.method}"
                )
            self.rejection_sampler = RejectionSampler(self.sampler)
        else:
            # No speculative decoding (or not the last PP rank): sampling paths
            # must not assume ``self.drafter`` / ``self.rejection_sampler`` exist.
            self.drafter = None
            self.rejection_sampler = None
        self.pending_hybrid_spec_bundle: HybridProposalBundle | None = None
        self.pending_pivot_expansion_plan: PivotExpansionPlan | None = None
        self._dit_debug_enabled = (
            os.environ.get("VLLM_SPEC_DIT_DEBUG", "0") == "1"
            or os.environ.get("VLLM_SPEC_SPECHIVE_DEBUG", "0") == "1"
        )
        self._dit_debug_summary = (
            os.environ.get("VLLM_SPEC_DIT_DEBUG_SUMMARY", "0") == "1"
            or os.environ.get("VLLM_SPEC_SPECHIVE_DEBUG_SUMMARY", "0") == "1"
        )
        self._dit_debug_step_id = 0
        self._dit_debug_check_counts: dict[str, list[int]] = defaultdict(
            lambda: [0, 0]
        )
        self._dit_debug_last_commit_len: dict[str, int] = {}
        self._dit_debug_last_commit_token: dict[str, int] = {}

        self.num_spec_tokens = 0
        if self.speculative_config:
            self.num_spec_tokens = (
                self.speculative_config.runner_num_speculative_tokens()
            )
            draft_config = self.speculative_config.draft_model_config
            if draft_config is not None and draft_config.max_model_len is not None:
                self.effective_drafter_max_model_len = draft_config.max_model_len
            else:
                self.effective_drafter_max_model_len = self.max_model_len

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}
        self.comm_stream = torch.cuda.Stream()

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        logits_processors = model_config.logits_processors
        custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (
            tuple(logits_processors) if logits_processors is not None else ()
        )
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoer
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            kernel_block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # We currently don't know whether a particular custom logits processor
            # uses output token ids so we set this conservatively.
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
        )

        # Intermediate-model frontier batch (spechive / standalone HV): shares
        # target block_id lists with the scheduler; see _sync_intermediate_states.
        self.intermediate_input_batch: InputBatch | None = None
        self.intermediate_requests: dict[str, CachedRequestState] = {}
        self.intermediate_committed_tokens: dict[str, int] = {}
        self.intermediate_num_accepted_tokens: CpuGpuBuffer | None = None
        if self._static_intermediate_kv_frontier_enabled(self.vllm_config):
            self.intermediate_input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.max_model_len, self.max_encoder_len),
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=[self.cache_config.block_size],
                kernel_block_sizes=[self.cache_config.block_size],
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                is_pooling_model=self.is_pooling_model,
                cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
            )
            self.intermediate_num_accepted_tokens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int64
            )
            # Scratch tensors for ``_prepare_inputs`` on the intermediate frontier batch
            # (keeps target persistent buffers untouched).
            self.intermediate_step_positions = self._make_buffer(
                self.max_num_tokens, dtype=torch.int64
            )
            self.intermediate_step_input_ids = self._make_buffer(
                self.max_num_tokens, dtype=torch.int32
            )
            self.intermediate_step_query_start_loc = self._make_buffer(
                self.max_num_reqs + 1, dtype=torch.int32
            )
            self.intermediate_step_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
            self.intermediate_step_discard_request_mask = self._make_buffer(
                self.max_num_reqs, dtype=torch.bool
            )
            self.intermediate_step_num_decode_draft_tokens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )

        # Draft-model frontier (standalone hierarchical_verification only).
        self.draft_input_batch: InputBatch | None = None
        self.draft_requests: dict[str, CachedRequestState] = {}
        self.draft_committed_tokens: dict[str, int] = {}
        self.draft_num_accepted_tokens: CpuGpuBuffer | None = None
        _spec_cfg = self.speculative_config
        if (
            self._static_intermediate_kv_frontier_enabled(self.vllm_config)
            and _spec_cfg is not None
            and _spec_cfg.method == "hierarchical_verification"
        ):
            self.draft_input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.max_model_len, self.max_encoder_len),
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=[self.cache_config.block_size],
                kernel_block_sizes=[self.cache_config.block_size],
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                is_pooling_model=self.is_pooling_model,
                cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
            )
            self.draft_num_accepted_tokens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int64
            )
            self.draft_step_positions = self._make_buffer(
                self.max_num_tokens, dtype=torch.int64
            )
            self.draft_step_input_ids = self._make_buffer(
                self.max_num_tokens, dtype=torch.int32
            )
            self.draft_step_query_start_loc = self._make_buffer(
                self.max_num_reqs + 1, dtype=torch.int32
            )
            self.draft_step_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
            self.draft_step_discard_request_mask = self._make_buffer(
                self.max_num_reqs, dtype=torch.bool
            )
            self.draft_step_num_decode_draft_tokens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )

        # Set for the duration of ``propose_draft_token_ids`` when intermediate
        # frontier is enabled so HV paths can call ``_prepare_intermediate_metadata``
        # with the current scheduler output.
        self._hv_scheduler_output: Any = None

        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        self.async_output_copy_stream: torch.cuda.Stream | None = None
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: torch.Event | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.cuda.Stream()
            self.prepare_inputs_event = torch.Event()

        # Cache the device properties.
        self._init_device_properties()

        # Encoder timing registry for observability
        self.encoder_timing_registry: dict[str, EncoderTimingStats] = {}
        self._encoder_timing_lock = threading.Lock()

        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.positions = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
        )
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.discard_request_mask = self._make_buffer(
            self.max_num_reqs, dtype=torch.bool
        )
        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int64
        )

        # Only relevant for multimodal models
        if self.supports_mm_inputs:
            # Double buffer to avoid race condition: previous iteration's async
            # copy may still be reading from CPU while current iteration writes.
            self.is_mm_embed_buffers = [
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
            ]
            self.is_mm_embed_idx = 0

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int64
            )

        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            # Similar to mrope but use assigned dimension number for RoPE, 4 as default.
            self.xdrope_positions = self._make_buffer(
                (self.uses_xdrope_dim, self.max_num_tokens + 1), dtype=torch.int64
            )

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),
            dtype=np.int64,
        )

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self.mm_budget = (
            MultiModalBudget(self.vllm_config, self.mm_registry)
            if self.supports_mm_inputs
            else None
        )

        self.reorder_batch_threshold: int | None = None

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
        self._draft_token_req_ids: list[str] | None = None
        self.transfer_event = torch.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        # Pre-allocated tensor for copying valid sampled token counts to CPU,
        # with dedicated stream for overlapping and event for coordination.
        self.valid_sampled_token_count_event: torch.Event | None = None
        self.valid_sampled_token_count_copy_stream: torch.cuda.Stream | None = None
        # We also copy the drafted tokens to the CPU asynchronously,
        # in case we need them for structured outputs.
        self.draft_token_ids_event: torch.Event | None = None
        self.draft_token_ids_copy_stream: torch.cuda.Stream | None = None
        self.valid_sampled_token_count_cpu: torch.Tensor | None = None
        self.draft_token_ids_cpu: torch.Tensor | None = None
        if self.num_spec_tokens:
            self.draft_token_ids_event = torch.Event()
            self.draft_token_ids_copy_stream = torch.cuda.Stream()
            self.draft_token_ids_cpu = torch.empty(
                (self.max_num_reqs, self.num_spec_tokens),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory,
            )
            if self.use_async_scheduling:
                self.valid_sampled_token_count_event = torch.Event()
                self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()
                self.valid_sampled_token_count_cpu = torch.empty(
                    self.max_num_reqs,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )

        # Model weight offloader
        # Make sure this is called before any get_offloader call
        set_offloader(create_offloader(self.offload_config))

        # Ephemeral state transferred between execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None
        self.kv_connector_output: KVConnectorOutput | None = None
        self.mamba_state_idx: dict[str, int] = {}
        self._mamba_copy_bufs: mamba_utils.MambaCopyBuffers | None = None
        self.layerwise_nvtx_hooks_registered = False

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        if self.speculative_config:
            draft_config = self.speculative_config.draft_model_config
            if draft_config is None or draft_config.max_model_len is None:
                self.effective_drafter_max_model_len = self.max_model_len

    def reset_mm_cache(self) -> None:
        """
        Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        if self.mm_budget:
            self.mm_budget.reset_cache()

    def reset_encoder_cache(self) -> None:
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_cache.clear()

    @torch.inference_mode()
    def init_fp8_kv_scales(self) -> None:
        """
        Re-initialize the KV cache and FP8 scales after waking from sleep.
        1. Zero out the KV cache tensors to remove garbage data from re-allocation.
        2. Reset Attention layer scaling factors (_k_scale, _v_scale) to 1.0.
          If these are left at 0.0 (default after wake_up), all KV cache values
          become effectively zero, causing gibberish output.
        """
        if not self.cache_config.cache_dtype.startswith("fp8"):
            return

        kv_caches = getattr(self, "kv_caches", [])
        for cache_tensor in kv_caches:
            if cache_tensor is not None:
                cache_tensor.zero_()

        k_attr_names = ("_k_scale", "k_scale")
        v_attr_names = ("_v_scale", "v_scale")

        attn_layers = self.compilation_config.static_forward_context
        for name, module in attn_layers.items():
            if isinstance(module, (Attention, MLAAttention)):
                # TODO: Generally, scale is 1.0 if user uses on-the-fly fp8
                # kvcache quant. However, to get better accuracy, compression
                # frameworks like llm-compressors allow users to tune the
                # scale. We may need to restore the specific calibrated scales
                # here in the future.
                k_scale_val, v_scale_val = 1.0, 1.0

                # Processing K Scale
                for attr in k_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(k_scale_val)

                # Processing V Scale
                for attr in v_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(v_scale_val)

    def _get_positions(self, num_tokens: Any):
        if isinstance(num_tokens, int):
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, :num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, :num_tokens]
            return self.positions.gpu[:num_tokens]
        else:
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, num_tokens]
            return self.positions.gpu[num_tokens]

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=numpy,
        )

    def _get_mamba_copy_bufs(self) -> mamba_utils.MambaCopyBuffers:
        if self._mamba_copy_bufs is None:
            self._mamba_copy_bufs = mamba_utils.MambaCopyBuffers.create(
                self.max_num_reqs,
                self.kv_cache_config,
                self.model.get_mamba_state_copy_func(),
                self._make_buffer,
            )
        return self._mamba_copy_bufs

    def _init_model_kwargs(self):
        model_kwargs = dict[str, Any]()

        if not self.is_pooling_model:
            return model_kwargs

        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if (
                param.extra_kwargs is not None
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids"))
                is not None
            ):
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return model_kwargs

        seq_lens = self.seq_lens.gpu[:num_reqs]
        token_type_ids = []

        for i in range(num_reqs):
            pos = token_type_id_requests.get(i, seq_lens[i])
            ids = (torch.arange(seq_lens[i]) >= pos).int()
            token_type_ids.append(ids)

        model_kwargs["token_type_ids"] = torch.concat(token_type_ids).to(
            device=self.device
        )
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_goups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    def _init_kv_zero_meta(self) -> None:
        """One-time precomputation for _zero_block_ids.

        Delegates to KVBlockZeroer.init_meta with the runner's state.
        Called from gpu_worker.py outside the CuMem pool context.
        """
        self._kv_block_zeroer = KVBlockZeroer(self.device, self.pin_memory)
        self._kv_block_zeroer.init_meta(
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self._kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=(self.compilation_config.static_forward_context),
        )

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if hasattr(self, "_kv_block_zeroer"):
            self._kv_block_zeroer.zero_block_ids(block_ids)

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.cuda.get_device_properties"""

        self.num_sms = num_compute_units(self.device.index)

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.cuda.synchronize()

    def _intermediate_shallow_req_clone(self, src: CachedRequestState) -> CachedRequestState:
        """Clone request state for the intermediate frontier; shares ``block_ids`` lists.

        Intermediate frontier — storage assumptions (see plan Phase A):

        - **Logical blocks**: Intermediate frontier ``CachedRequestState`` shares the
          same ``block_ids`` lists as the target request so the scheduler and block
          pool stay consistent with a single allocation story per request.

        - **Physical KV tensors**: The intermediate verifier model uses the draft /
          intermediate attention path (``SpecDecodeBaseProposer`` + ``set_forward_context``).
          Slot mappings for verify may be supplied from ``_prepare_intermediate_metadata``
          so attention reads/writes align with the ``intermediate_input_batch`` for that
          step.

        - **Rejected speculative suffix**: There is no separate GPU "delete" / rollback
          kernel pass after target sampling; intermediate frontier ``num_computed_tokens``
          / ``output_token_ids`` are realigned to the authoritative target in
          ``_reconcile_intermediate_frontier_after_target``, and the intermediate
          ``InputBatch`` block table is re-committed so device-side tables match.
          Stale KV bytes in shared blocks beyond the reconciled sequence length are
          not explicitly zeroed; later scheduled forwards overwrite those positions.
          Debug builds may add stricter checks if a backend requires explicit
          invalidation.

        - **HV verify geometry**: Adaptive/pivot spechive inner verification may still
          use ``_verify_chunk_with_prefix`` (prefix + roll) with optional
          ``mirror_kv_common_attn_metadata`` (legacy name) replacing only the **base**
          ``CommonAttentionMetadata``. Standalone ``hierarchical_verification`` uses
          intermediate frontier ``_prepare_intermediate_metadata`` + one direct
          intermediate forward and ``gather_hv_verification_logits_from_spec_decode_metadata``.
        """
        return CachedRequestState(
            req_id=src.req_id,
            prompt_token_ids=src.prompt_token_ids,
            prompt_embeds=src.prompt_embeds,
            mm_features=src.mm_features,
            sampling_params=src.sampling_params,
            pooling_params=src.pooling_params,
            generator=src.generator,
            block_ids=src.block_ids,
            num_computed_tokens=src.num_computed_tokens,
            output_token_ids=list(src.output_token_ids),
            mrope_positions=src.mrope_positions,
            mrope_position_delta=src.mrope_position_delta,
            xdrope_positions=src.xdrope_positions,
            lora_request=src.lora_request,
            prev_num_draft_len=src.prev_num_draft_len,
        )

    def _intermediate_speculative_token_budget(self) -> int:
        assert self.speculative_config is not None
        return int(self.speculative_config.runner_num_speculative_tokens())

    def _intermediate_assert_round_prefix_budget(
        self,
        *,
        prefix_len: int,
        extra_accepted: int,
        row_detail: str,
    ) -> None:
        """Invariant: inner HV prefix + acceptance must fit runner spec token budget."""
        budget = self._intermediate_speculative_token_budget()
        if prefix_len + extra_accepted <= budget:
            return
        msg = (
            "intermediate KV frontier: prefix+accept exceeds speculative budget "
            f"(prefix={prefix_len}, +={extra_accepted}, budget={budget}, {row_detail})"
        )
        if self._dit_debug_enabled or os.environ.get("VLLM_SPEC_SPECHIVE_DEBUG", "0") == "1":
            raise AssertionError(msg)
        logger.error(msg)

    def _sync_intermediate_states_with_scheduler(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Sync scheduler-driven lifecycle onto intermediate frontier batch (shared new_block_ids)."""
        if not self._intermediate_kv_frontier_enabled():
            return
        ib = self.intermediate_input_batch
        assert ib is not None
        for req_id in scheduler_output.finished_req_ids:
            self.intermediate_requests.pop(req_id, None)
            self.intermediate_committed_tokens.pop(req_id, None)
            if req_id in ib.req_id_to_index:
                ib.remove_request(req_id)
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = set(ib.req_id_to_index.keys())
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        unscheduled_req_ids = cached_req_ids - (
            set(scheduled_req_ids) - set(resumed_req_ids)
        )
        for req_id in unscheduled_req_ids:
            ib.remove_request(req_id)

        active = set(self.input_batch.req_id_to_index.keys())
        for req_id in list(ib.req_id_to_index.keys()):
            if req_id not in active:
                ib.remove_request(req_id)

        scheduled_spec = scheduler_output.scheduled_spec_decode_tokens
        for req_index in range(self.input_batch.num_reqs):
            req_id = self.input_batch.req_ids[req_index]
            tgt = self.requests[req_id]
            committed = self.intermediate_committed_tokens.get(
                req_id, tgt.num_computed_tokens
            )
            if req_id in ib.req_id_to_index:
                mir = self.intermediate_requests[req_id]
                mir.num_computed_tokens = committed
                mir_idx = ib.req_id_to_index[req_id]
                ib.num_computed_tokens_cpu[mir_idx] = committed
                ib.update_req_spec_token_ids(mir, scheduled_spec)
            else:
                mir = self.intermediate_requests.get(req_id)
                if mir is None:
                    mir = self._intermediate_shallow_req_clone(tgt)
                    self.intermediate_requests[req_id] = mir
                mir.num_computed_tokens = committed
                ib.add_request(mir)
                mir_idx = ib.req_id_to_index[req_id]
                ib.num_computed_tokens_cpu[mir_idx] = committed
                ib.update_req_spec_token_ids(mir, scheduled_spec)
        ib.condense()
        ib.refresh_metadata()

    def _prepare_intermediate_metadata(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
        req_ids_subset: list[str] | None = None,
    ) -> IntermediateFrontierPrepareResult | None:
        """Build step-shaped attention + spec metadata using the intermediate frontier batch.

        Uses scratch buffers so the target ``_prepare_inputs`` tensors are untouched.
        Returns ``None`` when the intermediate batch diverges from the target row set, when
        Mamba align-mode preprocessing would be required, or for unsupported layouts
        (M-RoPE / XD-RoPE / prompt embeds / async prev-sample path).
        """
        if not self._intermediate_kv_frontier_enabled():
            return None
        if req_ids_subset is not None:
            return None
        ib = self.intermediate_input_batch
        assert ib is not None
        if self.input_batch.prev_sampled_token_ids is not None:
            return None
        if self.cache_config.mamba_cache_mode == "align":
            return None
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return None
        num_reqs_ib = ib.num_reqs
        if (
            num_reqs_ib != self.input_batch.num_reqs
            or num_reqs_ib <= 0
            or tuple(ib.req_ids[:num_reqs_ib])
            != tuple(self.input_batch.req_ids[:num_reqs_ib])
        ):
            return None
        total_tok = scheduler_output.total_num_scheduled_tokens
        if total_tok <= 0:
            return None

        out_b = {
            "positions": self.intermediate_step_positions,
            "input_ids": self.intermediate_step_input_ids,
            "query_start_loc": self.intermediate_step_query_start_loc,
            "seq_lens": self.intermediate_step_seq_lens,
            "discard_request_mask": self.intermediate_step_discard_request_mask,
            "num_decode_draft_tokens": self.intermediate_step_num_decode_draft_tokens,
        }
        logits_indices, spec_decode_metadata = self._prepare_inputs(
            scheduler_output,
            num_scheduled_tokens,
            input_batch=ib,
            requests=self.intermediate_requests,
            out_buffers=out_b,
            skip_lora_swap=True,
        )

        cascade_attn_prefix_lens = None
        if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
            cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                num_scheduled_tokens,
                ib.num_computed_tokens_cpu[:num_reqs_ib],
                scheduler_output.num_common_prefix_blocks,
            )

        num_reqs = num_reqs_ib
        num_tokens_unpadded = total_tok
        max_num_scheduled_tokens = int(num_scheduled_tokens.max())
        (
            cudagraph_mode,
            batch_desc,
            should_ubatch,
            _num_tokens_across_dp,
            _cudagraph_stats,
        ) = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=cascade_attn_prefix_lens is not None,
            num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
        )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.parallel_config.num_ubatches,
        )

        has_separate_kv_update = not all(
            all(
                g.backend.forward_includes_kv_cache_update
                for g in self.attn_groups[id]
            )
            for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
            if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
        )
        pad_attn = cudagraph_mode == CUDAGraphMode.FULL

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=num_tokens_padded
            if pad_attn or has_separate_kv_update
            else num_tokens_unpadded,
            num_reqs_padded=(
                num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
            ),
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
            input_batch=ib,
        )

        nat_buf = self.intermediate_num_accepted_tokens or self.num_accepted_tokens
        attn_build = self._build_attention_metadata(
            num_tokens=num_tokens_unpadded,
            num_tokens_padded=num_tokens_padded if pad_attn else None,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded if pad_attn else None,
            max_query_len=max_num_scheduled_tokens,
            ubatch_slices=ubatch_slices_attn,
            logits_indices=logits_indices,
            use_spec_decode=use_spec_decode,
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
            cascade_attn_prefix_lens=cascade_attn_prefix_lens,
            slot_mappings=slot_mappings_by_group,
            input_batch=ib,
            common_attn_query_start_loc=self.intermediate_step_query_start_loc,
            common_attn_seq_lens=self.intermediate_step_seq_lens,
            common_attn_num_accepted_tokens=nat_buf,
            common_attn_num_decode_draft_tokens=self.intermediate_step_num_decode_draft_tokens,
        )

        return IntermediateFrontierPrepareResult(
            logits_indices=logits_indices,
            spec_decode_metadata=spec_decode_metadata,
            attn_metadata=attn_build.attn_metadata,
            spec_decode_common_attn_metadata=attn_build.spec_decode_common_attn_metadata,
            slot_mappings_by_layer=slot_mappings,
            spec_decode_common_attn_metadata_by_gid=(
                attn_build.spec_decode_common_attn_metadata_by_gid
            ),
        )

    def _advance_intermediate_frontier_after_round(
        self,
        decision: DitRoundDecision,
        *,
        batch_size: int,
        eff_bs: int,
        pivot_expansion_plan: PivotExpansionPlan | None,
        before_prefix_lens: list[int],
    ) -> None:
        """Advance intermediate working frontier after inner-round acceptance (no new blocks).

        When pivot expands the effective batch (``eff_bs != batch_size``), the mirror
        only tracks origin requests, so we skip advancing here; post-target reconcile
        realigns ``intermediate_requests`` to the authoritative target state.
        """
        if not self._intermediate_kv_frontier_enabled():
            return
        if pivot_expansion_plan is not None and eff_bs != batch_size:
            return
        for b, emitted in enumerate(decision.emitted_rows):
            if not emitted:
                continue
            origin_b = (
                int(pivot_expansion_plan.expanded_to_origin[b])
                if pivot_expansion_plan is not None
                else b
            )
            if origin_b < 0 or origin_b >= batch_size:
                continue
            req_id = self.input_batch.req_ids[origin_b]
            self._intermediate_assert_round_prefix_budget(
                prefix_len=before_prefix_lens[b],
                extra_accepted=len(emitted),
                row_detail=f"round_row={b}, req_id={req_id}",
            )
            if req_id not in self.intermediate_requests:
                continue
            mir = self.intermediate_requests[req_id]
            mir.num_computed_tokens += len(emitted)
            mir.output_token_ids.extend(emitted)
            mir_idx = self.intermediate_input_batch.req_id_to_index.get(req_id)
            if mir_idx is None:
                continue
            ib = self.intermediate_input_batch
            ib.num_computed_tokens_cpu[mir_idx] = mir.num_computed_tokens
        self.intermediate_input_batch.refresh_metadata()

    def _sync_intermediate_num_accepted_from_target(self) -> None:
        """Mirror hybrid ``num_accepted_tokens`` rows onto the intermediate batch."""
        if not self._intermediate_kv_frontier_enabled():
            return
        ib = self.intermediate_input_batch
        if ib is None or self.intermediate_num_accepted_tokens is None:
            return
        if not self.model_config.is_hybrid:
            return
        n = self.input_batch.num_reqs
        if n <= 0 or ib.num_reqs < n:
            return
        ib.num_accepted_tokens_cpu[:n] = self.input_batch.num_accepted_tokens_cpu[:n]
        self.intermediate_num_accepted_tokens.np[:n] = (
            self.input_batch.num_accepted_tokens_cpu[:n]
        )
        self.intermediate_num_accepted_tokens.copy_to_gpu(n)

    def _reconcile_intermediate_frontier_after_target(
        self,
        scheduler_output: "SchedulerOutput",
        req_ids_in_output_order: list[str],
        *,
        sampled_token_ids: torch.Tensor | None = None,
        spec_decode_metadata: SpecDecodeMetadata | None = None,
    ) -> None:
        """Align intermediate mirror with target after bookkeeping (drop provisional).

        Called from ``sample_tokens`` **after** ``_bookkeeping_sync`` so
        ``self.requests`` / ``output_token_ids`` match the scheduler-visible target
        state. An earlier hook right after ``_update_states_after_model_execute`` is
        not used here because hybrid bookkeeping has not yet appended accepted tokens
        to ``output_token_ids``; reconciling too early would fight the authoritative
        bookkeeping pass. Hybrid ``num_accepted_tokens`` are mirrored earlier via
        ``_sync_intermediate_num_accepted_from_target``. See
        ``_intermediate_shallow_req_clone`` docstring for physical KV policy.

        Ordering note: reconciling here (post-bookkeeping) means the authoritative
        ``num_computed_tokens`` / ``output_token_ids`` on ``self.requests`` already
        include target-verified acceptance; regression tests should assume mirror
        truncation is relative to that snapshot, not to the pre-bookkeeping hybrid view.
        """
        if not self._intermediate_kv_frontier_enabled():
            return
        ib = self.intermediate_input_batch
        assert ib is not None
        scheduled_spec = scheduler_output.scheduled_spec_decode_tokens
        for rid in req_ids_in_output_order:
            if rid not in self.intermediate_requests or rid not in self.requests:
                continue
            tgt = self.requests[rid]
            self.intermediate_committed_tokens[rid] = tgt.num_computed_tokens
            mir = self.intermediate_requests[rid]
            mir.num_computed_tokens = tgt.num_computed_tokens
            mir.output_token_ids[:] = list(tgt.output_token_ids)
            midx = ib.req_id_to_index.get(rid)
            if midx is not None:
                ib.num_computed_tokens_cpu[midx] = tgt.num_computed_tokens
                ib.update_req_spec_token_ids(mir, scheduled_spec)
        n_tgt = self.input_batch.num_reqs
        if n_tgt > 0 and ib.num_reqs >= n_tgt:
            ib.num_accepted_tokens_cpu[:n_tgt] = (
                self.input_batch.num_accepted_tokens_cpu[:n_tgt]
            )
        if (
            self.intermediate_num_accepted_tokens is not None
            and sampled_token_ids is not None
            and spec_decode_metadata is not None
        ):
            nd = spec_decode_metadata.num_draft_tokens
            nrows = int(sampled_token_ids.shape[0])
            if nd is not None and len(nd) == nrows:
                lens = get_target_verification_accepted_draft_prefix_lens(
                    sampled_token_ids,
                    list(nd),
                    placeholder_token_id=PLACEHOLDER_TOKEN_ID,
                )
                for i, rid in enumerate(self.input_batch.req_ids[: len(lens)]):
                    midx = ib.req_id_to_index.get(rid)
                    if midx is not None:
                        self.intermediate_num_accepted_tokens.np[midx] = int(lens[i])
                self.intermediate_num_accepted_tokens.copy_to_gpu(ib.num_reqs)
        ib.refresh_metadata()
        if ib.num_reqs > 0:
            ib.block_table.commit_block_table(ib.num_reqs)
        if self._is_dit_debug_enabled():
            for rid in self.input_batch.req_ids[: self.input_batch.num_reqs]:
                if rid not in self.intermediate_requests or rid not in self.requests:
                    continue
                mir = self.intermediate_requests[rid]
                tgt = self.requests[rid]
                self._dit_debug_assert(
                    mir.num_computed_tokens == tgt.num_computed_tokens,
                    "intermediate_reconcile_num_computed_matches_target",
                    detail=(
                        f"req_id={rid}, mir_num_computed={mir.num_computed_tokens}, "
                        f"tgt_num_computed={tgt.num_computed_tokens}"
                    ),
                )
                self._dit_debug_assert(
                    list(mir.output_token_ids) == list(tgt.output_token_ids),
                    "intermediate_reconcile_output_tokens_match_target",
                    detail=f"req_id={rid}",
                )

    def _draft_shallow_req_clone(self, src: CachedRequestState) -> CachedRequestState:
        """Clone request state for the draft HV frontier; same policy as intermediate."""
        return self._intermediate_shallow_req_clone(src)

    def _sync_draft_states_with_scheduler(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Sync scheduler-driven lifecycle onto the draft frontier batch."""
        if not self._draft_kv_frontier_enabled():
            return
        ib = self.draft_input_batch
        assert ib is not None
        for req_id in scheduler_output.finished_req_ids:
            self.draft_requests.pop(req_id, None)
            self.draft_committed_tokens.pop(req_id, None)
            if req_id in ib.req_id_to_index:
                ib.remove_request(req_id)
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = set(ib.req_id_to_index.keys())
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        unscheduled_req_ids = cached_req_ids - (
            set(scheduled_req_ids) - set(resumed_req_ids)
        )
        for req_id in unscheduled_req_ids:
            ib.remove_request(req_id)

        active = set(self.input_batch.req_id_to_index.keys())
        for req_id in list(ib.req_id_to_index.keys()):
            if req_id not in active:
                ib.remove_request(req_id)

        scheduled_spec = scheduler_output.scheduled_spec_decode_tokens
        for req_index in range(self.input_batch.num_reqs):
            req_id = self.input_batch.req_ids[req_index]
            tgt = self.requests[req_id]
            committed = self.draft_committed_tokens.get(
                req_id, tgt.num_computed_tokens
            )
            if req_id in ib.req_id_to_index:
                mir = self.draft_requests[req_id]
                mir.num_computed_tokens = committed
                mir_idx = ib.req_id_to_index[req_id]
                ib.num_computed_tokens_cpu[mir_idx] = committed
                ib.update_req_spec_token_ids(mir, scheduled_spec)
            else:
                mir = self.draft_requests.get(req_id)
                if mir is None:
                    mir = self._draft_shallow_req_clone(tgt)
                    self.draft_requests[req_id] = mir
                mir.num_computed_tokens = committed
                ib.add_request(mir)
                mir_idx = ib.req_id_to_index[req_id]
                ib.num_computed_tokens_cpu[mir_idx] = committed
                ib.update_req_spec_token_ids(mir, scheduled_spec)
        ib.condense()
        ib.refresh_metadata()

    def _prepare_draft_metadata(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
        req_ids_subset: list[str] | None = None,
    ) -> IntermediateFrontierPrepareResult | None:
        """Build step-shaped attention + spec metadata using the draft frontier batch."""
        if not self._draft_kv_frontier_enabled():
            return None
        if req_ids_subset is not None:
            return None
        ib = self.draft_input_batch
        assert ib is not None
        if self.input_batch.prev_sampled_token_ids is not None:
            return None
        if self.cache_config.mamba_cache_mode == "align":
            return None
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return None
        num_reqs_ib = ib.num_reqs
        if (
            num_reqs_ib != self.input_batch.num_reqs
            or num_reqs_ib <= 0
            or tuple(ib.req_ids[:num_reqs_ib])
            != tuple(self.input_batch.req_ids[:num_reqs_ib])
        ):
            return None
        total_tok = scheduler_output.total_num_scheduled_tokens
        if total_tok <= 0:
            return None

        out_b = {
            "positions": self.draft_step_positions,
            "input_ids": self.draft_step_input_ids,
            "query_start_loc": self.draft_step_query_start_loc,
            "seq_lens": self.draft_step_seq_lens,
            "discard_request_mask": self.draft_step_discard_request_mask,
            "num_decode_draft_tokens": self.draft_step_num_decode_draft_tokens,
        }
        logits_indices, spec_decode_metadata = self._prepare_inputs(
            scheduler_output,
            num_scheduled_tokens,
            input_batch=ib,
            requests=self.draft_requests,
            out_buffers=out_b,
            skip_lora_swap=True,
        )

        cascade_attn_prefix_lens = None
        if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
            cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                num_scheduled_tokens,
                ib.num_computed_tokens_cpu[:num_reqs_ib],
                scheduler_output.num_common_prefix_blocks,
            )

        num_reqs = num_reqs_ib
        num_tokens_unpadded = total_tok
        max_num_scheduled_tokens = int(num_scheduled_tokens.max())
        (
            cudagraph_mode,
            batch_desc,
            should_ubatch,
            _num_tokens_across_dp,
            _cudagraph_stats,
        ) = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=cascade_attn_prefix_lens is not None,
            num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
        )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.parallel_config.num_ubatches,
        )

        has_separate_kv_update = not all(
            all(
                g.backend.forward_includes_kv_cache_update
                for g in self.attn_groups[id]
            )
            for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
            if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
        )
        pad_attn = cudagraph_mode == CUDAGraphMode.FULL

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=num_tokens_padded
            if pad_attn or has_separate_kv_update
            else num_tokens_unpadded,
            num_reqs_padded=(
                num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
            ),
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
            input_batch=ib,
        )

        nat_buf = self.draft_num_accepted_tokens or self.num_accepted_tokens
        attn_build = self._build_attention_metadata(
            num_tokens=num_tokens_unpadded,
            num_tokens_padded=num_tokens_padded if pad_attn else None,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded if pad_attn else None,
            max_query_len=max_num_scheduled_tokens,
            ubatch_slices=ubatch_slices_attn,
            logits_indices=logits_indices,
            use_spec_decode=use_spec_decode,
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
            cascade_attn_prefix_lens=cascade_attn_prefix_lens,
            slot_mappings=slot_mappings_by_group,
            input_batch=ib,
            common_attn_query_start_loc=self.draft_step_query_start_loc,
            common_attn_seq_lens=self.draft_step_seq_lens,
            common_attn_num_accepted_tokens=nat_buf,
            common_attn_num_decode_draft_tokens=self.draft_step_num_decode_draft_tokens,
        )

        return IntermediateFrontierPrepareResult(
            logits_indices=logits_indices,
            spec_decode_metadata=spec_decode_metadata,
            attn_metadata=attn_build.attn_metadata,
            spec_decode_common_attn_metadata=attn_build.spec_decode_common_attn_metadata,
            slot_mappings_by_layer=slot_mappings,
            spec_decode_common_attn_metadata_by_gid=(
                attn_build.spec_decode_common_attn_metadata_by_gid
            ),
        )

    def _advance_draft_frontier_after_round(
        self,
        decision: DitRoundDecision,
        *,
        batch_size: int,
        eff_bs: int,
        pivot_expansion_plan: PivotExpansionPlan | None,
        before_prefix_lens: list[int],
    ) -> None:
        """Advance draft working frontier after inner-round acceptance."""
        if not self._draft_kv_frontier_enabled():
            return
        if pivot_expansion_plan is not None and eff_bs != batch_size:
            return
        for b, emitted in enumerate(decision.emitted_rows):
            if not emitted:
                continue
            origin_b = (
                int(pivot_expansion_plan.expanded_to_origin[b])
                if pivot_expansion_plan is not None
                else b
            )
            if origin_b < 0 or origin_b >= batch_size:
                continue
            req_id = self.input_batch.req_ids[origin_b]
            self._intermediate_assert_round_prefix_budget(
                prefix_len=before_prefix_lens[b],
                extra_accepted=len(emitted),
                row_detail=f"draft_round_row={b}, req_id={req_id}",
            )
            if req_id not in self.draft_requests:
                continue
            mir = self.draft_requests[req_id]
            mir.num_computed_tokens += len(emitted)
            mir.output_token_ids.extend(emitted)
            mir_idx = self.draft_input_batch.req_id_to_index.get(req_id)
            if mir_idx is None:
                continue
            ib = self.draft_input_batch
            ib.num_computed_tokens_cpu[mir_idx] = mir.num_computed_tokens
        self.draft_input_batch.refresh_metadata()

    def _sync_draft_num_accepted_from_target(self) -> None:
        """Mirror hybrid ``num_accepted_tokens`` rows onto the draft batch."""
        if not self._draft_kv_frontier_enabled():
            return
        ib = self.draft_input_batch
        if ib is None or self.draft_num_accepted_tokens is None:
            return
        if not self.model_config.is_hybrid:
            return
        n = self.input_batch.num_reqs
        if n <= 0 or ib.num_reqs < n:
            return
        ib.num_accepted_tokens_cpu[:n] = self.input_batch.num_accepted_tokens_cpu[:n]
        self.draft_num_accepted_tokens.np[:n] = (
            self.input_batch.num_accepted_tokens_cpu[:n]
        )
        self.draft_num_accepted_tokens.copy_to_gpu(n)

    def _reconcile_draft_frontier_after_target(
        self,
        scheduler_output: "SchedulerOutput",
        req_ids_in_output_order: list[str],
        *,
        sampled_token_ids: torch.Tensor | None = None,
        spec_decode_metadata: SpecDecodeMetadata | None = None,
    ) -> None:
        """Align draft mirror with target after bookkeeping (same contract as intermediate)."""
        if not self._draft_kv_frontier_enabled():
            return
        ib = self.draft_input_batch
        assert ib is not None
        scheduled_spec = scheduler_output.scheduled_spec_decode_tokens
        for rid in req_ids_in_output_order:
            if rid not in self.draft_requests or rid not in self.requests:
                continue
            tgt = self.requests[rid]
            self.draft_committed_tokens[rid] = tgt.num_computed_tokens
            mir = self.draft_requests[rid]
            mir.num_computed_tokens = tgt.num_computed_tokens
            mir.output_token_ids[:] = list(tgt.output_token_ids)
            midx = ib.req_id_to_index.get(rid)
            if midx is not None:
                ib.num_computed_tokens_cpu[midx] = tgt.num_computed_tokens
                ib.update_req_spec_token_ids(mir, scheduled_spec)
        n_tgt = self.input_batch.num_reqs
        if n_tgt > 0 and ib.num_reqs >= n_tgt:
            ib.num_accepted_tokens_cpu[:n_tgt] = (
                self.input_batch.num_accepted_tokens_cpu[:n_tgt]
            )
        if (
            self.draft_num_accepted_tokens is not None
            and sampled_token_ids is not None
            and spec_decode_metadata is not None
        ):
            nd = spec_decode_metadata.num_draft_tokens
            nrows = int(sampled_token_ids.shape[0])
            if nd is not None and len(nd) == nrows:
                lens = get_target_verification_accepted_draft_prefix_lens(
                    sampled_token_ids,
                    list(nd),
                    placeholder_token_id=PLACEHOLDER_TOKEN_ID,
                )
                for i, rid in enumerate(self.input_batch.req_ids[: len(lens)]):
                    midx = ib.req_id_to_index.get(rid)
                    if midx is not None:
                        self.draft_num_accepted_tokens.np[midx] = int(lens[i])
                self.draft_num_accepted_tokens.copy_to_gpu(ib.num_reqs)
        ib.refresh_metadata()
        if ib.num_reqs > 0:
            ib.block_table.commit_block_table(ib.num_reqs)
        if self._is_dit_debug_enabled():
            for rid in self.input_batch.req_ids[: self.input_batch.num_reqs]:
                if rid not in self.draft_requests or rid not in self.requests:
                    continue
                mir = self.draft_requests[rid]
                tgt = self.requests[rid]
                self._dit_debug_assert(
                    mir.num_computed_tokens == tgt.num_computed_tokens,
                    "draft_reconcile_num_computed_matches_target",
                    detail=(
                        f"req_id={rid}, mir_num_computed={mir.num_computed_tokens}, "
                        f"tgt_num_computed={tgt.num_computed_tokens}"
                    ),
                )
                self._dit_debug_assert(
                    list(mir.output_token_ids) == list(tgt.output_token_ids),
                    "draft_reconcile_output_tokens_match_target",
                    detail=f"req_id={rid}",
                )

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        if scheduler_output.new_block_ids_to_zero:
            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            if req_id in self.requests:
                # For streaming case only.
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            if self.uses_xdrope_dim > 0:
                self._init_xdrope_positions(req_state)

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Wait until valid_sampled_tokens_count is copied to cpu,
        # then use it to update actual num_computed_tokens of each request.
        valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            if req_state.prev_num_draft_len and self.use_async_scheduling:
                # prev_num_draft_len is used in async scheduling mode with
                # spec decode. it indicates if need to update num_computed_tokens
                # of the request. for example:
                # fist step: num_computed_tokens = 0, spec_tokens = [],
                # prev_num_draft_len = 0.
                # second step: num_computed_tokens = 100(prompt lenth),
                # spec_tokens = [a,b], prev_num_draft_len = 0.
                # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
                # prev_num_draft_len = 2.
                # num_computed_tokens in first step and second step does't contain
                # the spec tokens length, but in third step it contains the
                # spec tokens length. we only need to update num_computed_tokens
                # when prev_num_draft_len > 0.
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    req_state.output_token_ids.extend([-1] * num_accepted)

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                if not req_data.new_token_ids:
                    # Async scheduled PP: Sampled tokens propagated via GPU broadcast.
                    new_token_ids: list[int] = []
                else:
                    # Non-async scheduling with PP: The scheduler sends
                    # sampled token ids back because there's no direct communication
                    # between the first-stage worker and the last-stage worker.
                    new_token_ids = req_data.new_token_ids[i]
                    # Add the sampled token(s) from the previous step (if any).
                    # This doesn't include "unverified" tokens like spec tokens.
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        # Avoid slicing list in most common case.
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )
            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure. Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                if self.use_async_scheduling and num_output_tokens > 0:
                    # We must recover the output token ids for resumed requests in the
                    # async scheduling case, so that correct input_ids are obtained.
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Update the cached states after model execution.

        This is used for MTP/EAGLE for hybrid models, as in linear attention,
        only the last token's state is kept. In MTP/EAGLE, for draft tokens
        the state are kept util we decide how many tokens are accepted for
        each sequence, and a shifting is done during the next iteration
        based on the number of accepted tokens.
        """
        if not self.speculative_config or not self.model_config.is_hybrid:
            return

        # Find the number of accepted tokens for each sequence.
        num_reqs = output_token_ids.size(0)
        self.num_accepted_tokens.gpu[:num_reqs] = (
            (
                torch.cat(
                    [
                        output_token_ids,
                        torch.full(
                            (num_reqs, 1),
                            -1,
                            device=output_token_ids.device,
                        ),
                    ],
                    dim=1,
                )
                == -1
            )
            .int()
            .argmax(-1)
        )
        if self.cache_config.mamba_cache_mode == "align":
            for i, num_tokens in enumerate(
                self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()
            ):
                self.input_batch.num_accepted_tokens_cpu[i] = num_tokens

            mamba_utils.postprocess_mamba(
                scheduler_output,
                self.kv_cache_config,
                self.input_batch,
                self.requests,
                self.mamba_state_idx,
                self.compilation_config.static_forward_context,
                self.model.get_mamba_state_copy_func(),
                self._get_mamba_copy_bufs(),
            )
        else:
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
            )

    def _update_streaming_request(
        self, req_id: str, new_req_data: NewRequestData
    ) -> CachedRequestState:
        """Updates streaming session request from `scheduled_new_reqs`.

        Removes the request from InputBatch (if present), updates the cached
        state, and prepares it for re-addition to the batch.

        NOTE: prompt_token_ids includes intermediate output tokens - tokens
        previously generated but now are input context (part of the prompt).
        """
        self.input_batch.remove_request(req_id)
        req_state = self.requests[req_id]

        req_state.prompt_token_ids = new_req_data.prompt_token_ids
        req_state.mm_features = new_req_data.mm_features
        req_state.prompt_embeds = new_req_data.prompt_embeds
        req_state.sampling_params = new_req_data.sampling_params
        req_state.pooling_params = new_req_data.pooling_params
        req_state.block_ids = new_req_data.block_ids
        req_state.num_computed_tokens = new_req_data.num_computed_tokens
        req_state.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # Clear `output_token_ids` as previous output tokens are now part of
        # `prompt_token_ids`.
        req_state.output_token_ids.clear()

        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        return req_state

    def _init_mrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        assert supports_mrope(model), "M-RoPE support is not implemented."
        assert req_state.prompt_token_ids is not None, (
            "M-RoPE requires prompt_token_ids to be available."
        )
        mrope_model = cast(SupportsMRoPE, model)

        req_state.mrope_positions, req_state.mrope_position_delta = (
            mrope_model.get_mrope_input_positions(
                req_state.prompt_token_ids,
                req_state.mm_features,
            )
        )

    def _init_xdrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        xdrope_model = cast(SupportsXDRoPE, model)
        assert req_state.prompt_token_ids is not None, (
            "XD-RoPE requires prompt_token_ids to be available."
        )
        assert supports_xdrope(model), "XD-RoPE support is not implemented."

        req_state.xdrope_positions = xdrope_model.get_xdrope_input_positions(
            req_state.prompt_token_ids,
            req_state.mm_features,
        )

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:
            return {}

        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        for req in scheduler_output.scheduled_new_reqs:
            for feature in req.mm_features:
                if feature.data is not None:
                    mm_kwargs.append((feature.modality, feature.data))

        # Input all modalities at once
        mm_kwargs_combined: BatchedTensorInputs = {}
        for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
        ):
            mm_kwargs_combined.update(mm_kwargs_group)

        return mm_kwargs_combined

    def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
        if not self.is_multimodal_raw_input_only_model:
            return {}

        mm_budget = self.mm_budget
        assert mm_budget is not None

        if not mm_budget.mm_max_toks_per_item:
            return {}  # No tower modalities (embed-only mode)

        dummy_modality = mm_budget.get_modality_with_max_tokens()
        return self._get_mm_dummy_batch(dummy_modality, num_seqs)

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_input_ids(
        self,
        scheduler_output: "SchedulerOutput",
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
        *,
        input_batch: InputBatch | None = None,
        input_ids_buffer: CpuGpuBuffer | None = None,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        ib = input_batch if input_batch is not None else self.input_batch
        ids_b = input_ids_buffer if input_ids_buffer is not None else self.input_ids

        if ib.prev_sampled_token_ids is None:
            # Normal scheduling case
            ids_b.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = ib.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in ib.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
                # sample_flattened_indices = [0, 2, 5]
                # spec_flattened_indices = [1,   3, 4,    6, 7]
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(
                    range(flattened_index - draft_len + 1, flattened_index + 1)
                )
                start = prev_index * self.num_spec_tokens
                # prev_draft_token_indices is used to find which draft_tokens_id
                # should be copied to input_ids
                # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
                # flatten draft_tokens_id [1,2,3,4,5,6]
                # draft_len of each request [1, 2, 1]
                # then prev_draft_token_indices is [0,   2, 3,   4]
                prev_draft_token_indices.extend(range(start, start + draft_len))
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_commmon_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if num_commmon_tokens < total_without_spec:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the GPU first.
            ids_b.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_commmon_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            ids_b.gpu[:num_commmon_tokens].copy_(
                ib.prev_sampled_token_ids[:num_commmon_tokens, 0],
                non_blocking=True,
            )
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_commmon_tokens] = True
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        ids_b.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=ib.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        if self._draft_token_ids is None or not spec_flattened_indices:
            return

        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)

        ids_b.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    def _get_encoder_seq_lens(
        self,
        num_scheduled_tokens: dict[str, int],
        kv_cache_spec: KVCacheSpec,
        num_reqs: int,
        for_cudagraph_capture: bool = False,
    ) -> tuple[torch.Tensor | None, np.ndarray | None]:
        if not isinstance(kv_cache_spec, CrossAttentionSpec):
            return None, None

        # Zero out buffer for padding requests that are not actually scheduled (CGs)
        self.encoder_seq_lens.np[:num_reqs] = 0

        # Build encoder_seq_lens array mapping request indices to
        # encoder lengths for inputs scheduled in this batch
        for req_id in num_scheduled_tokens:
            req_index = self.input_batch.req_id_to_index[req_id]
            req_state = self.requests[req_id]
            if req_state.mm_features is None:
                self.encoder_seq_lens.np[req_index] = 0
                continue

            # Get the total number of encoder input tokens for running encoder requests
            # whether encoding is finished or not so that cross-attention knows how
            # many encoder tokens to attend to.
            encoder_input_tokens = sum(
                feature.mm_position.length for feature in req_state.mm_features
            )
            self.encoder_seq_lens.np[req_index] = encoder_input_tokens
        if for_cudagraph_capture:
            # During CUDA graph capture, we need to use realistic encoder lengths
            # so that max_seqlen_k is captured with the correct value.
            max_encoder_len = getattr(
                self.model_config.hf_config,
                "max_source_positions",
                self.max_encoder_len,
            )
            self.encoder_seq_lens.np[:num_reqs] = max_encoder_len

        self.encoder_seq_lens.copy_to_gpu(num_reqs)
        encoder_seq_lens = self.encoder_seq_lens.gpu[:num_reqs]
        encoder_seq_lens_cpu = self.encoder_seq_lens.np[:num_reqs]

        return encoder_seq_lens, encoder_seq_lens_cpu

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
        *,
        input_batch: InputBatch | None = None,
        requests: dict[str, CachedRequestState] | None = None,
        out_buffers: dict[str, CpuGpuBuffer] | None = None,
        skip_lora_swap: bool = False,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
    ]:
        """
        :return: tuple[
            logits_indices, spec_decode_metadata,
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        ib = self.input_batch if input_batch is None else input_batch
        rq = self.requests if requests is None else requests
        if out_buffers is None:
            pos_buf = self.positions
            in_buf = self.input_ids
            qsl_buf = self.query_start_loc
            sl_buf = self.seq_lens
            drm_buf = self.discard_request_mask
            nddt_buf = self.num_decode_draft_tokens
        else:
            assert not ib.req_prompt_embeds, (
                "scratch _prepare_inputs requires token-id rows (no req_prompt_embeds)"
            )
            assert not self.uses_mrope and self.uses_xdrope_dim == 0, (
                "scratch _prepare_inputs does not support M-RoPE / XD-RoPE yet"
            )
            assert not self.enable_prompt_embeds, (
                "scratch _prepare_inputs does not support prompt embeds yet"
            )
            pos_buf = out_buffers["positions"]
            in_buf = out_buffers["input_ids"]
            qsl_buf = out_buffers["query_start_loc"]
            sl_buf = out_buffers["seq_lens"]
            drm_buf = out_buffers["discard_request_mask"]
            nddt_buf = out_buffers["num_decode_draft_tokens"]

        num_reqs = ib.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        ib.block_table.commit_block_table(num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

        # Get positions.
        positions_np = pos_buf.np[:total_num_scheduled_tokens]
        np.add(
            ib.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np,
        )

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output, input_batch=ib, requests=rq)

        # Calculate XD-RoPE positions.
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output, input_batch=ib, requests=rq)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (
            positions_np + req_indices * ib.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            ib.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=in_buf.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = ib.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
            )

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if ib.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in ib.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = ib.req_prompt_embeds[req_idx]
                start_pos = ib.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[
                        output_idx : output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])

                output_idx += num_sched

        ib.block_table.compute_slot_mapping(req_indices, positions_np)
        ib.block_table.commit_slot_mapping(total_num_scheduled_tokens)

        # Prepare the attention metadata.
        qsl_buf.np[0] = 0
        qsl_buf.np[1 : num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        qsl_buf.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        qsl_buf.copy_to_gpu()
        query_start_loc = qsl_buf.gpu[: num_reqs + 1]

        sl_buf.np[:num_reqs] = (
            ib.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        )
        # Fill unused with 0 for full cuda graph mode.
        sl_buf.np[num_reqs:].fill(0)
        sl_buf.copy_to_gpu()

        num_tokens = [rq[r].num_tokens for r in ib.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # Record which requests should not be sampled,
        # so that we could clear the sampled tokens before returning
        drm_buf.np[:num_reqs] = (
            sl_buf.np[:num_reqs] < num_tokens_np
        )
        drm_buf.copy_to_gpu(num_reqs)

        # Copy the tensors to the GPU.
        self._prepare_input_ids(
            scheduler_output,
            total_num_scheduled_tokens,
            cu_num_tokens,
            input_batch=ib,
            input_ids_buffer=in_buf,
        )

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        else:
            # Common case (1D positions)
            pos_buf.copy_to_gpu(total_num_scheduled_tokens)

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = ib.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                if (
                    ib.num_computed_tokens_cpu[req_idx]
                    >= ib.num_prompt_tokens[req_idx]
                ):
                    num_decode_draft_tokens[req_idx] = len(draft_token_ids)
            pivot_plan = self.pending_pivot_expansion_plan
            pb = self.pending_hybrid_spec_bundle
            expected_p_pivot: int | None = None
            if (
                self.speculative_config is not None
                and self.speculative_config.method == "pivot"
            ):
                expected_p_pivot = (
                    self.speculative_config.pivot_packed_batch_size_for_origin_batch(
                        num_reqs
                    )
                )
            # Expanded pivot bundles are indexed by origin rows from propose-time;
            # if the scheduled batch shrinks, packed_sm_origin / expanded_to_origin
            # can go out of range.
            if (
                pivot_plan is not None
                and pb is not None
                and pivot_plan.expanded_batch_size > 0
                and len(pivot_plan.expanded_to_origin) == pivot_plan.expanded_batch_size
                and len(pb.num_draft_tokens) == pivot_plan.expanded_batch_size
            ):
                if pb.expansion_plan is not pivot_plan:
                    self._clear_pending_pivot_hybrid_at_prepare_boundary(
                        "bundle.expansion_plan != pending_pivot_expansion_plan",
                    )
                    pivot_plan, pb = None, None
                elif not pivot_expansion_indices_fit_prepare_batch(
                    pivot_plan,
                    num_reqs,
                    expected_packed_size=expected_p_pivot,
                ):
                    sm_idx = _pivot_plan_sm_indices(pivot_plan)
                    max_o = max(sm_idx) if sm_idx else -1
                    self._clear_pending_pivot_hybrid_at_prepare_boundary(
                        "pivot expansion plan does not fit current batch",
                        detail=f"num_reqs={num_reqs}, max_sm_origin_index={max_o}",
                    )
                    pivot_plan, pb = None, None
            use_pivot_expanded = (
                pivot_plan is not None
                and pb is not None
                and pivot_plan.expanded_batch_size > 0
                and len(pivot_plan.expanded_to_origin) == pivot_plan.expanded_batch_size
                and len(pb.num_draft_tokens) == pivot_plan.expanded_batch_size
            )
            if use_pivot_expanded:
                assert pb.expansion_plan is pivot_plan, (
                    "pending bundle expansion_plan must match pending pivot plan"
                )
                if pb.tree_plan is not None:
                    self._dit_debug_assert(
                        len(pb.num_draft_tokens) == len(pb.tree_plan.families),
                        "pivot_tree_prepare_inputs_family_count_match",
                    )
                assert int(pb.draft_token_ids.shape[0]) == int(
                    pb.cu_num_draft_tokens[-1].item()
                ), "bundle flat draft rows must match cu_num_draft_tokens tail"
                if (
                    pivot_plan.uses_fixed_capacity_packing
                    and pivot_plan.packed_sm_origin is not None
                    and pivot_plan.packed_row_is_active is not None
                ):
                    P = pivot_plan.packed_batch_size
                    num_draft_meta = np.zeros(P, dtype=np.int32)
                    cu_meta = np.zeros(P, dtype=np.int32)
                    sm = pivot_plan.packed_sm_origin
                    active = pivot_plan.packed_row_is_active
                    for j in range(P):
                        o = int(sm[j])
                        cu_meta[j] = cu_num_tokens[o]
                        if active[j]:
                            num_draft_meta[j] = num_draft_tokens[o]
                else:
                    num_draft_meta = num_draft_tokens[pivot_plan.expanded_to_origin]
                    cu_meta = cu_num_tokens[pivot_plan.expanded_to_origin]
                # Expanded rows may have fewer tokens than their origin's
                # collapsed count (intermediate verification can reject tokens
                # for some candidates while accepting for others).  Clamp the
                # per-row schedule count to the bundle's actual count so the
                # metadata stays in sync with the flat token tensor.
                pb_ndt = np.array(pb.num_draft_tokens, dtype=np.int32)
                needs_clamp = np.any(num_draft_meta != pb_ndt)
                if needs_clamp:
                    num_draft_meta = np.minimum(num_draft_meta, pb_ndt)
                    pb = _clamp_hybrid_bundle_to_max_counts(
                        pb, num_draft_meta
                    )
                    self.pending_hybrid_spec_bundle = pb
                spec_decode_metadata = self._calc_spec_decode_metadata(
                    num_draft_meta,
                    cu_meta,
                    expansion_plan=pivot_plan,
                )
                assert int(spec_decode_metadata.cu_num_draft_tokens[-1].item()) == int(
                    pb.draft_token_ids.shape[0]
                ), "metadata and bundle draft token counts must match"
                spec_decode_metadata = dataclass_replace(
                    spec_decode_metadata,
                    draft_token_ids=pb.draft_token_ids,
                )
            else:
                spec_decode_metadata = self._calc_spec_decode_metadata(
                    num_draft_tokens,
                    cu_num_tokens,
                    expansion_plan=None,
                )
                # Origin-only metadata: do **not** clear the pending bundle just because
                # len(pb.num_draft_tokens) != num_reqs. Batch shrink / reorder is handled
                # in _sample() via remap_hybrid_bundle_rows_for_metadata (subset survivor).
                # Only clear when the bundle still carries an expansion_plan while this
                # prepare path cannot use packed pivot metadata (true contract mismatch).
                if pb is not None and pb.expansion_plan is not None:
                    self._clear_pending_pivot_hybrid_at_prepare_boundary(
                        "stale hybrid bundle vs origin-only spec metadata",
                        detail=(
                            f"bundle_rows={len(pb.num_draft_tokens)}, "
                            f"num_reqs={num_reqs}, "
                            "bundle has expansion_plan but prepare uses origin-only metadata"
                        ),
                    )
            logits_indices = spec_decode_metadata.logits_indices
            if (
                use_pivot_expanded
                and pivot_plan is not None
                and pivot_plan.uses_fixed_capacity_packing
            ):
                # LoRA prompt mapping length must match spec verifier sample count
                # (sum over packed rows), not origin batch B.
                num_sampled_tokens = num_draft_meta.astype(np.int32) + 1
            else:
                num_sampled_tokens = num_draft_tokens + 1
            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            nddt_buf.np[:num_reqs] = num_decode_draft_tokens
            nddt_buf.np[num_reqs:].fill(-1)
            nddt_buf.copy_to_gpu()

        # Hot-Swap lora model
        if self.lora_config and not skip_lora_swap:
            assert (
                np.sum(num_sampled_tokens)
                <= self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            packed_lora = (
                use_spec_decode
                and spec_decode_metadata is not None
                and spec_decode_metadata.expansion_plan is not None
                and spec_decode_metadata.expansion_plan.uses_fixed_capacity_packing
                and spec_decode_metadata.expansion_plan.packed_sm_origin is not None
            )
            if packed_lora:
                exp_plan = spec_decode_metadata.expansion_plan
                assert exp_plan.packed_sm_origin is not None
                req_lora = ib.request_lora_mapping[:num_reqs]
                sm = exp_plan.packed_sm_origin
                P = exp_plan.packed_batch_size
                prompt_lora_mapping = tuple(
                    int(req_lora[int(sm[j])])
                    for j in range(P)
                    for _ in range(int(num_sampled_tokens[j]))
                )
                token_lora_mapping = tuple(
                    int(req_lora[i])
                    for i in range(num_reqs)
                    for _ in range(int(num_scheduled_tokens[i]))
                )
                self._set_active_loras(
                    prompt_lora_mapping,
                    token_lora_mapping,
                    set(ib.lora_id_to_lora_request.values()),
                    LoRAMappingType.LANGUAGE,
                )
            else:
                self.set_active_loras(
                    ib, num_scheduled_tokens, num_sampled_tokens
                )

        return (
            logits_indices,
            spec_decode_metadata,
        )

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        slot_mappings: dict[int, torch.Tensor] | None = None,
        input_batch: InputBatch | None = None,
        common_attn_query_start_loc: CpuGpuBuffer | None = None,
        common_attn_seq_lens: CpuGpuBuffer | None = None,
        common_attn_num_accepted_tokens: CpuGpuBuffer | None = None,
        common_attn_num_decode_draft_tokens: CpuGpuBuffer | None = None,
    ) -> AttentionMetadataBuildResult:
        """
        :return: ``AttentionMetadataBuildResult`` with per-``kv_cache_gid`` speculative
        ``CommonAttentionMetadata`` when spec decode is active.
        """
        ib = input_batch if input_batch is not None else self.input_batch
        qsl_cm = common_attn_query_start_loc or self.query_start_loc
        sl_cm = common_attn_seq_lens or self.seq_lens
        nat_cm = common_attn_num_accepted_tokens or self.num_accepted_tokens
        nddt_cm = common_attn_num_decode_draft_tokens or self.num_decode_draft_tokens
        # Attention metadata is not needed for attention free models
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return AttentionMetadataBuildResult({}, None, {})

        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        assert num_reqs_padded is not None and num_tokens_padded is not None

        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = sl_cm.np[:num_reqs].max().item()

        if use_spec_decode:
            nat_cm.np[:num_reqs] = (
                ib.num_accepted_tokens_cpu[:num_reqs]
            )
            nat_cm.np[num_reqs:].fill(1)
            nat_cm.copy_to_gpu()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_block_table(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                blk_table = ib.block_table[kv_cache_gid]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(-1)
            return blk_table_tensor

        assert slot_mappings is not None
        block_table_gid_0 = _get_block_table(0)
        slot_mapping_gid_0 = slot_mappings[0]

        if self.model_config.enable_return_routed_experts:
            self.slot_mapping = slot_mapping_gid_0[:num_tokens].cpu().numpy()
        cm_base = CommonAttentionMetadata(
            query_start_loc=qsl_cm.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=qsl_cm.cpu[: num_reqs_padded + 1],
            seq_lens=sl_cm.gpu[:num_reqs_padded],
            _seq_lens_cpu=sl_cm.cpu[:num_reqs_padded],
            _num_computed_tokens_cpu=ib.num_computed_tokens_cpu_tensor[:num_reqs_padded],
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
        )

        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
                sl_cm.cpu[:num_reqs],
                self.dcp_world_size,
                self.dcp_rank,
                self.parallel_config.cp_kv_cache_interleave_size,
            )
            self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
            self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)

            cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
            cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[
                :num_reqs_padded
            ]

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices
            )

        # Cache attention metadata builds across hybrid KV-cache groups
        # The only thing that changes between different hybrid KV-cache groups when the
        # same metadata builder and KVCacheSpec is the same is the block table, so we
        # can cache the attention metadata builds and just update the block table using
        # `builder.update_block_table` if the builder supports it.
        cached_attn_metadata: dict[
            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
        ] = {}

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            ubid: int | None = None,
        ) -> None:
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            builder = attn_group.get_metadata_builder(ubid or 0)
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
            cache_key = (kv_cache_spec, type(builder))

            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                if cascade_attn_prefix_lens
                else 0
            )

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(
                builder, (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)
            ):
                assert ubid is None, "UBatching not supported with GDN yet"
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=nat_cm.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=nddt_cm.cpu[
                        :num_reqs_padded
                    ],
                )

            if for_cudagraph_capture:
                attn_metadata_i = builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            elif (
                cache_key in cached_attn_metadata
                and builder.supports_update_block_table
            ):
                attn_metadata_i = builder.update_block_table(
                    cached_attn_metadata[cache_key],
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.slot_mapping,
                )
            else:
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )
                if builder.supports_update_block_table:
                    cached_attn_metadata[cache_key] = attn_metadata_i

            if ubid is None:
                assert isinstance(attn_metadata, dict)
                attn_metadata_dict = attn_metadata
            else:
                assert isinstance(attn_metadata, list)
                attn_metadata_dict = attn_metadata[ubid]

            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = attn_metadata_i

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        spec_decode_common_attn_metadata = None
        spec_decode_cad_by_gid: dict[int, CommonAttentionMetadata] = {}
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
            cm = copy(cm_base)  # shallow copy

            # Basically only the encoder seq_lens, block_table and slot_mapping change
            # for each kv_cache_group.
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
                for_cudagraph_capture=for_cudagraph_capture,
            )
            if kv_cache_gid > 0:
                cm.block_table_tensor = _get_block_table(kv_cache_gid)
                cm.slot_mapping = slot_mappings[kv_cache_gid]

            spec_decode_cad_by_gid[int(kv_cache_gid)] = cm

            if (
                self.speculative_config
                and self.drafter is not None
                and spec_decode_common_attn_metadata is None
            ):
                drafter_kv_gid = getattr(self.drafter, "kv_cache_gid", None)
                if drafter_kv_gid is not None:
                    if int(drafter_kv_gid) == kv_cache_gid:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm

            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                if ubatch_slices is not None:
                    for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):
                        _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)

                else:
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)

        if self.is_mm_prefix_lm:
            req_doc_ranges = {}
            for req_index in range(ib.num_reqs):
                req_id = ib.req_ids[req_index]
                image_doc_ranges = []
                req_state = self.requests[req_id]
                for mm_feature in req_state.mm_features:
                    pos_info = mm_feature.mm_position
                    img_doc_range = pos_info.extract_embeds_range()
                    image_doc_ranges.extend(img_doc_range)
                req_idx = ib.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges

            if isinstance(attn_metadata, list):
                for ub_metadata in attn_metadata:
                    for _metadata in ub_metadata.values():
                        _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]
            else:
                for _metadata in attn_metadata.values():
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]

        if spec_decode_common_attn_metadata is not None and (
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
        ):
            # Currently the drafter still only uses piecewise cudagraphs (and modifies
            # the attention metadata in directly), and therefore does not want to use
            # padded attention metadata.
            spec_decode_common_attn_metadata = (
                spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
            )
            spec_decode_cad_by_gid = {
                gid: cm_i.unpadded(num_tokens, num_reqs)
                for gid, cm_i in spec_decode_cad_by_gid.items()
            }

        return AttentionMetadataBuildResult(
            attn_metadata=attn_metadata,
            spec_decode_common_attn_metadata=spec_decode_common_attn_metadata,
            spec_decode_common_attn_metadata_by_gid=spec_decode_cad_by_gid,
        )

    def _compute_cascade_attn_prefix_lens(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: list[int],
    ) -> list[list[int]] | None:
        """
        :return: Optional[cascade_attn_prefix_lens]
            cascade_attn_prefix_lens is 2D: ``[kv_cache_group_id][attn_group_idx]``,
            None if we should not use cascade attention
        """

        use_cascade_attn = False
        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        cascade_attn_prefix_lens: list[list[int]] = [
            [] for _ in range(num_kv_cache_groups)
        ]

        for kv_cache_gid in range(num_kv_cache_groups):
            for attn_group in self.attn_groups[kv_cache_gid]:
                if isinstance(attn_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                    cascade_attn_prefix_len = 0
                else:
                    # 0 if cascade attention should not be used
                    cascade_attn_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_computed_tokens,
                        num_common_prefix_blocks[kv_cache_gid],
                        attn_group.kv_cache_spec,
                        attn_group.get_metadata_builder(),
                    )
                cascade_attn_prefix_lens[kv_cache_gid].append(cascade_attn_prefix_len)
                use_cascade_attn |= cascade_attn_prefix_len > 0

        return cascade_attn_prefix_lens if use_cascade_attn else None

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """Compute the length of the common prefix for cascade attention.

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """

        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            # Common case.
            return 0

        # NOTE(woosuk): Cascade attention uses two attention kernels: one
        # for the common prefix and the other for the rest. For the first
        # kernel, we concatenate all the query tokens (possibly from
        # different requests) and treat them as if they are from the same
        # request. Then, we use bi-directional attention to process the
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.

        # Consider the following example:
        # Request 1's input query: [D, E, X]
        # Request 1's kv cache: [A, B, C, D, E, X]
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        # Request 2's input query: [E, Y]
        # Request 2's kv cache: [A, B, C, D, E, Y]
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        # If we use [A, B, C, D, E] as the common prefix, then the
        # first kernel will compute the bi-directional attention between
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        # However, this is wrong because D in Request 1 should not attend to
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.
        # That is, the common prefix should be capped by the minimum
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.

        # In practice, we use [A, B, C] as the common prefix, instead of
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).
        # This is because of an implementation detail: We want to always
        # use two kernels for cascade attention. Let's imagine:
        # Request 3's input query: [D]
        # Request 3's kv cache: [A, B, C, D]
        # Request 3's num_computed_tokens: 3 (i.e., [A, B, C])
        # If we use [A, B, C, D] as the common prefix for Request 1-3,
        # then Request 3 will be processed only by the first kernel,
        # and the second kernel will get an empty input. While this is not
        # a fundamental problem, our current implementation does not support
        # this case.
        common_prefix_len = min(common_prefix_len, num_computed_tokens.min())
        # common_prefix_len should be a multiple of the block size.
        common_prefix_len = (
            common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size
        )
        use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.sliding_window is not None
        )
        use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.attention_chunk_size is not None
        )
        assert isinstance(kv_cache_spec, AttentionSpec)
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            use_local_attention=use_local_attention,
            num_sms=self.num_sms,
            dcp_world_size=self.dcp_world_size,
        )
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(
        self,
        scheduler_output: "SchedulerOutput",
        *,
        input_batch: InputBatch | None = None,
        requests: dict[str, CachedRequestState] | None = None,
    ):
        ib = input_batch if input_batch is not None else self.input_batch
        rq = requests if requests is not None else self.requests
        mrope_pos_ptr = 0
        for index, req_id in enumerate(ib.req_ids):
            req = rq[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = ib.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions.cpu[:, dst_start:dst_end] = req.mrope_positions[
                    :, src_start:src_end
                ]
                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                assert req.mrope_position_delta is not None
                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions.np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _calc_xdrope_positions(
        self,
        scheduler_output: "SchedulerOutput",
        *,
        input_batch: InputBatch | None = None,
        requests: dict[str, CachedRequestState] | None = None,
    ):
        ib = input_batch if input_batch is not None else self.input_batch
        rq = requests if requests is not None else self.requests
        xdrope_pos_ptr = 0
        for index, req_id in enumerate(ib.req_ids):
            req = rq[req_id]
            assert req.xdrope_positions is not None

            num_computed_tokens = ib.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's xdrope_positions are pre-computed
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.xdrope_positions.cpu[:, dst_start:dst_end] = req.xdrope_positions[
                    :, src_start:src_end
                ]
                xdrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's xdrope_positions on-the-fly
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + completion_part_len

                XDRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.xdrope_positions.np,
                    out_offset=dst_start,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                xdrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        *,
        expansion_plan: PivotExpansionPlan | None = None,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(
            self.device, non_blocking=True
        )
        logits_indices = torch.from_numpy(logits_indices).to(
            self.device, non_blocking=True
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
            expansion_plan=expansion_plan,
        )

    def _prepare_kv_sharing_fast_prefill(
        self,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_sharing_fast_prefill_logits_indices is not None
        num_logits = logits_indices.shape[0]
        assert num_logits > 0
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(logits_indices)
        # There might have leftover indices in logits_indices[num_logits:]
        # from previous iterations, whose values may be greater than the
        # batch size in the current iteration. To ensure indices are always
        # valid, we fill the padded indices with the last index.
        self.kv_sharing_fast_prefill_logits_indices[num_logits:].fill_(
            logits_indices[-1].item()
        )
        # Dispatch for the decoder portion of the model.
        _, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_logits, invalid_modes={CUDAGraphMode.FULL}
        )
        num_logits_padded = batch_desc.num_tokens
        logits_indices_padded = self.kv_sharing_fast_prefill_logits_indices[
            :num_logits_padded
        ]
        return logits_indices_padded

    def _batch_mm_inputs_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[
        list[str],
        list[tuple[str, MultiModalKwargsItem]],
        list[tuple[str, PlaceholderRange]],
    ]:
        """Batch multimodal inputs from scheduled encoder inputs.

        Args:
            scheduler_output: The scheduler output containing scheduled encoder
                inputs.

        Returns:
            A tuple of (mm_hashes, mm_kwargs, mm_lora_refs) where:
            - mm_hashes: List of multimodal hashes for each item
            - mm_kwargs: List of multimodal kwargs for each item
            - mm_lora_refs: List of (req_id, placeholder_range) for each item
        """
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return [], [], []

        mm_hashes = list[str]()
        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        # Multimodal LoRA reference info to map each multimodal item
        # back to its request & position
        mm_lora_refs = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue

                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append((mm_feature.modality, mm_feature.data))
                mm_lora_refs.append((req_id, mm_feature.mm_position))

        return mm_hashes, mm_kwargs, mm_lora_refs

    def _execute_mm_encoder(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[torch.Tensor]:
        mm_hashes, mm_kwargs, mm_lora_refs = self._batch_mm_inputs_from_scheduler(
            scheduler_output
        )

        if not mm_kwargs:
            return []

        should_time = bool(
            self.observability_config
            and self.observability_config.enable_mm_processor_stats
            and scheduler_output.scheduled_encoder_inputs
        )

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        model = cast(SupportsMultiModal, self.model)

        if self.lora_config and self.lora_manager.supports_tower_connector_lora():
            # Build LoRA mappings independently for encoder inputs
            # (encoder batch structure is different from main batch)
            prompt_lora_mapping = []
            token_lora_mapping = []
            lora_requests = set()
            encoder_token_counts = []

            for req_id, pos_info in mm_lora_refs:
                req_idx = self.input_batch.req_id_to_index[req_id]
                lora_id = int(self.input_batch.request_lora_mapping[req_idx])

                # Prefer pos_info.get_num_embeds to count precise MM embedding tokens.
                num_tokens = self.model.get_num_mm_encoder_tokens(  # type: ignore[attr-defined]
                    pos_info.get_num_embeds()
                )
                prompt_lora_mapping.append(lora_id)
                token_lora_mapping.extend([lora_id] * num_tokens)
                encoder_token_counts.append(num_tokens)

                if lora_id > 0:
                    lora_request = self.input_batch.lora_id_to_lora_request.get(lora_id)
                    if lora_request is not None:
                        lora_requests.add(lora_request)

            # Set tower adapter mapping
            tower_mapping = LoRAMapping(
                tuple(token_lora_mapping),
                tuple(prompt_lora_mapping),
                is_prefill=True,
                type=LoRAMappingType.TOWER,
            )
            self.lora_manager.set_active_adapters(lora_requests, tower_mapping)

            if hasattr(self.model, "get_num_mm_connector_tokens"):
                post_op_counts = [
                    self.model.get_num_mm_connector_tokens(num_tokens)  # type: ignore[attr-defined]
                    for num_tokens in encoder_token_counts
                ]

                connector_token_mapping = np.repeat(
                    np.array(prompt_lora_mapping, dtype=np.int32),
                    np.array(post_op_counts, dtype=np.int32),
                )
                connector_mapping = LoRAMapping(
                    index_mapping=tuple(connector_token_mapping.tolist()),
                    prompt_mapping=tuple(prompt_lora_mapping),
                    is_prefill=True,
                    type=LoRAMappingType.CONNECTOR,
                )

                self.lora_manager.set_active_adapters(
                    lora_requests,
                    connector_mapping,
                )

        encoder_outputs: list[torch.Tensor] = []
        # Track the current index in mm_kwargs/mm_lora_refs to map groups to request IDs
        current_item_idx = 0
        for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
        ):
            curr_group_outputs: MultiModalEmbeddings

            # EVS-related change.
            # (ekhvedchenia): Temporary hack to limit peak memory usage when
            # processing multimodal data. This solves the issue with scheduler
            # putting too many video samples into a single batch. Scheduler
            # uses pruned vision tokens count to compare it versus compute
            # budget which is incorrect (Either input media size or non-pruned
            # output vision tokens count should be considered)
            # TODO(ywang96): Fix memory profiling to take EVS into account and
            # remove this hack.
            if (
                self.is_multimodal_pruning_enabled
                and modality == "video"
                and num_items > 1
            ):
                curr_group_outputs_lst = list[torch.Tensor]()
                for video_idx in range(num_items):
                    video_mm_kwargs_item = mm_kwargs[current_item_idx + video_idx]
                    with self.timed_encoder_operation(
                        should_time, mm_lora_refs, current_item_idx + video_idx, 1
                    ):
                        _, _, micro_batch_mm_inputs = next(
                            group_mm_kwargs_by_modality(
                                [video_mm_kwargs_item],
                                device=self.device,
                                pin_memory=self.pin_memory,
                            )
                        )

                        micro_batch_outputs = model.embed_multimodal(
                            **micro_batch_mm_inputs
                        )

                        curr_group_outputs_lst.extend(micro_batch_outputs)

                curr_group_outputs = curr_group_outputs_lst
            else:
                # Run the encoder.
                # `curr_group_outputs` is either of the following:
                # 1. A tensor of shape (num_items, feature_size, hidden_size)
                # in case feature_size is fixed across all multimodal items.
                # 2. A list or tuple (length: num_items) of tensors,
                # each of shape (feature_size, hidden_size) in case the feature
                # size is dynamic depending on the input multimodal items.

                with self.timed_encoder_operation(
                    should_time, mm_lora_refs, current_item_idx, num_items
                ):
                    curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )
            encoder_outputs.extend(curr_group_outputs)

            current_item_idx += num_items

        # Cache the encoder outputs by mm_hash
        for mm_hash, output in zip(mm_hashes, encoder_outputs):
            self.encoder_cache[mm_hash] = output
            logger.debug("Finish execute for mm hash %s", mm_hash)
            self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)

        return encoder_outputs

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # Swap to the other buffer to avoid race condition with previous
        # iteration's async copy that may still be reading from CPU.
        self.is_mm_embed_idx = 1 - self.is_mm_embed_idx
        is_mm_embed_buf = self.is_mm_embed_buffers[self.is_mm_embed_idx]

        mm_embeds = list[torch.Tensor]()
        is_mm_embed = is_mm_embed_buf.cpu
        is_mm_embed[:total_num_scheduled_tokens] = False

        req_start_idx = 0
        should_sync_mrope_positions = False
        should_sync_xdrope_positions = False

        for req_id in self.input_batch.req_ids:
            mm_embeds_req: list[torch.Tensor] = []

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens

            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # If there are no embeddings in the current range, we skip
                # gathering the embeddings.
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                # OR mask for overlapping mm_features (use_audio_in_video)
                if is_embed is None:
                    is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                        True
                    )
                else:
                    is_mm_embed[
                        req_start_pos + start_idx : req_start_pos + end_idx
                    ] |= is_embed
                mm_embeds_req.append(mm_embeds_item)

            if self.is_multimodal_pruning_enabled and self.uses_mrope:
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                mm_embeds_req, new_mrope_positions, new_delta = (
                    self.model.recompute_mrope_positions(
                        input_ids=req_state.prompt_token_ids,
                        multimodal_embeddings=mm_embeds_req,
                        mrope_positions=req_state.mrope_positions,
                        num_computed_tokens=req_state.num_computed_tokens,
                    )
                )
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_scheduled_tokens

        is_mm_embed = is_mm_embed_buf.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_mrope_positions:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_xdrope_positions:
            self._calc_xdrope_positions(scheduler_output)
            self.xdrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        return mm_embeds, is_mm_embed

    def get_model(self) -> nn.Module:
        if not hasattr(self, "model"):
            raise ValueError("Cannot get model before model has been initialized")
        if isinstance(self.model, (CUDAGraphWrapper, UBatchWrapper)):
            # get raw model out of the cudagraph wrapper.
            return self.model.unwrap()
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        if supports_realtime(model):
            supported_tasks.append("realtime")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        supported_tasks = list(model.pooler.get_supported_tasks())

        if "score" in supported_tasks:
            num_labels = getattr(self.model_config.hf_config, "num_labels", 0)
            if num_labels != 1:
                supported_tasks.remove("score")
                logger.debug_once("Score API is only enabled for num_labels == 1.")

        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)

        # When sequence parallelism is enabled, the "residual" tensor is sharded
        # across tensor parallel ranks, so each rank only needs its own slice.
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = k == "residual" and is_rs
                copy_len = num_tokens // tp if is_scattered else num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True
                )

        return IntermediateTensors(
            {
                k: v[: num_tokens // tp]
                if k == "residual" and is_rs
                else v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            }
        )

    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        """
        Step for the EPLB (Expert Parallelism Load Balancing) state.
        """
        if not self.parallel_config.enable_eplb or self.eep_eplb_suppressed:
            return

        assert self.eplb_state is not None
        model = self.get_model()
        assert is_mixture_of_experts(model)
        self.eplb_state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        model = self.get_model()
        assert is_mixture_of_experts(model)

        self.eplb_state = EplbState.from_mapping(
            model=model,
            model_config=self.model_config,
            device=self.device,
            parallel_config=self.parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
            num_valid_physical_experts=old_num_physical_experts,
        )

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: KVConnectorOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        hidden_states = hidden_states[:num_scheduled_tokens]
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]

        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np, seq_lens_cpu, device=hidden_states.device
        )

        model = cast(VllmModelForPooling, self.model)
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        if raw_pooler_output is None or not any(finished_mask):
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        if self.use_async_scheduling:
            return AsyncGPUPoolingModelRunnerOutput(
                model_runner_output=model_runner_output,
                raw_pooler_output=raw_pooler_output,
                finished_mask=finished_mask,
                async_output_copy_stream=self.async_output_copy_stream,
            )

        model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
            raw_pooler_output=raw_pooler_output,
            finished_mask=finished_mask,
        )
        self._sync_device()

        return model_runner_output

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if self.compilation_config.pass_config.enable_sp and tp_size > 1:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _prepare_mm_inputs(
        self, num_tokens: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.model.requires_raw_input_tokens:
            input_ids = self.input_ids.gpu[:num_tokens]
        else:
            input_ids = None

        inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
        return input_ids, inputs_embeds

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,  # Padded
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        is_first_rank = get_pp_group().is_first_rank
        is_encoder_decoder = self.model_config.is_encoder_decoder

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        ec_connector_output = None

        if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
            # Run the multimodal encoder if any.
            with self.maybe_get_ec_connector_output(
                scheduler_output,
                encoder_cache=self.encoder_cache,
            ) as ec_connector_output:
                self._execute_mm_encoder(scheduler_output)
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.embed_input_ids(
                self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)

            input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
            model_kwargs = {
                **self._init_model_kwargs(),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = (
                self.is_token_ids.gpu[:num_scheduled_tokens]
                .nonzero(as_tuple=False)
                .squeeze(1)
            )
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs()
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs()

        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Run the encoder, just like we do with other multimodal inputs.
            # For an encoder-decoder model, our processing here is a bit
            # simpler, because the outputs are just passed to the decoder.
            # We are not doing any prompt replacement. We also will only
            # ever have a single encoder input.
            encoder_outputs = self._execute_mm_encoder(scheduler_output)
            model_kwargs.update({"encoder_outputs": encoder_outputs})

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    def set_pending_hybrid_spec_bundle(
        self, bundle: HybridProposalBundle | None
    ) -> None:
        if bundle is not None:
            bundle = expand_hybrid_bundle_for_pivot_expansion(bundle)
        self.pending_hybrid_spec_bundle = bundle
        self.pending_pivot_expansion_plan = (
            bundle.expansion_plan if bundle is not None else None
        )

    def _discard_drafter_pending_hierarchical_state(self) -> None:
        """Drop staged spechive/pivot state on the drafter when present."""
        drafter = self.drafter
        if drafter is None:
            return
        discard = getattr(
            drafter, "discard_pending_hierarchical_verification_state", None
        )
        if callable(discard):
            discard()

    def _take_drafter_staged_hybrid_and_publish(
        self,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> None:
        """Move drafter-staged hybrid bundle to ``pending_hybrid_spec_bundle`` once.

        Pivot row universe must match **target verification decode rows** for this
        step (``SpecDecodeMetadata.num_draft_tokens`` / related layout), not only
        ``CommonAttentionMetadata.num_reqs``, when mixed prefill/decode or split
        decode applies.
        """
        if self.drafter is None:
            self.set_pending_hybrid_spec_bundle(None)
            return
        take_bundle = getattr(self.drafter, "take_pending_spechive_bundle", None)
        if not callable(take_bundle):
            self.set_pending_hybrid_spec_bundle(None)
            return
        bundle = take_bundle()
        self.set_pending_hybrid_spec_bundle(bundle)
        if (
            bundle is not None
            and self._dit_debug_enabled
            and self.speculative_config is not None
            and self.speculative_config.method == "pivot"
            and spec_decode_metadata is not None
        ):
            logger.info(
                "PIVOT_DEBUG publish_bundle: bundle_rows=%d metadata_rows=%d",
                len(bundle.num_draft_tokens),
                len(spec_decode_metadata.num_draft_tokens),
            )

    def _clear_pending_pivot_hybrid_at_prepare_boundary(
        self, reason: str, *, detail: str | None = None
    ) -> None:
        """Drop runner + drafter pivot pending state when it cannot match this step."""
        self.pending_hybrid_spec_bundle = None
        self.pending_pivot_expansion_plan = None
        if (
            self.speculative_config is not None
            and self.speculative_config.method == "pivot"
            and self.drafter is not None
        ):
            self._discard_drafter_pending_hierarchical_state()
        if self._is_dit_debug_enabled():
            suffix = f" ({detail})" if detail else ""
            logger.warning(
                "Clearing pending pivot hybrid bundle/plan at prepare boundary: %s%s",
                reason,
                suffix,
            )

    def take_pending_hybrid_spec_bundle(self) -> HybridProposalBundle | None:
        bundle = self.pending_hybrid_spec_bundle
        self.pending_hybrid_spec_bundle = None
        self.pending_pivot_expansion_plan = (
            bundle.expansion_plan if bundle is not None else None
        )
        return bundle

    def _is_dit_debug_enabled(self) -> bool:
        return self._dit_debug_enabled

    def _dit_debug_event(self, event_name: str, payload: dict[str, Any]) -> None:
        if not self._is_dit_debug_enabled():
            return
        event = {
            "event": event_name,
            "step_id": self._dit_debug_step_id,
            **payload,
        }
        logger.info("DIT_DEBUG %s", json.dumps(event, sort_keys=True, default=str))

    def _dit_debug_assert(
        self, cond: bool, code: str, *, detail: str = "", context: dict[str, Any] | None = None
    ) -> bool:
        if not self._is_dit_debug_enabled():
            return cond
        counts = self._dit_debug_check_counts[code]
        if cond:
            counts[0] += 1
        else:
            counts[1] += 1
        payload: dict[str, Any] = {"check_code": code, "ok": cond, "detail": detail}
        if context is not None:
            payload.update(context)
        self._dit_debug_event("check", payload)
        return cond

    def _dit_debug_emit_summary(self, *, reason: str) -> None:
        if not (self._is_dit_debug_enabled() and self._dit_debug_summary):
            return
        summary = {
            code: {"pass": counts[0], "fail": counts[1]}
            for code, counts in self._dit_debug_check_counts.items()
        }
        self._dit_debug_event("summary", {"reason": reason, "checks": summary})
        self._dit_debug_check_counts.clear()

    @staticmethod
    def _pivot_profile_row_is_active(
        expansion_plan: PivotExpansionPlan | None, row_idx: int
    ) -> bool:
        if expansion_plan is None:
            return True
        active = expansion_plan.packed_row_is_active
        if active is None:
            return True
        return 0 <= row_idx < len(active) and bool(active[row_idx])

    def _populate_staged_profile_context_from_bundle(
        self,
        *,
        bundle: HybridProposalBundle,
        expansion_plan: PivotExpansionPlan | None,
        selected_rows: list[int] | None,
        origin_batch_size: int,
    ) -> None:
        pctx = self._current_profile_ctx
        if pctx is None or bundle.mode != "hierarchical_verification":
            return
        inter_verified_rows = bundle.inter_verified_counts
        inter_accepted_rows = bundle.inter_accepted_counts
        if inter_verified_rows is None or inter_accepted_rows is None:
            return

        inter_verified_per_req = [0] * origin_batch_size
        inter_accepted_per_req = [0] * origin_batch_size
        partial_accepted_per_req = [0] * origin_batch_size

        if expansion_plan is not None:
            sm = expansion_plan.packed_sm_origin
            if sm is None or len(sm) != len(bundle.num_draft_tokens):
                sm = expansion_plan.expanded_to_origin
            for row_idx, origin_row in enumerate(sm):
                if not self._pivot_profile_row_is_active(expansion_plan, row_idx):
                    continue
                if not (0 <= origin_row < origin_batch_size):
                    continue
                if row_idx < len(inter_verified_rows):
                    inter_verified_per_req[origin_row] += int(
                        inter_verified_rows[row_idx]
                    )
                if row_idx < len(inter_accepted_rows):
                    inter_accepted_per_req[origin_row] += int(
                        inter_accepted_rows[row_idx]
                    )
            if selected_rows is not None:
                for origin_row, row_idx in enumerate(selected_rows):
                    if origin_row >= origin_batch_size:
                        break
                    if 0 <= row_idx < len(inter_accepted_rows):
                        partial_accepted_per_req[origin_row] = int(
                            inter_accepted_rows[row_idx]
                        )
        else:
            limit = min(
                origin_batch_size,
                len(inter_verified_rows),
                len(inter_accepted_rows),
            )
            for row_idx in range(limit):
                inter_verified_per_req[row_idx] = int(inter_verified_rows[row_idx])
                inter_accepted_per_req[row_idx] = int(inter_accepted_rows[row_idx])
                partial_accepted_per_req[row_idx] = int(inter_accepted_rows[row_idx])

        pctx.inter_verified_per_req = inter_verified_per_req
        pctx.inter_accepted_per_req = inter_accepted_per_req
        pctx.partial_accepted_per_req = partial_accepted_per_req
        pctx.staged_verification_depth = int(bundle.staged_verification_depth)
        pctx.staged_verification_used = True

    def _emit_pivot_collapse_family_metadata(
        self,
        *,
        expansion_plan: PivotExpansionPlan | None,
        selected_rows: list[int] | None,
        accepted_lens: list[int] | None,
        req_ids: list[str],
    ) -> None:
        if expansion_plan is None or not expansion_plan.families:
            return
        if selected_rows is None or accepted_lens is None:
            return

        from vllm.v1.spec_decode.profiler_types import (  # noqa: E402
            SpecDecodeFamilyMetadataRecord,
        )

        step_id = getattr(self._current_profile_ctx, "step_id", 0)
        origin_batch_size = (
            int(expansion_plan.origin_batch_size)
            if expansion_plan.origin_batch_size > 0
            else len(selected_rows)
        )
        expansion_pct = (
            ((int(expansion_plan.expanded_batch_size) - origin_batch_size)
             / origin_batch_size * 100.0)
            if origin_batch_size > 0
            else 0.0
        )

        for fam_idx, fam in enumerate(expansion_plan.families):
            origin_row = int(fam.origin_row)
            if not (0 <= origin_row < len(selected_rows) and origin_row < len(req_ids)):
                continue
            winner_row_idx = int(selected_rows[origin_row])
            if winner_row_idx not in fam.expanded_rows:
                continue

            active_rows = [
                int(row_idx)
                for row_idx in fam.expanded_rows
                if self._pivot_profile_row_is_active(expansion_plan, int(row_idx))
            ]
            if not active_rows:
                continue

            winner_local_idx = active_rows.index(winner_row_idx)
            winner_accept_len = (
                int(accepted_lens[winner_row_idx])
                if 0 <= winner_row_idx < len(accepted_lens)
                else 0
            )
            selected_req_ids = [req_ids[origin_row]] * len(active_rows)
            self._spec_profiler.emit_family_metadata(
                SpecDecodeFamilyMetadataRecord(
                    step_id=step_id,
                    family_id=f"{req_ids[origin_row]}:f{fam_idx}:collapse",
                    origin_req_id=req_ids[origin_row],
                    family_stage="collapse",
                    family_width_before=len(active_rows),
                    family_width_after=1,
                    pruned_variant_count=max(0, len(active_rows) - 1),
                    winner_family_row_idx=winner_local_idx,
                    winner_accept_len=winner_accept_len,
                    topk_k=len(getattr(fam, "candidate_ranks", [])),
                    expansion_pct=expansion_pct,
                    collapse_reason=(
                        "winner_selected" if winner_accept_len > 0 else "all_rejected"
                    ),
                    selected_expansion_req_ids=selected_req_ids,
                )
            )

    # DIT shadow replay helpers were removed. DIT now runs with local round
    # orchestration and token buffers inside run_hierarchical_verification_rounds().

    def _prepare_hv_step(
        self,
        inter: IntermediateDraftModelProposer,
        inter_cad: CommonAttentionMetadata,
        *,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        candidate_tokens: torch.Tensor,
    ) -> HvVerifyPrefabrication:
        for ag in inter.draft_attn_groups:
            m_builder = ag.get_metadata_builder()
            if isinstance(
                m_builder,
                (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder),
            ):
                raise NotImplementedError(
                    "standalone hierarchical_verification (v1) does not support "
                    "Mamba2/GDN attention metadata builders on the intermediate verifier."
                )
        cad = clone_common_attn_metadata(inter_cad)
        (
            cad,
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
        ) = _canonicalize_reused_prefix_frontier(
            inter,
            cad=cad,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
        )
        block_size = inter.draft_attn_groups[0].kv_cache_spec.block_size
        roll_rows = [
            [int(tok) for tok in candidate_tokens[b].tolist()]
            for b in range(candidate_tokens.shape[0])
        ]
        (
            pref_toks,
            pref_pos,
            pref_hidden,
            pref_next,
            pref_cad,
            base_query_lens,
            prefix_lens,
            roll_lens,
        ) = hv_build_prefix_conditioned_inputs(
            cad=cad,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            prefix_rows=prefix_rows,
            roll_rows=roll_rows,
            block_size=block_size,
        )
        return HvVerifyPrefabrication(
            pref_toks=pref_toks,
            pref_pos=pref_pos,
            pref_hidden=pref_hidden,
            pref_next=pref_next,
            pref_cad=pref_cad,
            base_query_lens=base_query_lens,
            prefix_lens=prefix_lens,
            roll_lens=roll_lens,
            candidate_tokens=candidate_tokens.to(torch.int32),
        )

    def _run_hv_verify_step(
        self,
        inter: IntermediateDraftModelProposer,
        pre: HvVerifyPrefabrication,
        *,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> DitRoundVerification:
        num_tokens, token_indices_to_sample, pref_cad = inter.set_inputs_first_pass(
            target_token_ids=pre.pref_toks,
            next_token_ids=pre.pref_next,
            target_positions=pre.pref_pos,
            target_hidden_states=pre.pref_hidden,
            token_indices_to_sample=None,
            cad=pre.pref_cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        assert token_indices_to_sample is not None

        per_layer_attn_metadata: dict[str, object] = {}
        attn_metadata = None
        for attn_group in inter.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=pref_cad,
                draft_index=0,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        inter._check_per_layer_attn_metadata_contract(per_layer_attn_metadata)

        if inter.allowed_attn_types is not None and not isinstance(
            attn_metadata,
            inter.allowed_attn_types,
        ):
            raise ValueError(
                "hierarchical_verification: unsupported attention metadata type "
                f"{type(attn_metadata)}; allowed: {inter.allowed_attn_types}"
            )

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            inter._determine_batch_execution_and_padding(num_tokens)
        )
        if inter.supports_mm_inputs:
            inter.inputs_embeds[:num_tokens] = inter.model.embed_input_ids(
                inter.input_ids[:num_tokens],
            )
            input_ids = None
            inputs_embeds = inter.inputs_embeds[:num_input_tokens]
        else:
            input_ids = inter.input_ids[:num_input_tokens]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": inter._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        if inter.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = inter.hidden_states[:num_input_tokens]

        with set_forward_context(
            per_layer_attn_metadata,
            inter.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=inter._get_slot_mapping(
                num_input_tokens, pref_cad.slot_mapping
            ),
        ):
            ret_hidden_states = inter.model(**model_kwargs)
            if inter.model_returns_tuple():
                last_hidden_states, _ = ret_hidden_states
            else:
                last_hidden_states = ret_hidden_states
        all_logits = inter.model.compute_logits(last_hidden_states).to(torch.float32)
        logits_flat, bonus_logits = slice_hv_verification_logits(
            all_logits,
            pref_cad,
            pre.candidate_tokens,
            pre.base_query_lens,
            pre.prefix_lens,
            pre.roll_lens,
        )
        return DitRoundVerification(
            logits_flat=logits_flat, bonus_logits=bonus_logits
        )

    def _hv_sync_intermediate_batch_spec_tokens(
        self, proposal_tokens: torch.Tensor
    ) -> None:
        """Write per-row draft proposal tokens into intermediate batch spec slots."""
        ib = self.intermediate_input_batch
        assert ib is not None
        assert self._hv_scheduler_output is not None
        batch_size = int(proposal_tokens.shape[0])
        for b in range(batch_size):
            rid = str(self.input_batch.req_ids[b])
            mir = self.intermediate_requests.get(rid)
            if mir is None:
                continue
            toks = [int(x) for x in proposal_tokens[b].tolist()]
            ib.update_req_spec_token_ids(mir, {rid: toks})
        ib.refresh_metadata()

    def _run_hv_draft_step_from_frontier(
        self,
        draft: PinnedDraftNamespaceDraftModelProposer,
        *,
        base_next_token_ids: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        chunk_len: int,
        sampling_metadata: SamplingMetadata,
        use_draft_probs: bool,
    ) -> DitRoundProposal:
        """Draft HV chunk using draft frontier ``_prepare_inputs`` (no prefix packing)."""
        if not self._draft_kv_frontier_enabled():
            raise RuntimeError(
                "_run_hv_draft_step_from_frontier requires draft frontier mode."
            )
        if draft.supports_mm_inputs:
            raise RuntimeError(
                "hierarchical_verification draft frontier path does not support "
                "multimodal draft inputs."
            )
        if getattr(draft, "method", None) == "eagle3":
            raise RuntimeError(
                "hierarchical_verification draft frontier path does not support Eagle3."
            )
        so = self._hv_scheduler_output
        if so is None:
            raise RuntimeError(
                "hierarchical_verification: missing scheduler_output snapshot "
                "(_hv_scheduler_output); propose_draft_token_ids must wrap propose()."
            )
        nsched = [
            int(so.num_scheduled_tokens[str(rid)])
            for rid in self.input_batch.req_ids[: self.input_batch.num_reqs]
        ]
        nst = np.array(nsched, dtype=np.int32)
        prep = self._prepare_draft_metadata(so, nst)
        if prep is None:
            raise RuntimeError(
                "hierarchical_verification: draft frontier metadata preparation "
                "returned None (batch mismatch, async prev-sample, mamba_cache_mode "
                "align, or unsupported layout)."
            )
        gid = int(getattr(draft, "kv_cache_gid", -1))
        cad_map = prep.spec_decode_common_attn_metadata_by_gid or {}
        frontier_cad = cad_map.get(gid) or prep.spec_decode_common_attn_metadata
        if frontier_cad is None:
            raise RuntimeError(
                "hierarchical_verification: missing speculative CommonAttentionMetadata "
                f"for draft kv_cache_gid={gid}"
            )
        total_tok = int(so.total_num_scheduled_tokens)
        batch_size = int(frontier_cad.batch_size())
        if len(prefix_rows) != batch_size:
            raise RuntimeError(
                "hierarchical_verification: prefix_rows length mismatch "
                f"(prefix_rows={len(prefix_rows)}, batch_size={batch_size})"
            )
        next_list = [
            int(prefix_rows[b][-1]) if prefix_rows[b] else int(base_next_token_ids[b].item())
            for b in range(batch_size)
        ]
        next_tok_tensor = torch.tensor(
            next_list, dtype=torch.int32, device=base_next_token_ids.device
        )
        target_token_ids = self.draft_step_input_ids.gpu[:total_tok].contiguous()
        target_positions = self.draft_step_positions.gpu[:total_tok].contiguous()
        target_hidden_states = torch.zeros(
            (total_tok, int(draft.hidden_size)),
            dtype=draft.hidden_states.dtype,
            device=draft.hidden_states.device,
        )
        cad = clone_common_attn_metadata(frontier_cad)
        (
            cad,
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_tok_tensor,
        ) = _canonicalize_reused_prefix_frontier(
            draft,
            cad=cad,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_tok_tensor,
        )
        rows = draft.propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_tok_tensor,
            token_indices_to_sample=None,
            common_attn_metadata=cad,
            sampling_metadata=sampling_metadata,
            mm_embed_inputs=None,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            slot_mappings=None,
        ).to(torch.int32)
        rows = rows[:, :chunk_len]
        out_probs = None
        if use_draft_probs:
            probs_flat = getattr(draft, "last_draft_probs_flat", None)
            if probs_flat is not None and probs_flat.numel() > 0:
                bsz = int(rows.shape[0])
                out_probs = probs_flat.view(bsz, -1, probs_flat.shape[-1])[
                    :, : int(rows.shape[1])
                ].to(torch.float32)
        return DitRoundProposal(tokens=rows, probs=out_probs)

    def _run_hv_draft_step(
        self,
        draft: PinnedDraftNamespaceDraftModelProposer,
        draft_cad: CommonAttentionMetadata,
        *,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        chunk_len: int,
        sampling_metadata: SamplingMetadata,
        use_draft_probs: bool,
    ) -> DitRoundProposal:
        """One draft HV chunk proposal using runner-built speculative CAD.

        Standalone HV with draft frontier uses metadata-direct
        ``_run_hv_draft_step_from_frontier`` (no ``hv_step_packing``). Adaptive/pivot
        paths still use ``hv_step_packing.build_prefix_conditioned_inputs`` (runner-local
        seam), not ``adaptive_cascade._build_prefix_conditioned_inputs``. Standalone HV
        **verify** does not use ``hv_step_packing`` on the hot path.
        """
        if self._draft_kv_frontier_enabled():
            return self._run_hv_draft_step_from_frontier(
                draft,
                base_next_token_ids=next_token_ids,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                prefix_rows=prefix_rows,
                chunk_len=chunk_len,
                sampling_metadata=sampling_metadata,
                use_draft_probs=use_draft_probs,
            )
        cad = clone_common_attn_metadata(draft_cad)
        (
            cad,
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
        ) = _canonicalize_reused_prefix_frontier(
            draft,
            cad=cad,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
        )
        block_size = int(draft.draft_attn_groups[0].kv_cache_spec.block_size)
        (
            pref_toks,
            pref_pos,
            pref_hidden,
            pref_next,
            pref_cad,
            _,
            _,
            _,
        ) = hv_build_prefix_conditioned_inputs(
            cad=cad,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            prefix_rows=prefix_rows,
            roll_rows=None,
            block_size=block_size,
        )
        rows = draft.propose(
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
            probs_flat = getattr(draft, "last_draft_probs_flat", None)
            if probs_flat is not None and probs_flat.numel() > 0:
                bsz = int(rows.shape[0])
                out_probs = probs_flat.view(bsz, -1, probs_flat.shape[-1])[
                    :, : int(rows.shape[1])
                ].to(torch.float32)
        return DitRoundProposal(tokens=rows, probs=out_probs)

    def _run_hv_intermediate_verify_from_frontier(
        self,
        inter: IntermediateDraftModelProposer,
        *,
        proposal_tokens: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> DitRoundVerification:
        """Intermediate verify using intermediate frontier metadata + one direct forward."""
        if not self._intermediate_kv_frontier_enabled():
            raise RuntimeError(
                "_run_hv_intermediate_verify_from_frontier requires "
                "intermediate frontier mode (intermediate_input_batch + scheduler sync)."
            )
        so = self._hv_scheduler_output
        if so is None:
            raise RuntimeError(
                "hierarchical_verification: missing scheduler_output snapshot "
                "(_hv_scheduler_output); propose_draft_token_ids must wrap propose()."
            )
        self._hv_sync_intermediate_batch_spec_tokens(proposal_tokens)
        nsched = [
            int(so.num_scheduled_tokens[str(rid)])
            for rid in self.input_batch.req_ids[: self.input_batch.num_reqs]
        ]
        nst = np.array(nsched, dtype=np.int32)
        prep = self._prepare_intermediate_metadata(so, nst)
        if prep is None or prep.spec_decode_metadata is None:
            raise RuntimeError(
                "hierarchical_verification: intermediate frontier metadata preparation "
                "returned None (batch mismatch, async prev-sample, mamba_cache_mode "
                "align, or unsupported layout)."
            )
        gid = int(getattr(inter, "kv_cache_gid", -1))
        cad_map = prep.spec_decode_common_attn_metadata_by_gid or {}
        inter_cad = cad_map.get(gid) or prep.spec_decode_common_attn_metadata
        if inter_cad is None:
            raise RuntimeError(
                "hierarchical_verification: missing speculative CommonAttentionMetadata "
                f"for intermediate kv_cache_gid={gid}"
            )
        total_tok = int(so.total_num_scheduled_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            inter._determine_batch_execution_and_padding(total_tok)
        )
        inter.input_ids[:total_tok].copy_(
            self.intermediate_step_input_ids.gpu[:total_tok], non_blocking=True
        )
        if num_input_tokens > total_tok:
            pad_id = int(inter.input_ids[total_tok - 1].item())
            inter.input_ids[total_tok:num_input_tokens].fill_(pad_id)
        inter._set_positions(
            total_tok, self.intermediate_step_positions.gpu[:total_tok]
        )
        if num_input_tokens > total_tok:
            last_pos = inter.positions[total_tok - 1]
            inc = torch.arange(
                1,
                num_input_tokens - total_tok + 1,
                device=last_pos.device,
                dtype=last_pos.dtype,
            )
            inter.positions[total_tok:num_input_tokens] = last_pos + inc

        per_layer_attn_metadata: dict[str, object] = {}
        attn_metadata = None
        for attn_group in inter.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=inter_cad,
                draft_index=0,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        inter._check_per_layer_attn_metadata_contract(per_layer_attn_metadata)

        if inter.allowed_attn_types is not None and not isinstance(
            attn_metadata,
            inter.allowed_attn_types,
        ):
            raise ValueError(
                "hierarchical_verification: unsupported attention metadata type "
                f"{type(attn_metadata)}; allowed: {inter.allowed_attn_types}"
            )

        if inter.supports_mm_inputs:
            inter.inputs_embeds[:num_input_tokens] = inter.model.embed_input_ids(
                inter.input_ids[:num_input_tokens],
            )
            input_ids = None
            inputs_embeds = inter.inputs_embeds[:num_input_tokens]
        else:
            input_ids = inter.input_ids[:num_input_tokens]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": inter._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        if inter.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = inter.hidden_states[:num_input_tokens]

        with set_forward_context(
            per_layer_attn_metadata,
            inter.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=inter._get_slot_mapping(
                num_input_tokens, inter_cad.slot_mapping
            ),
        ):
            ret_hidden_states = inter.model(**model_kwargs)
            if inter.model_returns_tuple():
                last_hidden_states, _ = ret_hidden_states
            else:
                last_hidden_states = ret_hidden_states
        all_logits = inter.model.compute_logits(last_hidden_states).to(torch.float32)
        logits_flat, bonus_logits = (
            gather_hv_verification_logits_from_spec_decode_metadata(
                all_logits, prep.spec_decode_metadata
            )
        )
        return DitRoundVerification(
            logits_flat=logits_flat, bonus_logits=bonus_logits
        )

    def run_hv_rounds(
        self,
        *,
        drafter: HierarchicalVerificationProposer,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        spec_decode_cad_by_gid: dict[int, CommonAttentionMetadata] | None = None,
    ) -> tuple[torch.Tensor, HybridProposalBundle]:
        """Standalone hierarchical verification: fixed full-batch rounds."""
        assert self.speculative_config is not None
        if not self._intermediate_kv_frontier_enabled():
            raise ValueError(
                "standalone hierarchical_verification requires intermediate frontier "
                "mode: configure intermediate_model, set intermediate_kv_mode to "
                "'mirror_frontier' (intermediate frontier batch), and run on the last "
                "pipeline-parallel rank."
            )
        self._dit_debug_step_id += 1
        spec = self.speculative_config
        cap = int(spec.hv_max_spec_len())
        L = int(spec.num_speculative_tokens)
        n_inner = int(spec.num_hv_rounds)
        batch_size = common_attn_metadata.batch_size()
        use_draft_probs = spec.use_draft_probs_in_rejection and not sampling_metadata.all_greedy
        vocab_size = self.model_config.get_vocab_size()

        cad_map = spec_decode_cad_by_gid or {}
        draft_gid = getattr(drafter.draft, "kv_cache_gid", None)
        draft_cad = (
            cad_map[int(draft_gid)]
            if draft_gid is not None and int(draft_gid) in cad_map
            else common_attn_metadata
        )

        prefix_rows: list[list[int]] = [[] for _ in range(batch_size)]
        prefix_prob_rows: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        source_stage_rows: list[list[int]] = [[] for _ in range(batch_size)]
        inter_verified_rows: list[int] = [0 for _ in range(batch_size)]
        inter_accepted_rows: list[int] = [0 for _ in range(batch_size)]

        for _ in range(n_inner):
            before_lens = [len(r) for r in prefix_rows]
            sm_idxs = list(range(batch_size))
            round_sm = slice_sampling_metadata_for_subbatch(
                sampling_metadata,
                sm_idxs,
                provisional_prefix_rows=prefix_rows,
                sampled_ids_only=True,
            )
            proposal = drafter.propose_chunk_from_prefix(
                base_target_token_ids=target_token_ids,
                base_target_positions=target_positions,
                base_target_hidden_states=target_hidden_states,
                base_next_token_ids=next_token_ids,
                base_common_attn_metadata=draft_cad,
                base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                prefix_rows=prefix_rows,
                chunk_len=L,
                sampling_metadata=round_sm,
                use_draft_probs=use_draft_probs,
            )
            eff_bs = int(proposal.tokens.shape[0])
            if eff_bs != batch_size:
                raise RuntimeError(
                    "hierarchical_verification expects draft proposals for every "
                    f"batch row; got effective batch {eff_bs} != {batch_size}"
                )
            verification = self._run_hv_intermediate_verify_from_frontier(
                drafter.inter,
                proposal_tokens=proposal.tokens,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
            decision = drafter.run_inter_verification_acceptance(
                proposal=proposal,
                verification=verification,
                sampling_metadata=round_sm,
                rejection_sampler=self.rejection_sampler,
                vocab_size=vocab_size,
                use_draft_probs=use_draft_probs,
            )
            self._advance_intermediate_frontier_after_round(
                decision,
                batch_size=batch_size,
                eff_bs=batch_size,
                pivot_expansion_plan=None,
                before_prefix_lens=before_lens,
            )
            self._advance_draft_frontier_after_round(
                decision,
                batch_size=batch_size,
                eff_bs=batch_size,
                pivot_expansion_plan=None,
                before_prefix_lens=before_lens,
            )
            for b in range(batch_size):
                inter_verified_rows[b] += L
                inter_accepted_rows[b] += len(decision.emitted_rows[b])
            for b, emitted in enumerate(decision.emitted_rows):
                prefix_rows[b].extend(emitted)
                source_stage_rows[b].extend([0] * len(emitted))
                if use_draft_probs:
                    prefix_prob_rows[b].extend(decision.emitted_prob_rows[b])

        tail_len = L
        remaining_cap_per_row = [
            max(0, cap - len(prefix_rows[b])) for b in range(batch_size)
        ]
        if self._is_dit_debug_enabled():
            _tail_chk = validate_hierarchical_verification_tail_len_rowwise(
                tail_len=tail_len,
                interval_tokens=L,
                remaining_cap_per_row=remaining_cap_per_row,
            )
            self._dit_debug_assert(
                _tail_chk.ok,
                _tail_chk.code,
                detail=_tail_chk.detail,
            )
        tail_sm = slice_sampling_metadata_for_subbatch(
            sampling_metadata,
            list(range(batch_size)),
            provisional_prefix_rows=prefix_rows,
            sampled_ids_only=True,
        )
        tail = drafter.propose_chunk_from_prefix(
            base_target_token_ids=target_token_ids,
            base_target_positions=target_positions,
            base_target_hidden_states=target_hidden_states,
            base_next_token_ids=next_token_ids,
            base_common_attn_metadata=draft_cad,
            base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            chunk_len=tail_len,
            sampling_metadata=tail_sm,
            use_draft_probs=use_draft_probs,
        )
        if int(tail.tokens.shape[0]) != batch_size:
            raise RuntimeError(
                "hierarchical_verification tail expects full batch "
                f"{batch_size}, got {int(tail.tokens.shape[0])}"
            )
        tail_rows = [
            [int(tok) for tok in tail.tokens[b].tolist()] for b in range(batch_size)
        ]
        tail_prob_rows: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        if use_draft_probs and tail.probs is not None:
            for b in range(batch_size):
                tail_prob_rows[b] = [tail.probs[b, j] for j in range(tail_len)]
        for b in range(batch_size):
            prefix_rows[b].extend(tail_rows[b])
            source_stage_rows[b].extend([1] * len(tail_rows[b]))
            if use_draft_probs:
                prefix_prob_rows[b].extend(tail_prob_rows[b])

        dev = target_token_ids.device
        out_exp = torch.full(
            (batch_size, cap),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=dev,
        )
        source_stage_2d = torch.zeros(
            (batch_size, cap), dtype=torch.int32, device=dev
        )
        for b in range(batch_size):
            valid = min(cap, len(prefix_rows[b]))
            if valid > 0:
                out_exp[b, :valid] = torch.tensor(
                    prefix_rows[b][:valid],
                    dtype=torch.int32,
                    device=dev,
                )
                source_stage_2d[b, :valid] = torch.tensor(
                    source_stage_rows[b][:valid],
                    dtype=torch.int32,
                    device=dev,
                )
        draft_probs_flat = (
            _flatten_prob_rows_for_output(prefix_prob_rows, out_exp)
            if use_draft_probs
            else None
        )
        row_req_ids = _hybrid_bundle_row_req_ids_for_batch(
            self.input_batch,
            origin_batch_size=batch_size,
            num_bundle_rows=batch_size,
            pivot_expansion_plan=None,
        )
        bundle = _build_hybrid_bundle_from_rows(
            out_exp,
            mode="hierarchical_verification",
            draft_probs=draft_probs_flat,
            source_stage_2d=source_stage_2d,
            bundle_row_req_ids=row_req_ids,
        )
        bundle = dataclass_replace(
            bundle,
            inter_verified_counts=inter_verified_rows,
            inter_accepted_counts=inter_accepted_rows,
            staged_verification_depth=int(n_inner),
        )
        out = _collapse_draft_tensor_rows_for_scheduler(
            out_exp, pivot_expansion_plan=None, batch_size=batch_size
        )
        return out, bundle

    def run_hierarchical_verification_rounds(
        self,
        *,
        drafter: AdaptiveSpechiveProposer | PivotProposer,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[torch.Tensor, HybridProposalBundle]:
        assert self.speculative_config is not None
        assert drafter._inter_dit is not None
        self._dit_debug_step_id += 1
        cap = self.speculative_config.runner_num_speculative_tokens()
        L = self.speculative_config.num_speculative_tokens
        # pivot_spechive: inner D=>I round count (expanded rows flow through all).
        if (
            self.speculative_config.method == "pivot"
            and self.speculative_config.pivot_spechive
        ):
            n_inner = self.speculative_config.pivot_spechive_num_rounds
        else:
            n_inner = self.speculative_config.adaptive_spechive_num_rounds
        batch_size = common_attn_metadata.batch_size()
        req_ids = list(self.input_batch.req_ids[:batch_size])
        use_draft_probs = (
            self.speculative_config.use_draft_probs_in_rejection
            and not sampling_metadata.all_greedy
        )
        vocab_size = self.model_config.get_vocab_size()
        self._dit_debug_event(
            "spechive_rounds_start",
            {
                "batch_size": batch_size,
                "chunk_len": L,
                "num_rounds": n_inner,
                "runner_num_spec_tokens": cap,
            },
        )
        for b, req_id in enumerate(req_ids):
            if req_id in self._dit_debug_last_commit_token:
                expected = self._dit_debug_last_commit_token[req_id]
                got = int(next_token_ids[b].item())
                self._dit_debug_assert(
                    got == expected,
                    "check6_next_draft_starts_from_committed_token",
                    detail=f"req_id={req_id}, expected_next={expected}, got_next={got}",
                )

        prefix_rows: list[list[int]] = [[] for _ in range(batch_size)]
        prefix_prob_rows: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        source_stage_rows: list[list[int]] = [[] for _ in range(batch_size)]
        inter_verified_rows: list[int] = [0 for _ in range(batch_size)]
        inter_accepted_rows: list[int] = [0 for _ in range(batch_size)]
        pivot_expansion_plan: PivotExpansionPlan | None = None
        inter_state: IntermediateRoundState | None = None

        for round_idx in range(n_inner):
            if self._is_dit_debug_enabled():
                self._dit_debug_assert(
                    all(
                        len(prefix_rows[b]) == len(source_stage_rows[b])
                        for b in range(len(prefix_rows))
                    ),
                    "check_hv_prefix_source_stage_alignment",
                    detail=f"round={round_idx}",
                )
                if use_draft_probs:
                    self._dit_debug_assert(
                        all(
                            len(prefix_prob_rows[b]) == len(prefix_rows[b])
                            for b in range(len(prefix_rows))
                        ),
                        "check_hv_prefix_prob_alignment",
                        detail=f"round={round_idx}",
                    )
            sm_idxs = (
                _pivot_plan_sm_indices(pivot_expansion_plan)
                if pivot_expansion_plan is not None
                else list(range(batch_size))
            )
            round_sm = slice_sampling_metadata_for_subbatch(
                sampling_metadata,
                sm_idxs,
                provisional_prefix_rows=prefix_rows,
                sampled_ids_only=True,
            )
            if self._is_dit_debug_enabled():
                for b in range(len(prefix_rows)):
                    origin_b = (
                        sm_idxs[b]
                        if pivot_expansion_plan is not None
                        else b
                    )
                    base_len = len(sampling_metadata.output_token_ids[origin_b])
                    got_len = len(round_sm.output_token_ids[b])
                    want_len = base_len + len(prefix_rows[b])
                    self._dit_debug_assert(
                        got_len == want_len,
                        "check_meta_sampling_prefix_slicing",
                        detail=(
                            f"round={round_idx}, req={b}, base_len={base_len}, "
                            f"prefix_len={len(prefix_rows[b])}, got_len={got_len}"
                        ),
                    )
            # -- Profiler: draft_forward per HV round --
            _hv_prof = self._spec_profiler
            _hv_prof.snapshot_memory_before("draft_forward",
                                           invocation_idx=round_idx)
            _hv_prof.start_stage("draft_forward",
                                 invocation_idx=round_idx)
            staged_fast = getattr(
                drafter, "supports_staged_eagle_fastpath", lambda: False
            )()
            if self._intermediate_kv_frontier_enabled():
                # mirror_frontier: disable staged fastpath; mirror batch drives metadata.
                # draft_like (frontier off): keep staged fastpath + per-round refresh.
                staged_fast = False
            if staged_fast:
                if inter_state is None:
                    inter_state = drafter.bootstrap_intermediate_round_state(
                        base_target_token_ids=target_token_ids,
                        base_target_positions=target_positions,
                        base_target_hidden_states=target_hidden_states,
                        base_next_token_ids=next_token_ids,
                        base_common_attn_metadata=common_attn_metadata,
                        base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                        prefix_rows=prefix_rows,
                    )
                proposal = drafter.propose_chunk_from_intermediate_state(
                    inter_state=inter_state,
                    base_target_token_ids=target_token_ids,
                    base_target_positions=target_positions,
                    base_target_hidden_states=target_hidden_states,
                    base_next_token_ids=next_token_ids,
                    base_common_attn_metadata=common_attn_metadata,
                    base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                    prefix_rows=prefix_rows,
                    chunk_len=L,
                    sampling_metadata=round_sm,
                    use_draft_probs=use_draft_probs,
                )
            else:
                proposal = drafter.propose_chunk_from_prefix(
                    base_target_token_ids=target_token_ids,
                    base_target_positions=target_positions,
                    base_target_hidden_states=target_hidden_states,
                    base_next_token_ids=next_token_ids,
                    base_common_attn_metadata=common_attn_metadata,
                    base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                    prefix_rows=prefix_rows,
                    chunk_len=L,
                    sampling_metadata=round_sm,
                    use_draft_probs=use_draft_probs,
                )
            _hv_prof.end_stage("draft_forward",
                               invocation_idx=round_idx)
            _hv_prof.snapshot_memory_after("draft_forward",
                                          invocation_idx=round_idx)
            _hv_prof.snapshot_shape("draft_forward", {
                "origin_batch_size": batch_size,
                "effective_batch_size": int(proposal.tokens.shape[0]),
                "num_tokens_processed": int(proposal.tokens.numel()),
            }, invocation_idx=round_idx)

            root_only_check = validate_root_only_pivot_expansion(
                prefix_rows=prefix_rows,
                expansion_plan=proposal.expansion_plan,
            )
            self._dit_debug_assert(
                root_only_check.ok,
                root_only_check.code,
                detail=f"round={round_idx}, {root_only_check.detail}",
            )
            if proposal.expansion_plan is not None and pivot_expansion_plan is None:
                pivot_expansion_plan = proposal.expansion_plan
            eff_bs = int(proposal.tokens.shape[0])
            if eff_bs != len(prefix_rows):
                assert pivot_expansion_plan is not None, (
                    "prefix row count must match proposal without a pivot plan"
                )
                # -- Profiler: expand_collapse for pivot expansion --
                _hv_prof.snapshot_memory_before("expand_collapse",
                                               invocation_idx=round_idx)
                _hv_prof.start_stage("expand_collapse",
                                     invocation_idx=round_idx)
                _hv_prof.start_stage("expand_prefix_rows",
                                     invocation_idx=round_idx)
                prefix_rows = _expand_list_rows_by_pivot_plan(
                    prefix_rows, pivot_expansion_plan
                )
                prefix_prob_rows = _expand_list_rows_by_pivot_plan(
                    prefix_prob_rows, pivot_expansion_plan
                )
                source_stage_rows = _expand_list_rows_by_pivot_plan(
                    source_stage_rows, pivot_expansion_plan
                )
                inter_verified_rows = [
                    inter_verified_rows[o]
                    for o in _pivot_plan_sm_indices(pivot_expansion_plan)
                ]
                inter_accepted_rows = [
                    inter_accepted_rows[o]
                    for o in _pivot_plan_sm_indices(pivot_expansion_plan)
                ]
                _hv_prof.end_stage("expand_prefix_rows",
                                   invocation_idx=round_idx)
                (
                    target_token_ids,
                    target_positions,
                    target_hidden_states,
                    next_token_ids,
                    common_attn_metadata,
                    num_rejected_tokens_gpu,
                ) = drafter._expand_tail_proposer_frontier_for_plan(
                    plan=pivot_expansion_plan,
                    base_target_token_ids=target_token_ids,
                    base_target_positions=target_positions,
                    base_target_hidden_states=target_hidden_states,
                    base_next_token_ids=next_token_ids,
                    base_common_attn_metadata=common_attn_metadata,
                    base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                )
                _hv_prof.end_stage("expand_collapse",
                                   invocation_idx=round_idx)
                _hv_prof.snapshot_memory_after("expand_collapse",
                                              invocation_idx=round_idx)
                _exp_pct = (
                    ((eff_bs - batch_size) / batch_size * 100.0)
                    if batch_size > 0 else 0.0
                )
                _n_families = len(getattr(
                    pivot_expansion_plan, "families", []))
                _hv_prof.snapshot_shape("expand_collapse", {
                    "origin_batch_size": batch_size,
                    "effective_batch_size": eff_bs,
                    "num_expanded_rows": eff_bs - batch_size,
                    "uses_topk_expansion": True,
                    "expansion_pct": _exp_pct,
                    "num_families": _n_families,
                }, invocation_idx=round_idx)
                # Emit family metadata for each expansion family
                from vllm.v1.spec_decode.profiler_types import (  # noqa: E402
                    SpecDecodeFamilyMetadataRecord as _FamRec,
                )
                for _fi, _fam in enumerate(
                        getattr(pivot_expansion_plan, "families", [])):
                    _origin_b = getattr(_fam, "origin_row", -1)
                    _origin_rid = (
                        self.input_batch.req_ids[_origin_b]
                        if 0 <= _origin_b < len(self.input_batch.req_ids)
                        else ""
                    )
                    _hv_prof.emit_family_metadata(_FamRec(
                        step_id=getattr(
                            self._current_profile_ctx, "step_id", 0),
                        family_id=f"{_origin_rid}:f{_fi}",
                        origin_req_id=_origin_rid,
                        family_stage="expand",
                        family_width_before=1,
                        family_width_after=len(getattr(
                            _fam, "expanded_rows", [])),
                        topk_k=len(getattr(_fam, "candidate_ranks", [])),
                        expansion_pct=_exp_pct,
                    ))
            self._dit_debug_assert(
                eff_bs == len(prefix_rows),
                "check_pivot_expanded_prefix_alignment",
                detail=f"round={round_idx}, eff_bs={eff_bs}, prefix_lists={len(prefix_rows)}",
            )
            self._dit_debug_assert(
                int(proposal.tokens.shape[1]) == int(L),
                "check1_chunk_proposal_width",
                detail=(
                    f"round={round_idx}, expected_chunk={L}, "
                    f"proposal_shape={tuple(proposal.tokens.shape)}"
                ),
            )
            if inter_state is not None and pivot_expansion_plan is not None:
                inter_state = expand_intermediate_state_for_pivot_plan(
                    inter_state, pivot_expansion_plan
                )
            mirror_kv_common_attn_metadata = None
            if (
                self._intermediate_kv_frontier_enabled()
                and self._hv_scheduler_output is not None
                and self.intermediate_input_batch is not None
                and eff_bs == batch_size
                and int(common_attn_metadata.batch_size())
                == int(self.input_batch.num_reqs)
            ):
                so = self._hv_scheduler_output
                nsched = [
                    int(so.num_scheduled_tokens[str(rid)])
                    for rid in self.input_batch.req_ids[: self.input_batch.num_reqs]
                ]
                nst = np.array(nsched, dtype=np.int32)
                prep = self._prepare_intermediate_metadata(so, nst)
                if prep is not None and prep.spec_decode_common_attn_metadata is not None:
                    mirror_kv_common_attn_metadata = (
                        prep.spec_decode_common_attn_metadata
                    )
            # ``mirror_kv_common_attn_metadata`` supplies mirror CAD for adaptive/pivot
            # prefix-conditioned verify. Standalone ``hierarchical_verification`` uses
            # ``run_hv_rounds`` (metadata-direct path) instead of this block.
            # -- Profiler: intermediate_verify (inter-verifier forward) --
            _hv_prof.snapshot_memory_before("intermediate_verify",
                                           invocation_idx=round_idx)
            _hv_prof.start_stage("intermediate_verify",
                                 invocation_idx=round_idx)
            if staged_fast and inter_state is not None:
                verification = drafter.verify_chunk_with_intermediate_state(
                    inter_state=inter_state,
                    base_target_token_ids=target_token_ids,
                    base_target_positions=target_positions,
                    base_target_hidden_states=target_hidden_states,
                    base_next_token_ids=next_token_ids,
                    base_common_attn_metadata=common_attn_metadata,
                    base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                    prefix_rows=prefix_rows,
                    candidate_tokens=proposal.tokens,
                    mirror_kv_common_attn_metadata=mirror_kv_common_attn_metadata,
                )
            else:
                verification = drafter.verify_chunk_with_inter_verifier(
                    base_target_token_ids=target_token_ids,
                    base_target_positions=target_positions,
                    base_target_hidden_states=target_hidden_states,
                    base_next_token_ids=next_token_ids,
                    base_common_attn_metadata=common_attn_metadata,
                    base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                    prefix_rows=prefix_rows,
                    candidate_tokens=proposal.tokens,
                    mirror_kv_common_attn_metadata=mirror_kv_common_attn_metadata,
                )
            _hv_prof.end_stage("intermediate_verify",
                               invocation_idx=round_idx)
            _hv_prof.snapshot_memory_after("intermediate_verify",
                                          invocation_idx=round_idx)
            _hv_prof.snapshot_shape("intermediate_verify", {
                "origin_batch_size": batch_size,
                "effective_batch_size": eff_bs,
                "num_tokens_processed": int(verification.logits_flat.shape[0]),
                "sum_seq_lens": sum(len(r) for r in prefix_rows),
            }, invocation_idx=round_idx)
            # KV proxies for intermediate_verify
            _iv_kv_lens = common_attn_metadata.seq_lens.tolist()
            _iv_q_lens_cpu = common_attn_metadata.query_start_loc_cpu
            _iv_q_lens = [
                int(_iv_q_lens_cpu[i + 1] - _iv_q_lens_cpu[i])
                for i in range(min(eff_bs, len(_iv_kv_lens)))
            ]
            _hv_prof.record_kv_proxies(
                "intermediate_verify",
                q_lens=_iv_q_lens,
                kv_lens=[int(k) for k in _iv_kv_lens[:eff_bs]],
                block_table=common_attn_metadata.block_table_tensor,
                invocation_idx=round_idx,
            )

            self._dit_debug_assert(
                int(verification.logits_flat.shape[0]) == int(eff_bs * L),
                "check1_chunk_verification_batchxL",
                detail=(
                    f"round={round_idx}, expected_rows={eff_bs * L}, "
                    f"got={verification.logits_flat.shape[0]}"
                ),
            )
            # -- Profiler: reject_sample (acceptance decision) --
            _hv_prof.start_stage("reject_sample",
                                 invocation_idx=round_idx)
            decision = drafter.run_inter_verification_acceptance(
                proposal=proposal,
                verification=verification,
                sampling_metadata=round_sm,
                rejection_sampler=self.rejection_sampler,
                vocab_size=vocab_size,
                use_draft_probs=use_draft_probs,
            )
            _hv_prof.end_stage("reject_sample",
                               invocation_idx=round_idx)
            for b in range(eff_bs):
                if not self._pivot_profile_row_is_active(pivot_expansion_plan, b):
                    continue
                inter_verified_rows[b] += int(L)
                inter_accepted_rows[b] += len(decision.emitted_rows[b])
            before_lens = [len(r) for r in prefix_rows]
            for b, emitted in enumerate(decision.emitted_rows):
                prefix_rows[b].extend(emitted)
                source_stage_rows[b].extend([0] * len(emitted))
                if use_draft_probs:
                    prefix_prob_rows[b].extend(decision.emitted_prob_rows[b])
            after_lens = [len(r) for r in prefix_rows]
            self._dit_debug_assert(
                all(
                    after_lens[b] - before_lens[b] == len(decision.emitted_rows[b])
                    for b in range(eff_bs)
                ),
                "check_hv_round_prefix_growth",
                detail=f"round={round_idx}, before={before_lens}, after={after_lens}",
            )
            self._dit_debug_assert(
                all(
                    len(source_stage_rows[b]) == len(prefix_rows[b]) for b in range(eff_bs)
                ),
                "check_hv_probs_and_stage_alignment",
                detail=f"round={round_idx}, source_stage_lens={[len(r) for r in source_stage_rows]}",
            )
            if use_draft_probs:
                self._dit_debug_assert(
                    all(
                        len(prefix_prob_rows[b]) == len(prefix_rows[b])
                        and len(decision.emitted_prob_rows[b]) == len(decision.emitted_rows[b])
                        for b in range(eff_bs)
                    ),
                    "check_hv_probs_and_stage_alignment",
                    detail=f"round={round_idx}, prefix_prob_lens={[len(r) for r in prefix_prob_rows]}",
                )

            if staged_fast and not self._intermediate_kv_frontier_enabled():
                inter_state = drafter.refresh_intermediate_round_state(
                    base_target_token_ids=target_token_ids,
                    base_target_positions=target_positions,
                    base_target_hidden_states=target_hidden_states,
                    base_next_token_ids=next_token_ids,
                    base_common_attn_metadata=common_attn_metadata,
                    base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                    prefix_rows=prefix_rows,
                    old_state=inter_state,
                )
            elif self._intermediate_kv_frontier_enabled():
                self._advance_intermediate_frontier_after_round(
                    decision,
                    batch_size=batch_size,
                    eff_bs=eff_bs,
                    pivot_expansion_plan=pivot_expansion_plan,
                    before_prefix_lens=before_lens,
                )

        expected_rounds = (
            self.speculative_config.pivot_spechive_num_rounds
            if (
                self.speculative_config.method == "pivot"
                and self.speculative_config.pivot_spechive
            )
            else self.speculative_config.adaptive_spechive_num_rounds
        )
        self._dit_debug_assert(
            n_inner == int(expected_rounds),
            "check2_round_count_matches_config_rounds",
            detail=f"executed_rounds={n_inner}, expected_rounds={expected_rounds}",
        )

        remaining_cap_per_row = [
            max(0, cap - len(prefix_rows[b])) for b in range(len(prefix_rows))
        ]
        remaining_cap = max(remaining_cap_per_row, default=0)
        tail_len = min(L, remaining_cap)
        if self._is_dit_debug_enabled():
            _tail_chk = validate_hierarchical_verification_tail_len_rowwise(
                tail_len=tail_len,
                interval_tokens=L,
                remaining_cap_per_row=remaining_cap_per_row,
            )
            self._dit_debug_assert(
                _tail_chk.ok,
                _tail_chk.code,
                detail=_tail_chk.detail,
            )
        staged_fast_outer = getattr(
            drafter, "supports_staged_eagle_fastpath", lambda: False
        )()
        if self._intermediate_kv_frontier_enabled():
            staged_fast_outer = False
        if tail_len > 0:
            tail_sm_idxs = (
                _pivot_plan_sm_indices(pivot_expansion_plan)
                if pivot_expansion_plan is not None
                else list(range(batch_size))
            )
            tail_sm = slice_sampling_metadata_for_subbatch(
                sampling_metadata,
                tail_sm_idxs,
                provisional_prefix_rows=prefix_rows,
                sampled_ids_only=True,
            )
            tail = drafter.propose_chunk_from_prefix(
                base_target_token_ids=target_token_ids,
                base_target_positions=target_positions,
                base_target_hidden_states=target_hidden_states,
                base_next_token_ids=next_token_ids,
                base_common_attn_metadata=common_attn_metadata,
                base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                prefix_rows=prefix_rows,
                chunk_len=tail_len,
                sampling_metadata=tail_sm,
                use_draft_probs=use_draft_probs,
                reuse_intermediate_state=(
                    inter_state if staged_fast_outer else None
                ),
            )
            if tail.expansion_plan is not None and pivot_expansion_plan is None:
                pivot_expansion_plan = tail.expansion_plan
            tail_eff = int(tail.tokens.shape[0])
            if tail_eff != len(prefix_rows):
                assert pivot_expansion_plan is not None, "tail batch must match prefix rows"
                prefix_rows = _expand_list_rows_by_pivot_plan(
                    prefix_rows, pivot_expansion_plan
                )
                prefix_prob_rows = _expand_list_rows_by_pivot_plan(
                    prefix_prob_rows, pivot_expansion_plan
                )
                source_stage_rows = _expand_list_rows_by_pivot_plan(
                    source_stage_rows, pivot_expansion_plan
                )
                inter_verified_rows = [
                    inter_verified_rows[o]
                    for o in _pivot_plan_sm_indices(pivot_expansion_plan)
                ]
                inter_accepted_rows = [
                    inter_accepted_rows[o]
                    for o in _pivot_plan_sm_indices(pivot_expansion_plan)
                ]
            tail_eff = int(tail.tokens.shape[0])
            tail_rows = [
                [int(tok) for tok in tail.tokens[b].tolist()] for b in range(tail_eff)
            ]
            tail_prob_rows: list[list[torch.Tensor]] = [[] for _ in range(tail_eff)]
            if use_draft_probs and tail.probs is not None:
                for b in range(tail_eff):
                    tail_prob_rows[b] = [tail.probs[b, j] for j in range(tail_len)]
            for b, emitted in enumerate(tail_rows):
                slots_left = max(0, cap - len(prefix_rows[b]))
                if slots_left <= 0:
                    continue
                take = min(len(emitted), slots_left, tail_len)
                emitted_trim = emitted[:take]
                prefix_rows[b].extend(emitted_trim)
                source_stage_rows[b].extend([1] * len(emitted_trim))
                if use_draft_probs:
                    prefix_prob_rows[b].extend(tail_prob_rows[b][:take])
            self._dit_debug_assert(
                all(
                    len(source_stage_rows[b]) == len(prefix_rows[b]) for b in range(tail_eff)
                ),
                "check_hv_probs_and_stage_alignment",
                detail=f"tail, source_stage_lens={[len(r) for r in source_stage_rows]}",
            )
            if use_draft_probs:
                self._dit_debug_assert(
                    all(len(prefix_prob_rows[b]) == len(prefix_rows[b]) for b in range(tail_eff)),
                    "check_hv_probs_and_stage_alignment",
                    detail=f"tail, prefix_prob_lens={[len(r) for r in prefix_prob_rows]}",
                )

        eff_rows = len(prefix_rows)
        dev = target_token_ids.device
        out_exp = torch.full(
            (eff_rows, cap),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=dev,
        )
        source_stage_2d = torch.zeros(
            (eff_rows, cap), dtype=torch.int32, device=dev,
        )
        for b in range(eff_rows):
            valid = min(cap, len(prefix_rows[b]))
            if valid > 0:
                out_exp[b, :valid] = torch.tensor(
                    prefix_rows[b][:valid],
                    dtype=torch.int32,
                    device=dev,
                )
                source_stage_2d[b, :valid] = torch.tensor(
                    source_stage_rows[b][:valid],
                    dtype=torch.int32,
                    device=dev,
                )
        draft_probs_flat = (
            _flatten_prob_rows_for_output(prefix_prob_rows, out_exp)
            if use_draft_probs
            else None
        )
        row_req_ids = _hybrid_bundle_row_req_ids_for_batch(
            self.input_batch,
            origin_batch_size=batch_size,
            num_bundle_rows=eff_rows,
            pivot_expansion_plan=pivot_expansion_plan,
        )
        _hv_prof = self._spec_profiler
        _hv_prof.start_stage("bundle_assemble", invocation_idx=n_inner)
        bundle = _build_hybrid_bundle_from_rows(
            out_exp,
            mode="hierarchical_verification",
            draft_probs=draft_probs_flat,
            source_stage_2d=source_stage_2d,
            bundle_row_req_ids=row_req_ids,
        )
        _hv_prof.end_stage("bundle_assemble", invocation_idx=n_inner)
        bundle = dataclass_replace(
            bundle,
            inter_verified_counts=inter_verified_rows,
            inter_accepted_counts=inter_accepted_rows,
            staged_verification_depth=int(n_inner),
        )
        if pivot_expansion_plan is not None:
            bundle = dataclass_replace(bundle, expansion_plan=pivot_expansion_plan)
        if self._is_dit_debug_enabled():
            _layout_chk = validate_hybrid_bundle_draft_layout(
                bundle,
                expected_num_rows=eff_rows,
            )
            self._dit_debug_assert(
                _layout_chk.ok,
                _layout_chk.code,
                detail=_layout_chk.detail,
            )
        inter_verified_lens = [
            sum(1 for stage in source_stage_rows[b] if stage == 0)
            for b in range(eff_rows)
        ]
        tail_draft_lens = [
            sum(1 for stage in source_stage_rows[b] if stage == 1)
            for b in range(eff_rows)
        ]
        valid_out_lens = [
            int((out_exp[b] != PLACEHOLDER_TOKEN_ID).sum().item()) for b in range(eff_rows)
        ]
        self._dit_debug_assert(
            bundle.max_spec_len == cap,
            "check4_target_verify_width_equals_num_spec_tokens",
            detail=f"bundle.max_spec_len={bundle.max_spec_len}, cap={cap}",
        )
        self._dit_debug_assert(
            all(
                inter_verified_lens[b] + tail_draft_lens[b] == valid_out_lens[b]
                for b in range(eff_rows)
            ),
            "check5_target_verify_uses_intermediate_plus_tail_bundle",
            detail=(
                f"inter={inter_verified_lens}, tail={tail_draft_lens}, "
                f"valid={valid_out_lens}"
            ),
        )
        self._dit_debug_event(
            "spechive_bundle_pre_target_verify",
            {
                "inter_verified_lens": inter_verified_lens,
                "tail_draft_lens": tail_draft_lens,
                "valid_out_lens": valid_out_lens,
                "bundle_num_draft_tokens": bundle.num_draft_tokens,
            },
        )
        self._dit_debug_assert(
            all(stage in (0, 1) for row in source_stage_rows for stage in row),
            "check_hv_probs_and_stage_alignment",
            detail="source_stage must be in {0,1}",
        )
        # -- Profiler: expand_collapse for final collapse --
        _hv_prof = self._spec_profiler
        _hv_prof.start_stage("expand_collapse",
                             invocation_idx=n_inner)
        _hv_prof.start_stage("collapse_scheduler_rows",
                             invocation_idx=n_inner)
        out = _collapse_draft_tensor_rows_for_scheduler(
            out_exp, pivot_expansion_plan, batch_size
        )
        _hv_prof.end_stage("collapse_scheduler_rows",
                           invocation_idx=n_inner)
        _hv_prof.end_stage("expand_collapse",
                           invocation_idx=n_inner)
        return out, bundle

    def run_hierarchical_tree_verification_rounds(
        self,
        *,
        drafter: AdaptiveSpechiveProposer | PivotProposer,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[torch.Tensor, HybridProposalBundle]:
        """Tree-family Pivot path reusing flat verifier contracts."""
        out, bundle = self.run_hierarchical_verification_rounds(
            drafter=drafter,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        tree_plan: PivotExpandedTreePlan | None = None
        flatten_fn = getattr(drafter, "_flatten_family_trees_for_verification", None)
        if callable(flatten_fn) and bundle.expansion_plan is not None:
            # Reconstruct expanded-row tensor from bundle, not scheduler-collapsed out.
            eff_rows = len(bundle.num_draft_tokens)
            cap = int(bundle.max_spec_len)
            out_exp = torch.full(
                (eff_rows, cap),
                PLACEHOLDER_TOKEN_ID,
                dtype=torch.int32,
                device=target_token_ids.device,
            )
            offset = 0
            for b, row_len in enumerate(bundle.num_draft_tokens):
                valid = min(cap, int(row_len))
                if valid > 0:
                    out_exp[b, :valid] = bundle.draft_token_ids[offset : offset + valid].to(
                        torch.int32
                    )
                offset += int(row_len)
            _, tree_plan = flatten_fn(
                proposal_tokens=out_exp,
                expansion_plan=bundle.expansion_plan,
            )
        if tree_plan is not None:
            bundle = dataclass_replace(bundle, tree_plan=tree_plan)
        return out, bundle

    def _spec_decode_metadata_row_req_ids(
        self, meta: SpecDecodeMetadata
    ) -> list[str]:
        """Request id per ``meta.num_draft_tokens`` row (matches prepare-time layout)."""
        n = len(meta.num_draft_tokens)
        plan = meta.expansion_plan
        if plan is not None and plan.packed_sm_origin is not None:
            sm = plan.packed_sm_origin
            if len(sm) >= n:
                return [str(self.input_batch.req_ids[int(sm[j])]) for j in range(n)]
        if plan is not None and len(plan.expanded_to_origin) >= n:
            eto = plan.expanded_to_origin
            return [str(self.input_batch.req_ids[int(eto[j])]) for j in range(n)]
        return [str(self.input_batch.req_ids[j]) for j in range(n)]

    def _validate_hybrid_spec_bundle(
        self,
        bundle: HybridProposalBundle,
        spec_decode_metadata: SpecDecodeMetadata,
    ) -> HybridProposalBundle | None:
        """Validate proposer bundle against current target verification contract."""
        pre = {
            "num_draft_tokens": list(bundle.num_draft_tokens),
            "bundle_total": int(bundle.cu_num_draft_tokens[-1].item())
            if bundle.cu_num_draft_tokens.numel() > 0
            else 0,
            "expected_total": int(spec_decode_metadata.cu_num_draft_tokens[-1].item())
            if spec_decode_metadata.cu_num_draft_tokens.numel() > 0
            else 0,
            "max_spec_len": int(bundle.max_spec_len),
            "runner_num_spec_tokens": int(self.num_spec_tokens),
            "has_draft_probs": bundle.draft_probs is not None,
            "has_source_stage": bundle.source_stage is not None,
        }
        sanitized, info = sanitize_hybrid_bundle_for_metadata(
            bundle,
            spec_decode_metadata,
            runner_num_spec_tokens=self.num_spec_tokens,
        )
        post = {
            "sanitized_is_none": sanitized is None,
            "has_draft_probs": sanitized.draft_probs is not None if sanitized is not None else False,
            "has_source_stage": sanitized.source_stage is not None if sanitized is not None else False,
        }
        self._dit_debug_event(
            "spechive_bundle_sanitize",
            {**pre, **post, "info": info},
        )
        if sanitized is not None and sanitized.tree_plan is not None:
            tree_nodes = int(sanitized.tree_plan.template.num_nodes)
            self._dit_debug_assert(
                len(sanitized.num_draft_tokens) == len(sanitized.tree_plan.families),
                "pivot_tree_bundle_family_batch_match",
            )
            self._dit_debug_assert(
                all(int(n) == tree_nodes for n in sanitized.num_draft_tokens),
                "pivot_tree_bundle_uniform_num_nodes",
                detail=f"num_nodes={tree_nodes}, lens={sanitized.num_draft_tokens}",
            )
        if sanitized is None:
            logger.warning("Dropping hybrid bundle: %s.", info)
            return None
        if info is not None:
            logger.warning("Hybrid bundle note: %s.", info)
        if bundle.draft_probs is not None and sanitized.draft_probs is None:
            logger.warning(
                "Ignoring hybrid draft_probs due to shape mismatch: %s",
                tuple(bundle.draft_probs.shape),
            )
        if bundle.source_stage is not None and sanitized.source_stage is None:
            logger.warning(
                "Ignoring hybrid source_stage due to shape mismatch: %s",
                tuple(bundle.source_stage.shape),
            )
        return sanitized

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        # Update output token ids with tokens sampled in last step
        # if async scheduling and required by current sampling params.
        self.input_batch.update_async_output_token_ids()
        if spec_decode_metadata is None:
            self.pending_hybrid_spec_bundle = None
            self.pending_pivot_expansion_plan = None
            self._discard_drafter_pending_hierarchical_state()
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        # Update spec_token_ids with real draft tokens from pre step only when
        # output_token_ids is needed (penalties or bad_words are in use).
        if self.use_async_scheduling and self._draft_token_req_ids is not None:
            draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
            self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

        draft_probs = None
        if (
            self.speculative_config is not None
            and self.speculative_config.use_draft_probs_in_rejection
            and self.drafter is not None
        ):
            draft_probs = getattr(self.drafter, "last_draft_probs_flat", None)
        bundle = self.take_pending_hybrid_spec_bundle()
        if bundle is not None:
            target_row_ids = self._spec_decode_metadata_row_req_ids(spec_decode_metadata)
            remapped, remap_info = remap_hybrid_bundle_rows_for_metadata(
                bundle,
                target_row_ids,
                spec_decode_metadata=spec_decode_metadata,
            )
            if remapped is None:
                if (
                    self._dit_debug_enabled
                    and self.speculative_config is not None
                    and self.speculative_config.method == "pivot"
                ):
                    logger.info(
                        "PIVOT_DEBUG sample: metadata_rows=%d bundle_rows=%d "
                        "bundle_req_ids=%s detail=%s",
                        len(spec_decode_metadata.num_draft_tokens),
                        len(bundle.num_draft_tokens),
                        bundle.bundle_row_req_ids,
                        remap_info,
                    )
                if self._is_dit_debug_enabled():
                    logger.warning(
                        "Dropping hybrid bundle (row alignment to spec metadata failed): %s",
                        remap_info,
                    )
                self.pending_pivot_expansion_plan = None
                self._discard_drafter_pending_hierarchical_state()
                bundle = None
            else:
                bundle = remapped
                if remap_info is not None:
                    logger.debug("Hybrid bundle row order adjusted: %s", remap_info)
        if bundle is not None:
            validated_bundle = self._validate_hybrid_spec_bundle(
                bundle, spec_decode_metadata
            )
            if validated_bundle is None:
                if (
                    self._dit_debug_enabled
                    and self.speculative_config is not None
                    and self.speculative_config.method == "pivot"
                ):
                    logger.info(
                        "PIVOT_DEBUG sample: metadata_rows=%d bundle_rows=%d "
                        "bundle_req_ids=%s detail=%s",
                        len(spec_decode_metadata.num_draft_tokens),
                        len(bundle.num_draft_tokens),
                        bundle.bundle_row_req_ids,
                        "sanitize_validate_failed",
                    )
                self.pending_pivot_expansion_plan = None
                self._dit_debug_event(
                    "spechive_bundle_cleanup_on_validation_failure",
                    {"reason": "bundle_validation_failed"},
                )
                self._discard_drafter_pending_hierarchical_state()
            bundle = validated_bundle
        if bundle is not None and bundle.draft_probs is not None:
            draft_probs = bundle.draft_probs

        verity_sm = sampling_metadata
        if spec_decode_metadata is not None:
            plan = spec_decode_metadata.expansion_plan
            if (
                plan is not None
                and plan.expanded_batch_size > 0
                and len(plan.expanded_to_origin) == plan.expanded_batch_size
                and len(spec_decode_metadata.num_draft_tokens)
                == plan.expanded_batch_size
            ):
                verity_sm = slice_sampling_metadata_for_subbatch(
                    sampling_metadata,
                    _pivot_plan_sm_indices(plan),
                    sampled_ids_only=False,
                )

        self._maybe_capture_first_draft_target_top1_for_profile(
            spec_decode_metadata,
            logits,
            verity_sm,
        )
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            draft_probs,
            logits,
            verity_sm,
        )
        expansion_plan: PivotExpansionPlan | None = None
        tree_plan: PivotExpandedTreePlan | None = None
        if bundle is not None:
            expansion_plan = bundle.expansion_plan
            tree_plan = bundle.tree_plan
        if expansion_plan is None:
            expansion_plan = self.pending_pivot_expansion_plan
        if expansion_plan is None and spec_decode_metadata is not None:
            expansion_plan = spec_decode_metadata.expansion_plan
        num_draft_for_collapse = (
            spec_decode_metadata.num_draft_tokens
            if spec_decode_metadata is not None
            else None
        )
        log_pivot_accept_len = (
            self._dit_debug_enabled
            and self.speculative_config is not None
            and self.speculative_config.method == "pivot"
            and spec_decode_metadata is not None
            and (tree_plan is not None or expansion_plan is not None)
            and num_draft_for_collapse is not None
            and len(num_draft_for_collapse)
            == int(sampler_output.sampled_token_ids.shape[0])
        )
        pre_collapse_accept_mean: float | None = None
        pre_collapse_accept_rows: int | None = None
        if log_pivot_accept_len:
            pre_collapse_accept_mean, pre_collapse_accept_rows = (
                _pivot_mean_accepted_prefix_len(
                    sampler_output.sampled_token_ids,
                    num_draft_for_collapse,
                )
            )
        selected_rows: list[int] | None = None
        pre_collapse_accept_lens: list[int] | None = None
        if expansion_plan is not None:
            if (
                num_draft_for_collapse is not None
                and len(num_draft_for_collapse)
                == int(sampler_output.sampled_token_ids.shape[0])
            ):
                pre_collapse_accept_lens = (
                    get_target_verification_accepted_draft_prefix_lens(
                        sampler_output.sampled_token_ids,
                        num_draft_for_collapse,
                        placeholder_token_id=PLACEHOLDER_TOKEN_ID,
                    )
                )
            else:
                pre_collapse_accept_lens = get_accepted_draft_lens_from_sampled_tokens(
                    sampler_output.sampled_token_ids,
                    placeholder_token_id=PLACEHOLDER_TOKEN_ID,
                )
        if tree_plan is not None:
            reduced = collapse_family_tree_sampled_to_family_paths(
                sampler_output.sampled_token_ids,
                plan=tree_plan,
                num_draft_tokens=num_draft_for_collapse,
            )
            sampler_output.sampled_token_ids, selected_rows = collapse_family_paths_to_origin(
                sampler_output.sampled_token_ids,
                plan=tree_plan,
                reduced=reduced,
            )
            if expansion_plan is not None:
                selected = set(int(r) for r in selected_rows)
                cleanup = set(
                    get_unselected_cleanup_rows(
                        expansion_plan=expansion_plan,
                        selected_rows=selected_rows,
                    )
                )
                all_rows = {
                    int(r)
                    for fam in expansion_plan.families
                    for r in fam.expanded_rows
                }
                self._dit_debug_assert(
                    selected.isdisjoint(cleanup),
                    "pivot_tree_selected_cleanup_disjoint",
                )
                self._dit_debug_assert(
                    selected | cleanup == all_rows,
                    "pivot_tree_selected_cleanup_partition",
                    detail=(
                        f"selected={sorted(selected)}, "
                        f"cleanup={sorted(cleanup)}, all={sorted(all_rows)}"
                    ),
                )
        elif expansion_plan is not None:
            self._spec_profiler.start_stage("collapse_after_target_select")
            selected_rows = select_pivot_expanded_rows_to_origin(
                sampler_output.sampled_token_ids,
                expansion_plan,
                num_draft_tokens=num_draft_for_collapse,
            )
            sampler_output.sampled_token_ids = collapse_pivot_expanded_sampled_to_origin(
                sampler_output.sampled_token_ids,
                expansion_plan,
                num_draft_tokens=num_draft_for_collapse,
            )
            self._spec_profiler.end_stage("collapse_after_target_select")
        if log_pivot_accept_len and pre_collapse_accept_mean is not None:
            post_nd = _post_collapse_num_draft_per_origin(
                expansion_plan,
                tree_plan,
                spec_decode_metadata,
            )
            post_sampled = sampler_output.sampled_token_ids
            post_b = int(post_sampled.shape[0])
            post_nd_use = (
                post_nd
                if post_nd is not None and len(post_nd) == post_b
                else None
            )
            post_mean, post_n = _pivot_mean_accepted_prefix_len(
                post_sampled,
                post_nd_use,
            )
            origin_b = -1
            if expansion_plan is not None and int(expansion_plan.origin_batch_size) > 0:
                origin_b = int(expansion_plan.origin_batch_size)
            elif tree_plan is not None:
                origin_b = int(tree_plan.origin_batch_size)
            logger.info(
                "PIVOT_DEBUG accept_len: pre_collapse_mean=%.4f pre_rows=%d "
                "post_collapse_mean=%.4f post_rows=%d origin_batch=%d "
                "expanded_batch=%d tree=%s post_nd_aligned=%s",
                pre_collapse_accept_mean,
                int(pre_collapse_accept_rows or 0),
                post_mean,
                post_n,
                origin_b,
                len(num_draft_for_collapse or []),
                tree_plan is not None,
                post_nd_use is not None,
            )
        if bundle is not None:
            self._populate_staged_profile_context_from_bundle(
                bundle=bundle,
                expansion_plan=expansion_plan,
                selected_rows=selected_rows,
                origin_batch_size=int(sampler_output.sampled_token_ids.shape[0]),
            )
        if tree_plan is None:
            self._emit_pivot_collapse_family_metadata(
                expansion_plan=expansion_plan,
                selected_rows=selected_rows,
                accepted_lens=pre_collapse_accept_lens,
                req_ids=list(
                    self.input_batch.req_ids[: int(sampler_output.sampled_token_ids.shape[0])]
                ),
            )
        self.pending_pivot_expansion_plan = None
        if bundle is not None and bundle.mode == "hierarchical_verification":
            req_ids = list(self.input_batch.req_ids[: sampler_output.sampled_token_ids.shape[0]])
            committed_lens: list[int] = []
            committed_last_tokens: list[int] = []
            for b, req_id in enumerate(req_ids):
                row = sampler_output.sampled_token_ids[b]
                committed = int((row != PLACEHOLDER_TOKEN_ID).sum().item())
                committed_lens.append(committed)
                last_token = (
                    int(row[committed - 1].item()) if committed > 0 else PLACEHOLDER_TOKEN_ID
                )
                committed_last_tokens.append(last_token)
                self._dit_debug_last_commit_len[req_id] = committed
                if committed > 0:
                    self._dit_debug_last_commit_token[req_id] = last_token
            self._dit_debug_assert(
                len(committed_lens) != len(bundle.num_draft_tokens)
                or all(
                    committed_lens[b] <= bundle.num_draft_tokens[b] + 1
                    for b in range(len(committed_lens))
                ),
                "check5_target_verification_commits_from_bundle",
                detail=(
                    f"committed={committed_lens}, "
                    f"bundle_num_draft={bundle.num_draft_tokens}"
                ),
            )
            self._dit_debug_event(
                "spechive_target_commit",
                {
                    "committed_lens": committed_lens,
                    "committed_last_tokens": committed_last_tokens,
                    "bundle_num_draft_tokens": bundle.num_draft_tokens,
                },
            )
        if (
            self.speculative_config is not None
            and self.speculative_config.use_draft_probs_in_rejection
            and self.drafter is not None
        ):
            clear_fn = getattr(self.drafter, "clear_draft_probs", None)
            if clear_fn is not None:
                clear_fn()
        if bundle is not None and self.drafter is not None:
            on_target_verification = getattr(
                self.drafter, "on_target_verification", None
            )
            if callable(on_target_verification):
                on_target_verification(
                    bundle=bundle,
                    sampled_token_ids=sampler_output.sampled_token_ids,
                )
            if bundle.mode == "hierarchical_verification":
                self._dit_debug_emit_summary(reason="target_verification_done")
        return sampler_output

    @staticmethod
    def _req_id_row_index_map(req_ids_snapshot: tuple[str, ...]) -> dict[str, int]:
        """Map req_id -> first row index in the profile snapshot (O(B) once)."""
        m: dict[str, int] = {}
        for i, rid in enumerate(req_ids_snapshot):
            if rid not in m:
                m[rid] = i
        return m

    @staticmethod
    def _first_draft_topk_for_profile_row(
        row_idx: int,
        topk_info: RootTopKInfo,
        k: int = 5,
    ) -> tuple[list[int] | None, list[float] | None, int]:
        if row_idx < 0:
            return None, None, 0
        b = int(topk_info.topk_token_ids.shape[0])
        if row_idx >= b:
            return None, None, 0
        kcols = int(topk_info.topk_token_ids.shape[1])
        k_eff = min(k, kcols)
        ids = topk_info.topk_token_ids[row_idx, :k_eff].detach().cpu().tolist()
        probs = topk_info.topk_probs[row_idx, :k_eff].detach().cpu().tolist()
        return ids, probs, k_eff

    def _reset_profile_first_draft_topk_for_step(self) -> None:
        d = getattr(self, "drafter", None)
        if d is not None and hasattr(d, "reset_profile_first_draft_topk"):
            d.reset_profile_first_draft_topk()

    def _get_profile_first_draft_topk_bundle(
        self,
    ) -> tuple[RootTopKInfo | None, tuple[str, ...] | None]:
        d = getattr(self, "drafter", None)
        if d is None:
            return None, None
        getter = getattr(d, "get_profile_first_draft_topk", None)
        if not callable(getter):
            return None, None
        return getter()

    def _maybe_capture_first_draft_target_top1_for_profile(
        self,
        spec_decode_metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        verity_sm: SamplingMetadata,
    ) -> None:
        """Record target top-1 at the first draft verify row per req_id for JSONL."""
        pctx = self._current_profile_ctx
        if pctx is None or not self._spec_profiler.mode.metadata_enabled:
            return
        if self.rejection_sampler is None:
            return
        tidx = spec_decode_metadata.target_logits_indices
        if tidx.numel() == 0:
            return
        rs = self.rejection_sampler
        proc_block = logits[tidx].to(torch.float32)
        if not rs.is_processed_logprobs_mode:
            proc_block = proc_block.clone()
        proc_block = rs.apply_logits_processors(
            proc_block,
            verity_sm,
            spec_decode_metadata,
        )
        proc_block = apply_sampling_constraints(
            proc_block,
            spec_decode_metadata.cu_num_draft_tokens,
            verity_sm,
        )
        cu_cpu = spec_decode_metadata.cu_num_draft_tokens.detach().cpu()
        num_draft = spec_decode_metadata.num_draft_tokens
        row_req_ids = self._spec_decode_metadata_row_req_ids(spec_decode_metadata)
        first_rows: list[int] = []
        first_rids: list[str] = []
        seen_rid: set[str] = set()
        prev = 0
        for i, nd in enumerate(num_draft):
            if nd > 0:
                rid = row_req_ids[i]
                if rid not in seen_rid:
                    seen_rid.add(rid)
                    first_rows.append(prev)
                    first_rids.append(rid)
            prev = int(cu_cpu[i].item())
        if not first_rows:
            return
        row_idx = torch.tensor(first_rows, device=proc_block.device, dtype=torch.long)
        sub = proc_block[row_idx]
        probs = torch.softmax(sub, dim=-1, dtype=torch.float32)
        top1_conf, top1_ids = torch.max(probs, dim=-1)
        top1_ids_cpu = top1_ids.detach().cpu()
        top1_conf_cpu = top1_conf.detach().cpu()
        out = pctx.first_draft_target_top1_by_req_id
        out.clear()
        for j, rid in enumerate(first_rids):
            out[rid] = (
                int(top1_ids_cpu[j].item()),
                float(top1_conf_cpu[j].item()),
            )

    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
        dict[str, int],
        LogprobsLists | None,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
        dict[str, int] | None,
    ]:
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        num_reqs = self.input_batch.num_reqs
        discard_sampled_tokens_req_indices = np.nonzero(
            self.discard_request_mask.np[:num_reqs]
        )[0]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        pivot_post_collapse_accepted: dict[str, int] | None = None
        if (
            self.speculative_config is not None
            and self.speculative_config.method == "pivot"
            and sampled_token_ids.shape[-1] > 1
        ):
            pivot_counts: dict[str, int] = {}
            for req_idx in range(num_sampled_tokens):
                req_id = req_ids_output_copy[req_idx]
                spec_toks = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
                if not spec_toks:
                    continue
                nd = len(spec_toks)
                row = sampled_token_ids[req_idx : req_idx + 1]
                c = get_target_verification_accepted_draft_prefix_lens(
                    row,
                    [nd],
                    placeholder_token_id=PLACEHOLDER_TOKEN_ID,
                )[0]
                pivot_counts[req_id] = int(c)
            pivot_post_collapse_accepted = pivot_counts or None
        invalid_req_indices = []
        logprobs_lists = None
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()

                if logprobs_tensors is not None:
                    logprobs_lists = logprobs_tensors.tolists()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            # With spec decoding, this is done in propose_draft_token_ids().
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
            pivot_post_collapse_accepted,
        )

    @contextmanager
    def synchronize_input_prep(self):
        if self.prepare_inputs_event is None:
            yield
            return

        # Ensure prior step has finished with reused CPU tensors.
        # This is required in the async scheduling case because
        # the CPU->GPU transfer happens async.
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """Helper method to call the model forward pass.

        This method can be overridden by subclasses for model execution.
        Motivation: We can inspect only this method versus
        the whole execute_model, which has additional logic.

        Args:
            input_ids: Input token IDs
            positions: Token positions
            intermediate_tensors: Tensors from previous pipeline stages
            inputs_embeds: Input embeddings (alternative to input_ids)
            **model_kwargs: Additional model arguments

        Returns:
            Model output tensor
        """
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """
        Checks if it's a decode batch with same amount scheduled tokens
        across all requests.
        """
        return (
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
        # be improved in model runner v2)
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        has_encoder_output = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Compute LoRA state for cudagraph dispatch
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)

        def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):
            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                num_active_loras=num_active_loras,
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
            )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output
        )
        num_tokens_padded = batch_descriptor.num_tokens
        if self.compilation_config.pass_config.enable_sp:
            assert (
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be "
                "a multiple of tensor parallel size"
            )

        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.parallel_config,
                    allow_microbatching=allow_microbatching,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    num_scheduled_tokens_per_request=num_scheduled_tokens_np,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct batch_descriptor
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # Assert to make sure the agreed upon token count is correct otherwise
                # num_tokens_across_dp will no-longer be valid
                assert batch_descriptor.num_tokens == num_tokens_padded

        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _register_layerwise_nvtx_hooks(self) -> None:
        """
        Register layerwise NVTX hooks if --enable-layerwise-nvtx-tracing is enabled
        to trace detailed information of each layer or module in the model.
        """

        if (
            self.vllm_config.observability_config.enable_layerwise_nvtx_tracing
            and not self.layerwise_nvtx_hooks_registered
        ):
            if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when CUDA graph is "
                    "turned off; you may observe part or all of the model "
                    "missing NVTX markers"
                )

            # In STOCK_TORCH_COMPILE mode, after registering hooks here,
            # the __call__ function of nn.module will be recompiled with
            # fullgraph=True. Since nvtx.range_push/pop are not traceable
            # by torch dynamo, we can't register hook functions here
            # because hook functions will also be traced by torch dynamo.
            if (
                self.vllm_config.compilation_config.mode
                == CompilationMode.STOCK_TORCH_COMPILE
            ):
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when "
                    "CompilationMode is STOCK_TORCH_COMPILE, skipping "
                    "function hooks registration"
                )
            else:
                pyt_hooks = PytHooks()
                pyt_hooks.register_hooks(self.model, self.model.__class__.__name__)
                self.layerwise_nvtx_hooks_registered = True

    def _get_slot_mappings(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_tokens_unpadded: int,
        ubatch_slices: "UBatchSlices | None" = None,
        *,
        input_batch: InputBatch | None = None,
    ) -> tuple[
        dict[int, torch.Tensor] | None,
        dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ]:
        """
        Build slot mappings in both formats needed by the system.

        Args:
            num_tokens_padded: Total number of tokens (padded)
            num_reqs_padded: Total number of requests (padded)
            num_tokens_unpadded: Actual number of tokens (unpadded)
            ubatch_slices: Optional ubatch slicing info for DBO

        Returns:
            A tuple of:
            - slot_mappings_by_gid: dict[int, torch.Tensor] for attention metadata
            - slot_mappings_by_layer: dict[str, torch.Tensor] or list for ForwardContext
        """
        if not (
            hasattr(self, "kv_cache_config")
            and self.kv_cache_config is not None
            and len(self.kv_cache_config.kv_cache_groups) > 0
        ):
            return None, None

        ib = input_batch if input_batch is not None else self.input_batch

        def _get_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = self.kv_cache_config.kv_cache_groups[
                kv_cache_gid
            ].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = ib.block_table[kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)

            return slot_mapping

        slot_mappings_by_gid = {
            gid: _get_slot_mapping(gid)
            for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups)
        }

        slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        for gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            slot_mapping = slot_mappings_by_gid[gid]
            for layer_name in kv_cache_group.layer_names:
                slot_mappings_by_layer[layer_name] = slot_mapping

        if ubatch_slices is not None:
            result: list[dict[str, torch.Tensor]] = []
            for ubatch in ubatch_slices:
                sliced_mappings: dict[str, torch.Tensor] = {}
                for layer_name, slot_mapping in slot_mappings_by_layer.items():
                    sliced_mappings[layer_name] = slot_mapping[ubatch.token_slice]
                result.append(sliced_mappings)
            return slot_mappings_by_gid, result

        return slot_mappings_by_gid, slot_mappings_by_layer

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()  # noqa
            else:
                logger.error("RoutedExpertsCapturer not initialized.")

        if scheduler_output.preempted_req_ids and has_kv_transfer_group():
            get_kv_transfer_group().handle_preemptions(
                scheduler_output.preempted_req_ids
            )

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            # Update persistent batch states.
            self._update_states(scheduler_output)
            self._sync_intermediate_states_with_scheduler(scheduler_output)
            self._sync_draft_states_with_scheduler(scheduler_output)

            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            logger.debug(
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                "should_ubatch: %s, num_tokens_across_dp: %s",
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
            )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            logger.debug(
                "ubatch_slices: %s, ubatch_slices_padded: %s",
                ubatch_slices,
                ubatch_slices_padded,
            )

            # True if any attention backend handles KV cache update separately
            # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
            # slot_mappings must use padded dimensions to match the key/value tensors.
            has_separate_kv_update = not all(
                all(
                    g.backend.forward_includes_kv_cache_update
                    for g in self.attn_groups[id]
                )
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
            )
            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            if self.cache_config.mamba_cache_mode == "align":
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                    self._get_mamba_copy_bufs(),
                )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
                num_tokens_padded=num_tokens_padded
                if pad_attn or has_separate_kv_update
                else num_tokens_unpadded,
                num_reqs_padded=(
                    num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
                ),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )

            attn_build = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded if pad_attn else None,
                max_query_len=max_num_scheduled_tokens,
                ubatch_slices=ubatch_slices_attn,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                slot_mappings=slot_mappings_by_group,
            )
            attn_metadata = attn_build.attn_metadata
            spec_decode_common_attn_metadata = (
                attn_build.spec_decode_common_attn_metadata
            )
            spec_decode_common_attn_metadata_by_gid = (
                attn_build.spec_decode_common_attn_metadata_by_gid
            )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # When spec decode is enabled, delay clearing connector metadata
        # until after draft model runs in sample_tokens.
        clear_kv_metadata = self.speculative_config is None
        spec_config = self.speculative_config
        full_verification_time_sec = 0.0

        # -- Unified profiler: begin step --
        _profiler = self._spec_profiler
        _fwd_reason = ("spec_verify" if use_spec_decode else "decode")
        _profile_ctx = _profiler.begin_step(
            scheduler_output.spec_profile_step_id, _fwd_reason)
        self._current_profile_ctx = _profile_ctx

        # -- Unified profiler: batch metadata --
        from vllm.v1.spec_decode.profiler_types import (  # noqa: E402
            SpecDecodeBatchMetadataRecord,
            SpecDecodeRequestMetadataRecord,
        )
        _profiler.emit_batch_metadata(SpecDecodeBatchMetadataRecord(
            step_id=scheduler_output.spec_profile_step_id,
            speculative_method=(
                spec_config.method if spec_config is not None else ""),
            batch_size=num_reqs,
            total_num_scheduled_tokens=(
                scheduler_output.total_num_scheduled_tokens),
            num_tokens_unpadded=num_tokens_unpadded,
            num_tokens_padded=num_tokens_padded,
            max_query_len=max_num_scheduled_tokens,
            speculative_decode_active=use_spec_decode,
            cudagraph_mode=str(cudagraph_mode),
        ))

        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
                skip_compiled=has_encoder_input,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(
                scheduler_output, clear_metadata=clear_kv_metadata
            ) as kv_connector_output,
        ):
            t_forward_start = time.perf_counter()

            # -- Unified profiler: target_verify stage (only for spec verify) --
            _do_deep = (
                self._deep_profiler is not None
                and _fwd_reason == "spec_verify"
                and self._deep_profiler.should_profile_step(
                    scheduler_output.spec_profile_step_id)
                and self._deep_profiler.should_profile_stage("target_verify")
            )

            if _fwd_reason == "spec_verify":
                _profiler.snapshot_memory_before("target_verify")
                _profiler.start_stage("target_verify")

            if _do_deep:
                with self._deep_profiler.profile_stage(
                    "target_verify",
                    ctx=_profile_ctx,
                    origin_batch_size=num_reqs,
                    effective_batch_size=num_reqs,
                    num_tokens=num_tokens_unpadded,
                ):
                    model_output = self._model_forward(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs,
                    )
            else:
                model_output = self._model_forward(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if _fwd_reason == "spec_verify":
                _profiler.end_stage("target_verify")
                _profiler.snapshot_memory_after("target_verify")
                _tv_shape: dict[str, Any] = {
                    "origin_batch_size": num_reqs,
                    "effective_batch_size": num_reqs,
                    "num_tokens_processed": num_tokens_unpadded,
                }
                _tv_cam = spec_decode_common_attn_metadata
                if _tv_cam is not None:
                    _tv_shape["max_seq_len"] = int(_tv_cam.max_seq_len)
                    _tv_shape["sum_seq_lens"] = int(
                        _tv_cam.seq_lens.sum().item())
                _profiler.snapshot_shape("target_verify", _tv_shape)
                if _tv_cam is not None:
                    _q_lens = []
                    _qloc_cpu = _tv_cam.query_start_loc_cpu
                    for i in range(num_reqs):
                        _q_lens.append(int(
                            _qloc_cpu[i + 1] - _qloc_cpu[i]))
                    _kv_lens = _tv_cam.seq_lens.tolist()
                    _profiler.record_kv_proxies(
                        "target_verify",
                        q_lens=_q_lens,
                        kv_lens=[int(k) for k in _kv_lens],
                        block_table=_tv_cam.block_table_tensor,
                    )


        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        kv_connector_output,
                    )

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_tokens_padded
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
            spec_decode_common_attn_metadata_by_gid,
            full_verification_time_sec,
        )
        self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        if self.execute_model_state is None:
            kv_connector_output = self.kv_connector_output
            self.kv_connector_output = None
            # receive sampled token ids from the last PP rank.
            if self.use_async_scheduling and get_pp_group().world_size > 1:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
            spec_decode_common_attn_metadata_by_gid,
            full_verification_time_sec,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        spec_config = self.speculative_config

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        # -- Profiler: reject_sample for target-level sampling/rejection --
        if use_spec_decode:
            self._spec_profiler.start_stage("reject_sample")
        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)
        if use_spec_decode:
            self._spec_profiler.end_stage("reject_sample")

        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
        # Mirror hybrid acceptance counts; full mirror vs target token alignment runs
        # in _reconcile_intermediate_frontier_after_target after bookkeeping.
        if self._intermediate_kv_frontier_enabled():
            self._sync_intermediate_num_accepted_from_target()
            self._sync_draft_num_accepted_from_target()
        if self.use_async_scheduling:
            pp = get_pp_group()
            # For torchrun external_launcher PP mode with broadcast_pp_output=True,
            # PP outputs have been broadcasted to all ranks at logits computation.
            # Therefore, here is no need to send sampled token ids again in this case.
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(
                    sampler_output.sampled_token_ids
                )

        self._draft_token_ids = None
        self._draft_token_req_ids = None
        self.input_batch.prev_sampled_token_ids = None

        if use_spec_decode and self._current_profile_ctx is not None:
            self._reset_profile_first_draft_topk_for_step()

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,
                    spec_decode_common_attn_metadata_by_gid,
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
            use_gpu_toks = (
                spec_config.uses_gpu_sampled_tokens_for_drafting()
            ) and not spec_config.disable_padded_drafter_batch
            if use_gpu_toks:
                # EAGLE/DraftModel/HV/etc. can use the GPU sampled tokens as inputs and
                # does not need to wait for bookkeeping to finish.
                assert isinstance(
                    self.drafter,
                    EagleProposer
                    | DraftModelProposer
                    | AdaptiveSpechiveProposer
                    | HierarchicalVerificationProposer
                    | PivotProposer
                    | ExtractHiddenStatesProposer,
                )
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    self._spec_profiler.snapshot_memory_before(
                        "draft_forward")
                    self._spec_profiler.start_stage("draft_forward")
                    propose_draft_token_ids(sampled_token_ids)
                    self._spec_profiler.end_stage("draft_forward")
                    self._spec_profiler.snapshot_memory_after(
                        "draft_forward")
                    _df_shape: dict[str, Any] = {
                        "origin_batch_size": len(
                            self.input_batch.req_ids),
                        "effective_batch_size": len(
                            self.input_batch.req_ids),
                        "num_tokens_processed": len(
                            self.input_batch.req_ids) * (
                            self.num_spec_tokens),
                    }
                    _df_cam = spec_decode_common_attn_metadata
                    if _df_cam is not None:
                        _df_shape["max_seq_len"] = int(
                            _df_cam.max_seq_len)
                        _df_shape["sum_seq_lens"] = int(
                            _df_cam.seq_lens.sum().item())
                    self._spec_profiler.snapshot_shape(
                        "draft_forward", _df_shape)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count = (
                        self.drafter.prepare_next_token_ids_padded(
                            spec_decode_common_attn_metadata,
                            sampled_token_ids,
                            self.requests,
                            self.input_batch,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
                    # Since we couldn't run the drafter,
                    # just use zeros for the draft tokens.
                    self._draft_token_ids = torch.zeros(
                        1, device=self.device, dtype=torch.int32
                    ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
            else:
                propose_drafts_after_bookkeeping = input_fits_in_drafter

        # -- Profiler: bookkeeping stage --
        self._spec_profiler.start_stage("bookkeeping")
        _bk_t0 = time.perf_counter()
        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
                pivot_post_collapse_accepted,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
            )
        if self._intermediate_kv_frontier_enabled():
            self._reconcile_intermediate_frontier_after_target(
                scheduler_output,
                req_ids_output_copy,
                sampled_token_ids=sampler_output.sampled_token_ids,
                spec_decode_metadata=spec_decode_metadata,
            )
            self._reconcile_draft_frontier_after_target(
                scheduler_output,
                req_ids_output_copy,
                sampled_token_ids=sampler_output.sampled_token_ids,
                spec_decode_metadata=spec_decode_metadata,
            )
        self._spec_profiler.end_stage("bookkeeping")
        if self._current_profile_ctx is not None:
            self._current_profile_ctx.bookkeeping_metadata_build_ms = (
                (time.perf_counter() - _bk_t0) * 1000.0)

        if propose_drafts_after_bookkeeping:
            self._spec_profiler.snapshot_memory_before("draft_forward")
            self._spec_profiler.start_stage("draft_forward")
            propose_draft_token_ids(valid_sampled_token_ids)
            self._spec_profiler.end_stage("draft_forward")
            self._spec_profiler.snapshot_memory_after("draft_forward")
            _df2_shape: dict[str, Any] = {
                "origin_batch_size": len(self.input_batch.req_ids),
                "effective_batch_size": len(self.input_batch.req_ids),
            }
            _df2_cam = spec_decode_common_attn_metadata
            if _df2_cam is not None:
                _df2_shape["max_seq_len"] = int(_df2_cam.max_seq_len)
                _df2_shape["sum_seq_lens"] = int(
                    _df2_cam.seq_lens.sum().item())
            self._spec_profiler.snapshot_shape(
                "draft_forward", _df2_shape)

        # Clear KV connector metadata after draft model runs (if spec decode).
        # This was deferred from target model forward to allow draft model
        # to also save its KV cache.
        if self.speculative_config is not None:
            self.clear_kv_connector_metadata()

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        # self.kv_connector_output may be modified during drafting
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            if self.model_config.enable_return_routed_experts:
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.save_captured_experts(indices=self.slot_mapping)  # noqa
                else:
                    logger.error("RoutedExpertsCapturer not initialized.")

            # Populate legacy cost breakdown from unified profiler data.
            spec_decode_cost_breakdown = None
            _pctx_for_breakdown = self._current_profile_ctx
            if (use_spec_decode and spec_config is not None
                    and _pctx_for_breakdown is not None
                    and _pctx_for_breakdown.stage_timings):
                _st = _pctx_for_breakdown.stage_timings
                _draft_ms = sum(_st.get("draft_forward", []))
                _target_ms = sum(_st.get("target_verify", []))
                _reject_ms = sum(_st.get("reject_sample", []))
                _inter_ms = sum(_st.get("intermediate_verify", []))
                _expand_ms = sum(_st.get("expand_collapse", []))
                num_reqs = len(req_ids_output_copy)
                _partial_acc = (
                    _pctx_for_breakdown.partial_accepted_per_req
                    if _pctx_for_breakdown.partial_accepted_per_req
                    else [0] * num_reqs
                )
                spec_decode_cost_breakdown = SpecDecodeCostBreakdown(
                    draft_time_sec=_draft_ms / 1000.0,
                    compression_time_sec=_expand_ms / 1000.0,
                    partial_verification_time_sec=_inter_ms / 1000.0,
                    full_verification_time_sec=_target_ms / 1000.0,
                    num_partial_accepted_per_req=_partial_acc,
                    reject_sample_time_sec=_reject_ms / 1000.0,
                )

            # -- Unified profiler: emit request + family metadata --
            from vllm.v1.spec_decode.profiler_types import (
                SpecDecodeRequestMetadataRecord,
            )
            if use_spec_decode and spec_decode_metadata is not None:
                _step_id = scheduler_output.spec_profile_step_id
                _max_rps = getattr(
                    self._spec_profiler, "_max_reqs_per_step", None)
                _incl_tids = getattr(
                    self._spec_profiler, "_include_token_ids", False)
                _pctx_rm = self._current_profile_ctx
                _topk_bundle_info, _topk_snap = (
                    self._get_profile_first_draft_topk_bundle())
                _topk_idx_map = (
                    self._req_id_row_index_map(_topk_snap)
                    if _topk_snap is not None else {})
                _topk_rows_ok = (
                    _topk_bundle_info is not None
                    and _topk_snap is not None
                    and _topk_bundle_info.topk_token_ids.shape[0] == len(
                        _topk_snap))
                _req_emitted = 0
                for _ri, _rid in enumerate(req_ids_output_copy):
                    if _max_rps is not None and _req_emitted >= _max_rps:
                        break
                    _sd_toks = (
                        scheduler_output.scheduled_spec_decode_tokens
                        .get(_rid))
                    _n_draft = len(_sd_toks) if _sd_toks else 0
                    # Non-pivot default:
                    # parse_output row is [accepted draft tokens..., bonus/recovery],
                    # so accepted draft-prefix length is len(row) - 1.
                    _n_accepted = 0
                    if (_n_draft > 0 and not self.use_async_scheduling
                            and _ri < len(valid_sampled_token_ids)):
                        _gen_ids = valid_sampled_token_ids[_ri]
                        if _gen_ids:
                            _n_accepted = max(
                                0, min(_n_draft, len(_gen_ids) - 1))
                    # Pivot override: use post-collapse accepted draft-prefix len.
                    if (pivot_post_collapse_accepted is not None
                            and _rid in pivot_post_collapse_accepted):
                        _n_accepted = int(
                            min(
                                _n_draft,
                                pivot_post_collapse_accepted[_rid],
                            ))

                    # Intermediate/staged fields from profile context
                    _n_inter_verified = 0
                    _n_inter_accepted = 0
                    _n_partial_accepted = 0
                    _staged_depth = 0
                    _staged_used = False
                    if _pctx_rm is not None:
                        if (_pctx_rm.inter_verified_per_req
                                and _ri < len(
                                    _pctx_rm.inter_verified_per_req)):
                            _n_inter_verified = (
                                _pctx_rm.inter_verified_per_req[_ri])
                        if (_pctx_rm.inter_accepted_per_req
                                and _ri < len(
                                    _pctx_rm.inter_accepted_per_req)):
                            _n_inter_accepted = (
                                _pctx_rm.inter_accepted_per_req[_ri])
                        if (_pctx_rm.partial_accepted_per_req
                                and _ri < len(
                                    _pctx_rm.partial_accepted_per_req)):
                            _n_partial_accepted = (
                                _pctx_rm.partial_accepted_per_req[_ri])
                        _staged_depth = _pctx_rm.staged_verification_depth
                        _staged_used = _pctx_rm.staged_verification_used

                    # Before/after computed tokens
                    _n_comp_before = 0
                    _n_comp_after = 0
                    if _ri < len(self.input_batch.req_ids):
                        _n_comp_before = int(
                            self.input_batch.num_tokens_no_spec[_ri])
                        _n_comp_after = _n_comp_before + _n_accepted

                    _fd_tid: list[int] | None = None
                    _fd_conf: list[float] | None = None
                    _fd_k = 0
                    _fd_src: str | None = None
                    if _topk_rows_ok and _topk_bundle_info is not None:
                        _row = _topk_idx_map.get(_rid)
                        if _row is not None:
                            _fd_tid, _fd_conf, _fd_k = (
                                self._first_draft_topk_for_profile_row(
                                    _row, _topk_bundle_info))
                            if _fd_tid is not None:
                                _fd_src = "profile_first_draft_topk"
                    if _fd_src is None:
                        _fd_src = "unavailable"
                        _fd_tid = None
                        _fd_conf = None
                        _fd_k = 0

                    _tgt1_tid: int | None = None
                    _tgt1_conf: float | None = None
                    _tgt1_src: str | None = None
                    if _pctx_rm is not None:
                        _tgt_map = _pctx_rm.first_draft_target_top1_by_req_id
                        if _rid in _tgt_map:
                            _tgt1_tid, _tgt1_conf = _tgt_map[_rid]
                            _tgt1_src = "target_verify_logits"
                    if _tgt1_src is None:
                        _tgt1_src = "unavailable"

                    _req_rec = SpecDecodeRequestMetadataRecord(
                        step_id=_step_id,
                        req_id=_rid,
                        req_index=_ri,
                        num_draft_tokens=_n_draft,
                        num_accepted_tokens=_n_accepted,
                        num_rejected_tokens=max(
                            0, _n_draft - _n_accepted),
                        num_intermediate_verified_tokens=(
                            _n_inter_verified),
                        num_intermediate_accepted_tokens=(
                            _n_inter_accepted),
                        num_partial_accepted_tokens=_n_partial_accepted,
                        staged_verification_depth=_staged_depth,
                        staged_verification_used=_staged_used,
                        num_computed_tokens_before=_n_comp_before,
                        num_computed_tokens_after=_n_comp_after,
                        first_draft_topk_token_ids=_fd_tid,
                        first_draft_topk_confidences=_fd_conf,
                        first_draft_topk_k=_fd_k,
                        first_draft_topk_source=_fd_src,
                        first_draft_target_top1_token_id=_tgt1_tid,
                        first_draft_target_top1_confidence=_tgt1_conf,
                        first_draft_target_top1_source=_tgt1_src,
                    )
                    # Attach token IDs if configured
                    if _incl_tids and _sd_toks:
                        _req_rec.draft_token_ids = list(_sd_toks)
                        if (_ri < len(valid_sampled_token_ids)
                                and valid_sampled_token_ids[_ri]):
                            _req_rec.accepted_token_ids = list(
                                valid_sampled_token_ids[_ri])

                    self._spec_profiler.emit_request_metadata(_req_rec)
                    _req_emitted += 1

            # -- Unified profiler: finalize step --
            _profile_transport = None
            _pctx = self._current_profile_ctx
            if _pctx is not None:
                _profile_transport = self._spec_profiler.finalize_step(
                    _pctx)
                self._current_profile_ctx = None

            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
                spec_decode_cost_breakdown=spec_decode_cost_breakdown,
                pivot_post_collapse_accepted_draft_tokens=pivot_post_collapse_accepted,
                spec_decode_profile_transport=_profile_transport,
            )

        if not self.use_async_scheduling:
            return output

        with record_function_or_nullcontext(
            "gpu_model_runner: AsyncGPUModelRunnerOutput"
        ):
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
        with record_function_or_nullcontext(
            "gpu_model_runner: set_async_sampled_token_ids"
        ):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

    def _pp_broadcast_prev_sampled_token_ids(
        self, sampled_token_ids: torch.Tensor
    ) -> None:
        """Broadcast sampled token ids (GPU) from last PP stage"""
        pp = get_pp_group()
        assert pp.is_last_rank
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        assert sampled_token_ids.dim() == 2 and sampled_token_ids.shape[-1] == 1, (
            "PP+async expects sampled_token_ids to have shape [num_reqs, 1]"
        )
        torch.distributed.broadcast(
            sampled_token_ids, src=pp.rank, group=pp.device_group
        )

    def _pp_receive_prev_sampled_token_ids_to_input_batch(self) -> None:
        """Receive sampled token ids broadcast from last PP stage"""
        pp = get_pp_group()
        assert not pp.is_last_rank
        num_reqs = self.input_batch.num_reqs
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)
        torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
        self.input_batch.prev_sampled_token_ids = recv

        # construct `prev_req_id_to_index` here so `_prepare_input_ids`
        # can map req_id -> previous batch row
        discard_req_indices = np.nonzero(self.discard_request_mask.np[:num_reqs])[0]
        discard_req_indices_set = set(discard_req_indices)
        prev_req_id_to_index: dict[str, int] = {}
        for i, req_id in enumerate(self.input_batch.req_ids):
            if i in discard_req_indices_set:
                continue
            prev_req_id_to_index[req_id] = i
            # PP+async scheduling: advance per-request local cached output length by
            # appending a placeholder (-1) token id.
            if (req_state := self.requests.get(req_id)) is not None:
                req_state.output_token_ids.append(-1)
        self.input_batch.prev_req_id_to_index = prev_req_id_to_index

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        return DraftTokenIds(req_ids, draft_token_ids)

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        # Check if we need to copy draft tokens to CPU. In async scheduling,
        # we only copy when needed for structured output, penalties or bad_words.
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        # We must also set the corresponding request ids.
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_copy_stream is not None
        assert self.draft_token_ids_cpu is not None
        default_stream = torch.cuda.current_stream()
        num_reqs = draft_token_ids.shape[0]
        with torch.cuda.stream(self.draft_token_ids_copy_stream):
            if not zeros_only:
                # Trigger async copy of draft token ids to cpu.
                self.draft_token_ids_copy_stream.wait_stream(default_stream)
                self.draft_token_ids_cpu[:num_reqs].copy_(
                    draft_token_ids, non_blocking=True
                )
            else:
                # No copy needed, just zero-out cpu tensor.
                self.draft_token_ids_cpu[:num_reqs] = 0
            self.draft_token_ids_event.record()

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        if isinstance(self._draft_token_ids, list):
            return self._draft_token_ids, self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        if req_ids is None:
            return [], []
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_cpu is not None
        self.draft_token_ids_event.synchronize()
        rows = self.draft_token_ids_cpu[: len(req_ids)].tolist()
        # Match prior list export from ``propose_draft_token_ids``: omit placeholder
        # draft ids for the scheduler while keeping the full padded tensor on GPU
        # for async input-id scatter.
        return (
            [
                [int(tok) for tok in row if int(tok) != PLACEHOLDER_TOKEN_ID]
                for row in rows
            ],
            req_ids,
        )

    def _pad_ragged_draft_token_lists_to_tensor(
        self, rows: list[list[int]]
    ) -> torch.Tensor:
        """Pack per-request draft token lists into a dense GPU tensor.

        ``apply_tetris`` returns ragged Python lists; async
        :meth:`_prepare_input_ids` requires ``_draft_token_ids`` to be a
        tensor shaped ``(batch, num_spec_tokens)`` with
        ``PLACEHOLDER_TOKEN_ID`` padding so scatter indices stay valid.
        """
        if not rows:
            return torch.empty(
                (0, self.num_spec_tokens),
                device=self.device,
                dtype=torch.int64,
            )
        out = torch.full(
            (len(rows), self.num_spec_tokens),
            PLACEHOLDER_TOKEN_ID,
            device=self.device,
            dtype=torch.int64,
        )
        for i, row in enumerate(rows):
            if not row:
                continue
            li = min(len(row), self.num_spec_tokens)
            if li:
                out[i, :li] = torch.tensor(
                    row[:li], device=self.device, dtype=torch.int64
                )
        return out

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        default_stream = torch.cuda.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            assert counts_cpu is not None
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        # Wait until valid_sampled_tokens_count is copied to cpu,
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        sampled_count_event = self.valid_sampled_token_count_event
        if sampled_count_event is None or prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        assert counts_cpu is not None
        sampled_count_event.synchronize()
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
        spec_decode_cad_by_gid: dict[int, CommonAttentionMetadata] | None = None,
    ) -> list[list[int]] | torch.Tensor:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        spec_config = self.speculative_config
        assert spec_config is not None
        self.set_pending_hybrid_spec_bundle(None)
        if spec_config.method == "ngram":
            from vllm.v1.spec_decode.ngram_proposer import NgramProposer

            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.drafter.propose(
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                slot_mappings=slot_mappings,
            )
        elif spec_config.method == "suffix":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, SuffixDecodingProposer)
            draft_token_ids = self.drafter.propose(
                self.input_batch, sampled_token_ids, slot_mappings=slot_mappings
            )
        elif spec_config.method == "medusa":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, MedusaProposer)

            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                assert spec_decode_metadata is not None, (
                    "No spec decode metadata for medusa"
                )
                for num_draft, tokens in zip(
                    spec_decode_metadata.num_draft_tokens, sampled_token_ids
                ):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = torch.tensor(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            draft_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
                slot_mappings=slot_mappings,
            )
        elif spec_config.uses_extract_hidden_states():
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)
            assert isinstance(sampled_token_ids, torch.Tensor), (
                "sampled_token_ids should be a torch.Tensor for "
                "extract_hidden_states method."
            )
            if not self.use_aux_hidden_state_outputs or aux_hidden_states is None:
                raise ValueError(
                    "aux_hidden_states are required when using `extract_hidden_states`"
                )
            target_hidden_states = [h[:num_scheduled_tokens] for h in aux_hidden_states]

            draft_token_ids, drafter_kv_connector_output = self.drafter.propose(
                sampled_token_ids=sampled_token_ids,
                target_hidden_states=target_hidden_states,
                common_attn_metadata=common_attn_metadata,
                scheduler_output=scheduler_output,
                slot_mappings=slot_mappings,
            )
            # Combine KVConnectorOutputs or select the non-empty one
            if self.kv_connector_output and drafter_kv_connector_output:
                self.kv_connector_output = KVConnectorOutput.merge(
                    self.kv_connector_output, drafter_kv_connector_output
                )
            else:
                self.kv_connector_output = (
                    self.kv_connector_output or drafter_kv_connector_output
                )

            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    common_attn_metadata,
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_mask.gpu,
                )
            )
            self._copy_valid_sampled_token_count(
                next_token_ids, valid_sampled_tokens_count
            )

        elif spec_config.uses_model_based_drafter():
            assert isinstance(
                self.drafter,
                EagleProposer
                | DraftModelProposer
                | AdaptiveSpechiveProposer
                | HierarchicalVerificationProposer
                | PivotProposer,
            )

            if spec_config.disable_padded_drafter_batch:
                # When padded-batch is disabled, the sampled_token_ids should be
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), (
                    "sampled_token_ids should be a python list when"
                    "padded-batch is disabled."
                )
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    scheduler_output.num_scheduled_tokens,
                )
            else:
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), (
                    "sampled_token_ids should be a torch.Tensor when"
                    "padded-batch is enabled."
                )
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )

            num_rejected_tokens_gpu = None
            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                if spec_config.disable_padded_drafter_batch:
                    token_indices_to_sample = None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens,
                    )
                    target_token_ids = self.input_ids.gpu[token_indices]
                    target_positions = self._get_positions(token_indices)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[token_indices] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[token_indices]
                else:
                    (
                        common_attn_metadata,
                        token_indices_to_sample,
                        num_rejected_tokens_gpu,
                    ) = self.drafter.prepare_inputs_padded(
                        common_attn_metadata,
                        spec_decode_metadata,
                        valid_sampled_tokens_count,
                    )
                    total_num_tokens = common_attn_metadata.num_actual_tokens
                    # When padding the batch, token_indices is just a range
                    target_token_ids = self.input_ids.gpu[:total_num_tokens]
                    target_positions = self._get_positions(total_num_tokens)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[:total_num_tokens] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[:total_num_tokens]

            if self.supports_mm_inputs and self.drafter.supports_mm_inputs:
                mm_embed_inputs = self._gather_mm_embeddings(
                    scheduler_output,
                    shift_computed_tokens=1,
                )
            else:
                mm_embed_inputs = None

            intermediate_frontier_sched_capture = (
                self._static_intermediate_kv_frontier_enabled(self.vllm_config)
            )
            if intermediate_frontier_sched_capture:
                self._hv_scheduler_output = scheduler_output
            try:
                propose_kw: dict[str, Any] = dict(
                    target_token_ids=target_token_ids,
                    target_positions=target_positions,
                    target_hidden_states=target_hidden_states,
                    next_token_ids=next_token_ids,
                    token_indices_to_sample=token_indices_to_sample,
                    sampling_metadata=sampling_metadata,
                    common_attn_metadata=common_attn_metadata,
                    mm_embed_inputs=mm_embed_inputs,
                    num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                    slot_mappings=slot_mappings,
                )
                if spec_config.method == "hierarchical_verification" and isinstance(
                    self.drafter, HierarchicalVerificationProposer
                ):
                    propose_kw["spec_decode_common_attn_metadata_by_gid"] = (
                        spec_decode_cad_by_gid
                    )
                draft_token_ids = self.drafter.propose(**propose_kw)
                self._take_drafter_staged_hybrid_and_publish(spec_decode_metadata)
            finally:
                if intermediate_frontier_sched_capture:
                    self._hv_scheduler_output = None

        # ---- TETRIS post-processing ----------------------------------------
        spec_config = self.speculative_config
        if (
            spec_config is not None
            and getattr(spec_config, "tetris", False)
            and isinstance(draft_token_ids, torch.Tensor)
            and hasattr(self.drafter, "last_draft_logprobs")
            and self.drafter.last_draft_logprobs is not None
        ):
            tetris_rows = apply_tetris(
                draft_token_ids=draft_token_ids,
                draft_token_logprobs=self.drafter.last_draft_logprobs,
                base_k=getattr(spec_config, "tetris_base_k", None)
                or spec_config.num_speculative_tokens,
                extra_proposals=getattr(
                    spec_config, "tetris_extra_proposals", 0
                ),
                turn_on_batch_size=getattr(
                    spec_config, "tetris_turn_on_batch_size", None
                ),
            )
            draft_token_ids = self._pad_ragged_draft_token_lists_to_tensor(
                tetris_rows
            )
        elif (
            isinstance(draft_token_ids, torch.Tensor)
            and spec_config is not None
            and getattr(spec_config, "tetris", False)
            and not (
                hasattr(self.drafter, "last_draft_logprobs")
                and self.drafter.last_draft_logprobs is not None
            )
        ):
            # TETRIS on but logprobs missing: narrow columns to base_k; keep tensor
            # so async ``_prepare_input_ids`` can scatter padded draft rows.
            logger.warning_once(
                "TETRIS is enabled but draft logprobs are missing "
                "(drafter.last_draft_logprobs is None); falling back "
                "to base_k tokens without TETRIS. Typical causes: "
                "parallel_drafting, use_local_argmax_reduction, or a "
                "proposer path that does not record per-step logprobs.",
                scope="local",
            )
            num_spec = draft_token_ids.shape[1]
            base_k = getattr(spec_config, "tetris_base_k", None)
            if base_k is not None:
                num_spec = base_k
            else:
                extra = getattr(spec_config, "tetris_extra_proposals", 0)
                num_spec = max(1, num_spec - extra)
            draft_token_ids = draft_token_ids[:, :num_spec].contiguous()
        # ---- end TETRIS ----------------------------------------------------

        return draft_token_ids

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}"
            )
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    @instrument(span_name="Loading (GPU)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        """
        Args:
            load_dummy_weights: load dummy weights instead of real weights.
        """
        logger.info_once(
            "Starting to load model %s...",
            self.model_config.model,
            scope="global",
        )

        if self.parallel_config.enable_eplb:
            self.eplb_state = EplbState(self.parallel_config, self.device)
            eplb_models = 0

        try:
            with DeviceMemoryProfiler() as m:
                time_before_load = time.perf_counter()
                if load_dummy_weights:
                    self.load_config.load_format = "dummy"
                model_loader = get_model_loader(self.load_config)
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                )
                if self.lora_config:
                    self.model = self.load_lora_model(
                        self.model, self.vllm_config, self.device
                    )
                if self.drafter is not None:
                    logger.info_once("Loading drafter model...")
                    self.drafter.load_model(self.model)
                    if (
                        hasattr(self.drafter, "model")
                        and is_mixture_of_experts(self.drafter.model)
                        and self.parallel_config.enable_eplb
                    ):
                        assert not self.parallel_config.enable_elastic_ep, (
                            "Elastic EP is not supported with drafter model."
                        )
                        spec_config = self.vllm_config.speculative_config
                        assert spec_config is not None
                        assert spec_config.draft_model_config is not None
                        logger.info_once(
                            "EPLB is enabled for drafter model %s.",
                            spec_config.draft_model_config.model,
                        )
                        if self.eplb_state is None:
                            self.eplb_state = EplbState(
                                self.parallel_config, self.device
                            )
                        self.eplb_state.add_model(
                            self.drafter.model,
                            spec_config.draft_model_config,
                        )
                        eplb_models += 1

                if self.use_aux_hidden_state_outputs:
                    if not supports_eagle3(self.get_model()):
                        raise RuntimeError(
                            "Model does not support EAGLE3 interface but "
                            "aux_hidden_state_outputs was requested"
                        )

                    # Try to get auxiliary layers from speculative config,
                    # otherwise use model's default layers
                    aux_layers = self._get_eagle3_aux_layers_from_config()
                    if aux_layers:
                        logger.info(
                            "Using auxiliary layers from speculative config: %s",
                            aux_layers,
                        )
                    else:
                        aux_layers = self.model.get_eagle3_aux_hidden_state_layers()

                    self.model.set_aux_hidden_state_layers(aux_layers)
                time_after_load = time.perf_counter()
            self.model_memory_usage = m.consumed_memory
        except torch.cuda.OutOfMemoryError as e:
            msg = (
                "Failed to load model - not enough GPU memory. "
                "Try lowering --gpu-memory-utilization to free memory for weights, "
                "increasing --tensor-parallel-size, or using --quantization. "
                "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
                "for more tips."
            )
            combined_msg = f"{msg} (original error: {e})"
            logger.error(combined_msg)
            raise e
        logger.info_once(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(self.model_memory_usage),
            time_after_load - time_before_load,
            scope="local",
        )
        if not load_dummy_weights:
            prepare_communication_buffer_for_model(self.model)
            if (drafter := getattr(self, "drafter", None)) and (
                drafter_model := getattr(drafter, "model", None)
            ):
                prepare_communication_buffer_for_model(drafter_model)
            if (drafter := getattr(self, "drafter", None)) and (
                im := getattr(drafter, "intermediate_model", None)
            ):
                prepare_communication_buffer_for_model(im)
        mm_config = self.model_config.multimodal_config
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.get_model())
            and mm_config is not None
            and mm_config.is_multimodal_pruning_enabled()
        )

        if (
            is_mixture_of_experts(self.model)
            and self.parallel_config.enable_eplb
            and not load_dummy_weights
        ):
            logger.info_once("EPLB is enabled for model %s.", self.model_config.model)
            assert self.eplb_state is not None
            self.eplb_state.add_model(
                self.model,
                self.model_config,
            )
            if self.eplb_state.is_async:
                self.eplb_state.start_async_loop()

        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
        ):
            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
            compilation_counter.stock_torch_compile_count += 1
            self.model.compile(fullgraph=True, backend=backend)
            return
        # for other compilation modes, cudagraph behavior is controlled by
        # CudagraphWraper and CudagraphDispatcher of vllm.

        # wrap the model with full cudagraph wrapper if needed.
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        if (
            cudagraph_mode.has_full_cudagraphs()
            and not self.parallel_config.use_ubatching
        ):
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        elif self.parallel_config.use_ubatching:
            if cudagraph_mode.has_full_cudagraphs():
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.FULL, self.device
                )
            else:
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.NONE, self.device
                )

        get_offloader().post_init()

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        """Extract Eagle3 auxiliary layer indices from speculative config.

        These indices specify which hidden states from the base model should
        be used as auxiliary inputs for the Eagle3 drafter model during
        speculative decoding.

        Returns:
            Tuple of layer indices if found in draft model config,
            None otherwise.
        """
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        hf_config = self.speculative_config.draft_model_config.hf_config
        if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):
            return None

        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return None

    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None:
        """
        Reload weights from a weights iterator or from disk

        :param weights_iterator: weights to load into model
        :param weights_path: path to load weights from if weights_iterator is not
            provided. Use path of original model if neither is provided.
        :param is_checkpoint_format: set to False if weights have already been processed
            into kernel format (repacking, renaming, ect.)
        """
        # TODO(@kylesayrs): generalize to all runners and loaders
        # argument validation
        if weights_iterator is None and not is_checkpoint_format:
            logger.warning(
                "Reloading from disk means that weights will be in checkpoint format. "
                "Please use `is_checkpoint_format=True` "
                "to avoid weight reloading errors"
            )

        model = self.get_model()
        weights_to_load = {name for name, _ in model.named_parameters()}
        counter_before_reloading = time.perf_counter()

        # load weights from disk if none are provided
        if weights_iterator is None:
            model_loader = get_model_loader(self.load_config)
            if not hasattr(model_loader, "get_all_weights"):
                raise NotImplementedError(
                    f"Model reloading with `{self.load_config.load_format}` format"
                )

            if weights_path is not None:
                self.model_config.model = weights_path
            weights_iterator = model_loader.get_all_weights(self.model_config, model)
            weights_iterator = cast(
                Iterable[tuple[str, torch.Tensor]], weights_iterator
            )

        # begin loading weights
        logger.info_once("Reloading weights inplace...", scope="local")
        load_device = (
            self.vllm_config.load_config.device or self.vllm_config.device_config.device
        )
        with torch.device(load_device):
            if is_checkpoint_format:
                # load weights from checkpoint/ original model format
                initialize_layerwise_reload(model)
                loaded_weights = model.load_weights(weights_iterator)
                finalize_layerwise_reload(model, self.model_config)

            else:
                # load weights from kernel format
                logger.warning_once(
                    "Reloading with `is_checkpoint_format=True` requires that "
                    "weights be in kernel format and already sharded",
                    scope="local",
                )
                loaded_weights = set()
                for name, loaded_weight in weights_iterator:
                    param = model.get_parameter(name)  # TODO: buffers?
                    param.copy_(loaded_weight)
                    loaded_weights.add(name)

        # logging and validation
        counter_after_reloading = time.perf_counter()
        diff_seconds = counter_after_reloading - counter_before_reloading
        logger.info_once(
            "Reloading and processing weights took %.2f seconds",
            diff_seconds,
            scope="local",
        )
        if self.model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                logger.warning(
                    "Following weights were not loaded from checkpoint: %s",
                    weights_not_loaded,
                )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: torch.Tensor | None,
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None and req_index < logits.shape[0]
                    else 0
                )
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None
    ):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run
        """

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        elif input_ids is not None:

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids.gpu,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                )

            logger.debug_once("Randomizing dummy input_ids for DP Rank")
            input_ids.copy_(rand_input_ids()[: input_ids.size(0)], non_blocking=True)
            yield
            input_ids.fill_(0)
        else:

            @functools.cache
            def rand_inputs_embeds() -> torch.Tensor:
                return torch.randn_like(
                    self.inputs_embeds.gpu,
                )

            assert inputs_embeds is not None
            logger.debug_once("Randomizing dummy inputs_embeds for DP Rank")
            inputs_embeds.copy_(
                rand_inputs_embeds()[: inputs_embeds.size(0)], non_blocking=True
            )
            yield
            inputs_embeds.fill_(0)

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """Dummy data for profiling and precompiling multimodal models."""
        assert self.mm_budget is not None

        # Don't use `max_items_per_batch` here to avoid redundant computation
        dummy_mm_inputs = self.mm_registry.get_dummy_mm_inputs(
            self.model_config,
            mm_counts={modality: 1},
            cache=self.mm_budget.cache,
        )
        dummy_mm_item = dummy_mm_inputs["mm_kwargs"][modality][0]

        # We use the cache so that the item is saved to the cache,
        # but not read from the cache
        assert dummy_mm_item is not None, "Item should not already be cached"

        return next(
            mm_kwargs_group
            for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                [(modality, dummy_mm_item)] * max_items_per_batch,
                device=self.device,
                pin_memory=self.pin_memory,
            )
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            num_active_loras: Number of distinct active LoRAs to capture for.
                LoRA is activated when num_active_loras > 0.
        """
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.is_valid_runtime_mode()
        )

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.max_num_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        # Match worst-case packed-pivot verifier sample count for LoRA / capture warmup
        # (sum over P rows of at most 1 + num_spec speculative slots each).
        if (
            _uses_pivot_linear_fixed_capacity_packing(self.speculative_config)
            and num_reqs > 0
        ):
            spec = self.speculative_config
            assert spec is not None
            P = spec.pivot_packed_batch_size_for_origin_batch(num_reqs)
            if P > 0:
                sp1 = 1 + int(spec.num_speculative_tokens)
                target = P * sp1
                base, rem = divmod(target, num_reqs)
                num_sampled_tokens = np.full(num_reqs, base, dtype=np.int32)
                if rem:
                    num_sampled_tokens[:rem] += 1

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=num_active_loras > 0,
                # `force_num_active_loras` is used for cudagraph capture; because we
                # need to capture graphs for specific num_active_loras counts
                force_num_active_loras=num_active_loras,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=num_tokens,
            num_reqs_padded=num_reqs_padded,
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
        )

        # _dummy_run shares pinned CPU buffers (seq_lens, query_start_loc,
        # etc.) with execute_model.  It must participate in the same event
        # protocol so that back-to-back dummy/real steps don't overwrite
        # pinned memory while a prior non_blocking H2D DMA is still reading.
        with self.synchronize_input_prep():
            # If force_attention is True, we always capture attention.
            # Otherwise, it only happens for cudagraph_runtime_mode=FULL.
            if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
                if create_mixed_batch:
                    # In the mixed batch mode (used for FI warmup), we use
                    # shorter sequence lengths to run faster.
                    # TODO(luka) better system for describing dummy batches
                    seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
                else:
                    seq_lens = max_query_len  # type: ignore[assignment]
                self.seq_lens.np[:num_reqs] = seq_lens
                self.seq_lens.np[num_reqs:] = 0
                self.seq_lens.copy_to_gpu()

                cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
                self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
                self.query_start_loc.copy_to_gpu()

                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
                attn_build_dummy = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs_padded,
                    max_query_len=max_query_len,
                    ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),
                    for_cudagraph_capture=is_graph_capturing,
                    slot_mappings=slot_mappings_by_group,
                    use_spec_decode=self.speculative_config is not None,
                )
                attn_metadata = attn_build_dummy.attn_metadata

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            num_active_loras,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens_padded, None, False
                )

            if ubatch_slices_padded is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                    slot_mapping=slot_mappings,
                ),
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and (
                self.speculative_config.uses_model_based_drafter()
                or self.speculative_config.uses_extract_hidden_states()
            ):
                assert isinstance(
                    self.drafter,
                    EagleProposer
                    | DraftModelProposer
                    | AdaptiveSpechiveProposer
                    | HierarchicalVerificationProposer
                    | PivotProposer
                    | ExtractHiddenStatesProposer,
                )
                assert self.speculative_config is not None
                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager
                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if (
                    self.compilation_config.cudagraph_specialize_lora
                    and num_active_loras > 0
                ):
                    use_cudagraphs = False

                dr_kwargs: dict[str, Any] = dict(
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
                )
                if isinstance(self.drafter, PivotProposer):
                    dr_kwargs["origin_batch_size"] = num_reqs
                self.drafter.dummy_run(num_tokens, **dr_kwargs)

        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        self._register_layerwise_nvtx_hooks()

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )
        return hidden_states, hidden_states[logit_indices_device]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, we use a random tensor here instead.

        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model no need to run sampler.
            return torch.tensor([])

        hidden_states = torch.rand_like(hidden_states)

        logits = self.model.compute_logits(hidden_states)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            spec_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        try:
            sampler_output = self.sampler(
                logits=logits, sampling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device
            )

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            # draft_probs = torch.randn(
            #     num_tokens, logits.shape[-1], device=self.device,
            #     dtype=logits.dtype)
            draft_probs = None
            logits = torch.randn(
                num_tokens + num_reqs,
                logits.shape[-1],
                device=self.device,
                dtype=logits.dtype,
            )
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                logits,
                dummy_metadata,
            )
        return sampler_output

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_np = np.full(num_reqs, min_tokens_per_req)
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs
        assert np.sum(num_scheduled_tokens_np) == num_tokens
        assert len(num_scheduled_tokens_np) == num_reqs

        req_num_tokens = num_tokens // num_reqs

        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(self.model_config)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            pooling_params=[dummy_pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for i in range(num_reqs)],
        )

        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu=dummy_prompt_lens,
            device=hidden_states.device,
        )

        try:
            return model.pooler(
                hidden_states=hidden_states, pooling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> PoolerOutput:
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model not need to run pooler.
            return torch.tensor([])

        # Find the task that has the largest output for subsequent steps
        supported_pooling_tasks = self.get_supported_pooling_tasks()

        if not supported_pooling_tasks:
            raise RuntimeError(
                f"Model {self.model_config.model} does not support "
                "any pooling tasks. See "
                "https://docs.vllm.ai/en/latest/models/pooling_models.html "
                "to learn more."
            )

        output_size = dict[PoolingTask, float]()
        for task in supported_pooling_tasks:
            # Run a full batch with each task to ensure none of them OOMs
            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = sum(o.nbytes for o in output if o is not None)
            del output  # Allow GC

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        return self._dummy_pooler_run_task(hidden_states, max_task)

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        if self.supports_mm_inputs:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and mm_config.skip_mm_profiling:
                logger.info(
                    "Skipping memory profiling for multimodal encoder and "
                    "encoder cache."
                )
            else:
                mm_budget = self.mm_budget
                assert mm_budget is not None

                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                    if not mm_budget.mm_max_toks_per_item:
                        # All modality limits are 0 — embedding-only mode.
                        # Budget is non-zero for embedding storage, but
                        # there's no encoder to profile.
                        logger.info(
                            "Skipping encoder profiling for embedding-only "
                            "mode (all modality limits=0 with "
                            "enable_mm_embeds=True).",
                        )
                    else:
                        # NOTE: Currently model is profiled with a single
                        # non-text modality with the max possible input
                        # tokens even when it supports multiple.
                        dummy_modality = mm_budget.get_modality_with_max_tokens()
                        max_mm_items_per_batch = mm_budget.mm_max_items_per_batch[
                            dummy_modality
                        ]

                        logger.info(
                            "Encoder cache will be initialized with a "
                            "budget of %s tokens, and profiled with "
                            "%s %s items of the maximum feature size.",
                            encoder_budget,
                            max_mm_items_per_batch,
                            dummy_modality,
                        )

                        # Create dummy batch of multimodal inputs.
                        batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                            dummy_modality,
                            max_mm_items_per_batch,
                        )

                        # Run multimodal encoder.
                        dummy_encoder_outputs = self.model.embed_multimodal(
                            **batched_dummy_mm_inputs
                        )

                        sanity_check_mm_encoder_outputs(
                            dummy_encoder_outputs,
                            expected_num_items=max_mm_items_per_batch,
                        )
                        for i, output in enumerate(dummy_encoder_outputs):
                            self.encoder_cache[f"tmp_{i}"] = output

        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True
        )
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()

    @instrument(span_name="Capture model")
    def capture_model(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()

        @contextmanager
        def freeze_gc():
            # Optimize garbage collection during CUDA graph capture.
            # Clean up, then freeze all remaining objects from being included
            # in future collections.
            gc.collect()
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
            if should_freeze:
                gc.freeze()
            try:
                yield
            finally:
                if should_freeze:
                    gc.unfreeze()
                    gc.collect()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)
        with freeze_gc(), graph_capture(device=self.device):
            start_free_gpu_memory = torch.cuda.mem_get_info()[0]

            for (
                runtime_mode,
                batch_descs,
            ) in self.cudagraph_dispatcher.get_capture_descs():
                self._capture_cudagraphs(
                    batch_descriptors=batch_descs,
                    cudagraph_runtime_mode=runtime_mode,
                )

            torch.cuda.synchronize()
            end_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        # Lock workspace to prevent resizing during execution.
        # Max workspace sizes should have been captured during warmup/profiling.
        lock_workspace()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info_once(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
            scope="local",
        )
        return cuda_graph_size

    def _capture_cudagraphs(
        self,
        batch_descriptors: list[BatchDescriptor],
        cudagraph_runtime_mode: CUDAGraphMode,
    ):
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.is_valid_runtime_mode()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"

        if not batch_descriptors:
            return

        uniform_decode = batch_descriptors[0].uniform
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL

        dummy_run = functools.partial(
            self._dummy_run,
            uniform_decode=uniform_decode,
            skip_eplb=True,
            remove_lora=False,
            force_attention=force_attention,
        )

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            batch_descriptors = tqdm(
                batch_descriptors,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    cudagraph_runtime_mode.name,
                ),
            )

        # We skip EPLB here since we don't want to record dummy metrics
        for batch_desc in batch_descriptors:
            num_tokens = batch_desc.num_tokens
            num_active_loras = batch_desc.num_active_loras

            # We currently only capture ubatched graphs when its a FULL
            # cudagraph, a uniform decode batch, and the number of tokens
            # is above the threshold. Otherwise we just capture a non-ubatched
            # version of the graph
            allow_microbatching = (
                self.parallel_config.use_ubatching
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and uniform_decode
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=num_tokens,
                    uniform_decode=uniform_decode,
                )
            )

            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                # Use CUDAGraphRuntimeStyle.NONE (default) for warmup.
                # But be careful, warm up with `NONE` is orthogonal to
                # if we want to warm up attention or not. This is
                # different from the case where `FULL` implies capture
                # attention while `PIECEWISE` implies no attention.

                dummy_run(
                    num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    allow_microbatching=allow_microbatching,
                    num_active_loras=num_active_loras,
                )

            # Capture run
            dummy_run(
                num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                allow_microbatching=allow_microbatching,
                num_active_loras=num_active_loras,
                is_graph_capturing=True,
            )
        self.maybe_remove_all_loras(self.lora_config)

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layer_type = cast(type[Any], AttentionLayerBase)
            layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, kv_cache_group_spec.layer_names
            )
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",
                        attn_backend,  # type: ignore[arg-type]
                    )

                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(
                    attn_backend, layer_kv_cache_spec
                )
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
            kv_cache_group_id: int,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():
                attn_group = AttentionGroup(
                    attn_backend,
                    layer_names,
                    kv_cache_spec,
                    kv_cache_group_id,
                )

                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        # Resolve cudagraph_mode before actually initialize metadata_builders
        self._check_and_update_cudagraph_mode(
            attention_backend_list, kv_cache_config.kv_cache_groups
        )

        # Check if attention backend supports PCP&DCP and related features.
        check_attention_cp_compatibility(self.vllm_config)

        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))

    def initialize_metadata_builders(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Create the metadata builders for all KV cache groups and attn groups.
        """
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None,
                    num_metadata_builders=1
                    if not self.parallel_config.use_ubatching
                    else self.parallel_config.num_ubatches,
                )
        # Calculate reorder batch threshold (if needed)
        # Note (tdoublep): do this *after* constructing builders,
        # because some of them change the threshold at init time.
        self.calculate_reorder_batch_threshold()

        # Initialize drafter attention backend
        if self.speculative_config and self.speculative_config.uses_model_based_drafter():
            assert isinstance(
                self.drafter,
                EagleProposer
                | DraftModelProposer
                | AdaptiveSpechiveProposer
                | HierarchicalVerificationProposer
                | PivotProposer,
            )
            self.drafter.initialize_attn_backend(kv_cache_config, kernel_block_sizes)

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        """
        Resolve the cudagraph_mode when there are multiple attention
        groups with potential conflicting CUDA graph support.
        Then initialize the cudagraph_dispatcher based on the resolved
        cudagraph_mode.
        """
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_backend_name = None

        for attn_backend_set, kv_cache_group in zip(
            attention_backends, kv_cache_groups
        ):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()

                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_backend_name = attn_backend.__name__
        # Flexible resolve the cudagraph mode
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        # check cudagraph for mixed batch is supported
        if (
            cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
            and min_cg_support != AttentionCGSupport.ALWAYS
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if min_cg_support == AttentionCGSupport.NEVER:
                # if not supported any full cudagraphs, just raise it.
                msg += (
                    "; please try cudagraph_mode=PIECEWISE, and "
                    "make sure compilation mode is VLLM_COMPILE"
                )
                raise ValueError(msg)

            # attempt to resolve the full cudagraph related mode
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_AND_PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_DECODE_ONLY
                )
            logger.warning(msg)

        # check that if we are doing decode full-cudagraphs it is supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if self.compilation_config.mode == CompilationMode.VLLM_COMPILE and (
                self.compilation_config.splitting_ops_contain_attention()
                or self.compilation_config.use_inductor_graph_partition
            ):
                msg += (
                    "; setting cudagraph_mode=PIECEWISE because "
                    "attention is compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += (
                    "; setting cudagraph_mode=NONE because "
                    "attention is not compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # check that if we are doing spec-decode + decode full-cudagraphs it is
        # supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and self.uniform_decode_query_len > 1
            and min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported"
                f" with spec-decode for attention backend "
                f"{min_cg_backend_name} (support: {min_cg_support})"
            )
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=NONE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # double check that we can support full cudagraph if they are requested
        # even after automatic downgrades
        if (
            cudagraph_mode.has_full_cudagraphs()
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            raise ValueError(
                f"CUDAGraphMode.{cudagraph_mode.name} is not "
                f"supported with {min_cg_backend_name} backend ("
                f"support:{min_cg_support}) "
                "; please try cudagraph_mode=PIECEWISE, "
                "and make sure compilation mode is VLLM_COMPILE"
            )

        # if we have dedicated decode cudagraphs, and spec-decode is enabled,
        # we need to adjust the cudagraph sizes to be a multiple of the uniform
        # decode query length to avoid: https://github.com/vllm-project/vllm/issues/28207
        # temp-fix: https://github.com/vllm-project/vllm/issues/28207#issuecomment-3504004536
        # Will be removed in the near future when we have separate cudagraph capture
        # sizes for decode and mixed prefill-decode.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
            and self.uniform_decode_query_len > 1
        ):
            self.compilation_config.adjust_cudagraph_sizes_for_spec_decode(
                self.uniform_decode_query_len, self.parallel_config.tensor_parallel_size
            )

        # Trigger cudagraph dispatching keys initialization after
        # resolved cudagraph mode.
        self.compilation_config.cudagraph_mode = cudagraph_mode
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, self.uniform_decode_query_len
        )

        # Initialize drafter's cudagraph dispatcher if using spec decode.
        if self.speculative_config and (
            self.speculative_config.uses_model_based_drafter()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer
                | DraftModelProposer
                | AdaptiveSpechiveProposer
                | HierarchicalVerificationProposer
                | PivotProposer
                | ExtractHiddenStatesProposer,
            )
            # Linear fixed-capacity pivot uses origin-B rows for the root proposal and
            # packed-P rows for tail proposal; PivotProposer.dummy_run warms both.
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Choose the minimum reorder batch threshold from all attention groups.
        Backends should be able to support lower threshold then what they request
        just may have a performance penalty due to that backend treating decodes
        as prefills.
        """
        min_none_high = lambda a, b: a if b is None else b if a is None else min(a, b)

        reorder_batch_thresholds: list[int | None] = [
            group.get_metadata_builder().reorder_batch_threshold
            for group in self._attn_group_iterator()
        ]
        # If there are no attention groups (attention-free model) or no backend
        # reports a threshold, leave reordering disabled.
        if len(reorder_batch_thresholds) == 0:
            self.reorder_batch_threshold = None
            return
        self.reorder_batch_threshold = reduce(min_none_high, reorder_batch_thresholds)  # type: ignore[assignment]

    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """
        block_sizes = []
        max_num_blocks = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            block_size = kv_cache_group.kv_cache_spec.block_size
            block_sizes.append(block_size)
            max_num_blocks_per_req = cdiv(
                max_model_len, block_size * get_total_cp_world_size()
            )
            if isinstance(kv_cache_group.kv_cache_spec, MambaSpec):
                max_num_blocks_per_req = (
                    max_num_blocks_per_req
                    if self.cache_config.enable_prefix_caching
                    else 1
                ) + kv_cache_group.kv_cache_spec.num_speculative_blocks
            max_num_blocks.append(max_num_blocks_per_req)

        if block_sizes != [self.cache_config.block_size] or kernel_block_sizes != [
            self.cache_config.block_size
        ]:
            assert self.offload_config.uva.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                is_pooling_model=self.is_pooling_model,
            )
            if self._static_intermediate_kv_frontier_enabled(self.vllm_config):
                self.intermediate_input_batch = InputBatch(
                    max_num_reqs=self.max_num_reqs,
                    max_model_len=max_model_len,
                    max_num_batched_tokens=self.max_num_tokens,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    vocab_size=self.model_config.get_vocab_size(),
                    block_sizes=block_sizes,
                    kernel_block_sizes=kernel_block_sizes,
                    max_num_blocks_per_req=max_num_blocks,
                    is_spec_decode=bool(self.vllm_config.speculative_config),
                    logitsprocs=self.input_batch.logitsprocs,
                    logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                    is_pooling_model=self.is_pooling_model,
                    cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
                )
                self.intermediate_requests.clear()
                self.intermediate_committed_tokens.clear()

    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(
                kv_cache_tensor.size, dtype=torch.int8, device=self.device
            )
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            if group.kv_cache_group_id == len(kernel_block_sizes):
                # There may be a last group for layers without kv cache.
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,
                        kernel_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=self.cache_config.cache_dtype,
                    )
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(
                        kv_cache_shape[i] for i in kv_cache_stride_order
                    )
                    # Maintain original KV shape view.
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = (
                        kv_cache_raw_tensors[layer_name]
                        .view(dtype)
                        .view(kv_cache_shape)
                        .permute(*inv_order)
                    )
                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    state_tensors = []
                    storage_offset_bytes = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size
                        )
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size

                    kv_caches[layer_name] = state_tensors
                else:
                    raise NotImplementedError

        if has_attn and has_mamba:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    def _update_hybrid_attention_mamba_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """
        Update the layout of attention layers from (2, num_blocks, ...) to
        (num_blocks, 2, ...).

        Args:
            kv_caches: The KV cache buffer of each layer.
        """

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                if isinstance(kv_cache_spec, AttentionSpec) and kv_cache.shape[0] == 2:
                    assert kv_cache.shape[1] != 2, (
                        "Fail to determine whether the layout is "
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
                        f"a tensor of shape {kv_cache.shape}"
                    )
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(
                        size=kv_cache.shape,
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                    )

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
            kernel_block_sizes: The kernel block sizes for each KV cache group.

        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """

        # Try creating KV caches optimized for kv-connector transfers
        cache_dtype = self.cache_config.cache_dtype
        if self.use_uniform_kv_cache(self.attn_groups, cache_dtype):
            kv_caches, cross_layers_kv_cache, attn_backend = (
                self.allocate_uniform_kv_caches(
                    kv_cache_config,
                    self.attn_groups,
                    cache_dtype,
                    self.device,
                    kernel_block_sizes,
                )
            )
            self.cross_layers_kv_cache = cross_layers_kv_cache
            self.cross_layers_attn_backend = attn_backend
        else:
            # Fallback to the general case
            # Initialize the memory buffer for KV cache
            kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

            # Change the memory buffer to the desired shape
            kv_caches = self._reshape_kv_cache_tensors(
                kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes
            )

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        return kv_caches

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        if not self.shared_kv_cache_layers:
            # No cross-layer KV sharing, return
            return

        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            self.runner_only_attn_layers,
        )

        if self.cache_config.kv_sharing_fast_prefill:
            # In You Only Cache Once (https://arxiv.org/abs/2405.05254) or other
            # similar KV sharing setups, only the layers that generate KV caches
            # are involved in the prefill phase, enabling prefill to early exit.
            attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(layer_name)
                else:
                    break

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self._mamba_copy_bufs = None
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group, but the attention
        # backends for that group only supports block_size 64, we will return
        # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
        # tokens each.
        kernel_block_sizes = prepare_kernel_block_sizes(
            kv_cache_config, self.attn_groups
        )
        self._kernel_block_sizes = kernel_block_sizes

        # create metadata builders
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # Reinitialize need to after initialize_attn_backend
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        kv_caches = self.initialize_kv_cache_tensors(
            kv_cache_config, kernel_block_sizes
        )

        if (
            self.speculative_config
            and self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            kv_transfer_group = get_kv_transfer_group()
            if self.cross_layers_kv_cache is not None:
                assert self.cross_layers_attn_backend is not None
                kv_transfer_group.register_cross_layers_kv_cache(
                    self.cross_layers_kv_cache, self.cross_layers_attn_backend
                )
            else:
                kv_transfer_group.register_kv_caches(kv_caches)
            kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    def init_routed_experts_capturer(self):
        logger.info(
            "Initializing routed experts capturer, enable_return_routed_experts: %s",
            self.model_config.enable_return_routed_experts,
        )
        routed_experts_capturer = RoutedExpertsCapturer.create()
        block_size = self.cache_config.block_size
        self.max_num_kv_tokens = (
            self.kv_cache_config.num_blocks // len(self.kv_cache_config.kv_cache_groups)
            + 1
        ) * block_size
        routed_experts_capturer.init_buffer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_kv_tokens=self.max_num_kv_tokens,
            vllm_config=self.vllm_config,
        )
        self._bind_routed_experts_capturer(routed_experts_capturer)

    def _bind_routed_experts_capturer(self, capturer: RoutedExpertsCapturer) -> None:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            BaseRouter,
        )

        for module in self.compilation_config.static_forward_context.values():
            if isinstance(module, FusedMoE) and isinstance(module.router, BaseRouter):
                layer_id = module.layer_id

                def _capture_fn(topk_ids, _layer_id=layer_id, _capturer=capturer):
                    _capturer.capture(_layer_id, topk_ids)

                module.router.set_capture_fn(_capture_fn)

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """
        Add encoder-only layers to the KV cache config.
        """
        block_size = self.vllm_config.cache_config.block_size
        encoder_only_attn_specs: dict[AttentionSpec, list[str]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                )
                encoder_only_attn_specs[attn_spec].append(layer_name)
                self.runner_only_attn_layers.add(layer_name)
        if len(encoder_only_attn_specs) > 0:
            assert len(encoder_only_attn_specs) == 1, (
                "Only support one encoder-only attention spec now"
            )
            spec, layer_names = encoder_only_attn_specs.popitem()
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
            )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        if has_ec_transfer() and get_ec_transfer().is_producer:
            return {}
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention) and (
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name
            ):
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec

        return kv_cache_spec

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """
        Get encoder timing stats for all requests and clear the registry.

        Returns:
            Dictionary mapping request_id to stats dict.
        """
        with self._encoder_timing_lock:
            stats = {
                req_id: stats_obj.to_dict()
                for req_id, stats_obj in self.encoder_timing_registry.items()
            }
            self.encoder_timing_registry.clear()
            return stats

    @contextmanager
    def timed_encoder_operation(
        self,
        should_time: bool,
        group_lora_refs: list[tuple[str, Any]],
        current_item_idx: int,
        num_items: int,
    ):
        """
        Context manager to time encoder forward operations.

        Args:
            should_time: Whether timing is enabled
            group_lora_refs: Full list of (request_id, pos_info) tuples
            current_item_idx: Starting index for this group
            num_items: Number of items in this group
        """
        if not should_time:
            yield
            return

        group_refs = group_lora_refs[current_item_idx : current_item_idx + num_items]
        group_request_ids = {req_id for req_id, _ in group_refs}

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            per_request_time = elapsed / max(len(group_request_ids), 1)

            with self._encoder_timing_lock:
                for req_id in group_request_ids:
                    if req_id not in self.encoder_timing_registry:
                        self.encoder_timing_registry[req_id] = EncoderTimingStats()

                    stats = self.encoder_timing_registry[req_id]
                    stats.encoder_forward_secs += per_request_time
                    stats.num_encoder_calls += 1


@dataclass
class EncoderTimingStats:
    """Per-request timing statistics for encoder forward pass."""

    encoder_forward_secs: float = 0.0
    """Time spent in vision encoder forward pass (seconds)."""

    num_encoder_calls: int = 0
    """Number of times encoder was called for this request."""

    def to_dict(self) -> dict[str, float | int]:
        return {
            "encoder_forward_secs": self.encoder_forward_secs,
            "num_encoder_calls": self.num_encoder_calls,
        }
