# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Golden-style checks for HV metadata-direct verify (packing + logit slicing)."""

from types import SimpleNamespace

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.adaptive_cascade import _build_prefix_conditioned_inputs
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.hv_step_packing import (
    build_prefix_conditioned_inputs,
    gather_hv_verification_logits_from_spec_decode_metadata,
    slice_hv_verification_logits,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def _tiny_cad(*, device: str = "cpu") -> CommonAttentionMetadata:
    """Two requests, two query tokens each; block table large enough for extensions."""
    bs = 2
    tok_per = 2
    num_tok = bs * tok_per
    qsl = torch.tensor([0, tok_per, 2 * tok_per], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([10, 12], dtype=torch.int32, device=device)
    block_table = torch.zeros((bs, 32), dtype=torch.int32, device=device)
    block_table[:, 0] = 1
    slot_mapping = torch.arange(num_tok, dtype=torch.int64, device=device)
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.cpu(),
        seq_lens=seq_lens,
        num_reqs=bs,
        num_actual_tokens=num_tok,
        max_query_len=tok_per,
        max_seq_len=int(seq_lens.max().item()),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
    )


def _fake_proposer(block_size: int) -> SpecDecodeBaseProposer:
    kv = SimpleNamespace(block_size=block_size)
    spec = SimpleNamespace(kv_cache_spec=kv)
    grp = SimpleNamespace(kv_cache_spec=kv)
    return SimpleNamespace(draft_attn_groups=[grp])  # type: ignore[return-value]


def test_hv_packing_matches_adaptive_wrapper() -> None:
    """``_build_prefix_conditioned_inputs`` must delegate to the same pure helper."""
    device = "cpu"
    cad = _tiny_cad(device=device)
    H = 16
    target_token_ids = torch.randint(0, 1000, (cad.num_actual_tokens,), device=device)
    target_positions = torch.arange(cad.num_actual_tokens, device=device, dtype=torch.long)
    target_hidden_states = torch.randn(
        cad.num_actual_tokens, H, dtype=torch.float32, device=device
    )
    next_token_ids = torch.tensor([3, 4], dtype=torch.int32, device=device)
    prefix_rows = [[7], [8]]
    roll_rows = [[30, 31], [40, 41]]
    proposer = _fake_proposer(block_size=4)
    a = _build_prefix_conditioned_inputs(
        proposer,  # type: ignore[arg-type]
        cad=cad,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        prefix_rows=prefix_rows,
        roll_rows=roll_rows,
    )
    b = build_prefix_conditioned_inputs(
        cad=cad,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        prefix_rows=prefix_rows,
        roll_rows=roll_rows,
        block_size=4,
    )
    for i in range(4):
        assert torch.equal(a[i], b[i])
    ca, cb = a[4], b[4]
    assert torch.equal(ca.query_start_loc, cb.query_start_loc)
    assert torch.equal(ca.query_start_loc_cpu, cb.query_start_loc_cpu)
    assert torch.equal(ca.seq_lens, cb.seq_lens)
    assert torch.equal(ca.slot_mapping, cb.slot_mapping)
    assert ca.num_actual_tokens == cb.num_actual_tokens
    assert ca.max_query_len == cb.max_query_len
    assert a[5:] == b[5:]


def test_slice_hv_verification_logits_matches_stacked_reference() -> None:
    """Sanity: stacked per-row slices match independent row-wise construction."""
    device = "cpu"
    cad = _tiny_cad(device=device)
    draft_tokens = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32, device=device)
    bsz, chunk_len = draft_tokens.shape
    base_query_lens = [2, 2]
    prefix_lens = [1, 1]
    roll_lens = [2, 2]
    vocab = 11
    total_rows = int(cad.query_start_loc[-1].item()) + bsz * 3
    all_logits = torch.randn(total_rows, vocab, dtype=torch.float32, device=device)
    flat, bonus = slice_hv_verification_logits(
        all_logits,
        cad,
        draft_tokens,
        base_query_lens,
        prefix_lens,
        roll_lens,
    )
    assert flat.shape == (bsz * chunk_len, vocab)
    assert bonus.shape == (bsz, vocab)
    qsl = cad.query_start_loc
    for b in range(bsz):
        chain_len = prefix_lens[b] + roll_lens[b] + 1
        start = int(qsl[b].item()) + base_query_lens[b] - 1
        idx = torch.arange(start, start + chain_len, device=device)
        chain = all_logits[idx]
        v0 = chain[prefix_lens[b] : prefix_lens[b] + chunk_len]
        assert torch.equal(v0, flat.view(bsz, chunk_len, -1)[b])
        assert torch.equal(chain[prefix_lens[b] + chunk_len], bonus[b])


def test_legacy_hv_slice_matches_spec_decode_gather_indices() -> None:
    """When target/bonus row indices match legacy slice geometry, gather must agree."""
    device = "cpu"
    cad = _tiny_cad(device=device)
    draft_tokens = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32, device=device)
    bsz, chunk_len = draft_tokens.shape
    base_query_lens = [2, 2]
    prefix_lens = [1, 1]
    roll_lens = [2, 2]
    vocab = 11
    total_rows = int(cad.query_start_loc[-1].item()) + bsz * 3
    all_logits = torch.randn(total_rows, vocab, dtype=torch.float32, device=device)
    legacy_flat, legacy_bonus = slice_hv_verification_logits(
        all_logits,
        cad,
        draft_tokens,
        base_query_lens,
        prefix_lens,
        roll_lens,
    )
    qsl = cad.query_start_loc
    target_rows: list[int] = []
    bonus_rows: list[int] = []
    for b in range(bsz):
        chain_len = prefix_lens[b] + roll_lens[b] + 1
        start = int(qsl[b].item()) + base_query_lens[b] - 1
        for j in range(chunk_len):
            target_rows.append(start + prefix_lens[b] + j)
        bonus_rows.append(start + prefix_lens[b] + chunk_len)
    meta = SpecDecodeMetadata(
        draft_token_ids=torch.zeros(bsz * chunk_len, dtype=torch.int32, device=device),
        num_draft_tokens=[chunk_len] * bsz,
        cu_num_draft_tokens=torch.tensor(
            [chunk_len, chunk_len * 2], dtype=torch.int32, device=device
        ),
        cu_num_sampled_tokens=torch.tensor(
            [chunk_len + 1, (chunk_len + 1) * 2], dtype=torch.int32, device=device
        ),
        target_logits_indices=torch.tensor(target_rows, dtype=torch.int32, device=device),
        bonus_logits_indices=torch.tensor(bonus_rows, dtype=torch.int32, device=device),
        logits_indices=torch.zeros(bsz * (chunk_len + 1), dtype=torch.int32, device=device),
        expansion_plan=None,
    )
    g_flat, g_bonus = gather_hv_verification_logits_from_spec_decode_metadata(
        all_logits, meta
    )
    assert torch.allclose(legacy_flat, g_flat)
    assert torch.allclose(legacy_bonus, g_bonus)


def test_static_intermediate_kv_frontier_enabled_for_standalone_hv() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    spec = SimpleNamespace(
        method="hierarchical_verification",
        intermediate_model="x",
        intermediate_kv_mode="mirror_frontier",
        adaptive_spechive_mode="draft_target",
        pivot_spechive=False,
    )
    vc = SimpleNamespace(speculative_config=spec)
    assert GPUModelRunner._static_intermediate_kv_frontier_enabled(vc)  # type: ignore[arg-type]