# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache compression for hierarchical speculative verification.

When compress_method is "random", we randomly sample a fraction (compression_ratio)
of KV positions (same random token indices for all layers and heads) to use for
partial verification.
"""

from dataclasses import dataclass

import numpy as np
import random
import torch

from vllm.config.speculative import HierarchicalVerificationCompressMethod
from vllm.utils.math_utils import cdiv


def random_compression_mask(
    num_positions: int,
    compression_ratio: float,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
    seed: int | None = None,
) -> torch.Tensor:
    """Build a boolean mask that keeps a random subset of KV positions.

    mask[i] is True if position i is kept (used in partial verification).
    Exactly floor(num_positions * compression_ratio) positions are True.

    Args:
        num_positions: Total number of KV positions (e.g. sequence length).
        compression_ratio: Fraction of positions to keep, in (0, 1].
        device: Device for the output tensor.
        dtype: Output dtype (bool or uint8 for compatibility).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Boolean tensor of shape (num_positions,) with compression_ratio
        fraction of True.
    """
    if num_positions <= 0:
        return torch.zeros(num_positions, dtype=dtype, device=device)
    n_keep = max(1, int(num_positions * compression_ratio))
    n_keep = min(n_keep, num_positions)
    rng = random.Random(seed)
    indices = set(rng.sample(range(num_positions), n_keep))
    mask = torch.tensor(
        [i in indices for i in range(num_positions)],
        dtype=dtype,
        device=device,
    )
    return mask


def random_compression_indices(
    num_positions: int,
    compression_ratio: float,
    device: torch.device,
    seed: int | None = None,
) -> torch.Tensor:
    """Return indices of positions to keep for compressed KV (random sampling).

    Args:
        num_positions: Total number of KV positions.
        compression_ratio: Fraction to keep in (0, 1].
        device: Device for the output tensor.
        seed: Optional RNG seed.

    Returns:
        Long tensor of shape (n_keep,) with indices in [0, num_positions).
    """
    if num_positions <= 0:
        return torch.zeros(0, dtype=torch.int64, device=device)
    n_keep = max(1, int(num_positions * compression_ratio))
    n_keep = min(n_keep, num_positions)
    rng = random.Random(seed)
    indices = rng.sample(range(num_positions), n_keep)
    return torch.tensor(sorted(indices), dtype=torch.int64, device=device)


def build_compression_mask(
    num_positions: int,
    compression_ratio: float,
    compress_method: HierarchicalVerificationCompressMethod,
    device: torch.device,
    seed: int | None = None,
) -> torch.Tensor | None:
    """Build a compression mask for partial verification.

    Returns None if compression_ratio >= 1 (no compression).
    """
    if compression_ratio >= 1.0 or num_positions <= 0:
        return None
    if compress_method == "random":
        return random_compression_mask(
            num_positions, compression_ratio, device, seed=seed
        )
    raise ValueError(f"Unknown compress_method: {compress_method!r}")


def get_compression_indices_batch(
    num_tokens: int,
    compression_ratio: float,
    compress_method: HierarchicalVerificationCompressMethod,
    device: torch.device,
    seed: int | None = None,
) -> torch.Tensor | None:
    """Return batch token indices to keep for compressed KV (same for all layers).

    Args:
        num_tokens: Total number of tokens in the batch.
        compression_ratio: Fraction to keep in (0, 1].
        compress_method: Compression method (e.g. "random").
        device: Device for the output tensor.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Long tensor of shape (n_keep,) with batch indices in [0, num_tokens),
        sorted in ascending order. None if no compression (ratio >= 1 or
        num_tokens <= 0).
    """
    if compression_ratio >= 1.0 or num_tokens <= 0:
        return None
    if compress_method == "random":
        return random_compression_indices(
            num_tokens, compression_ratio, device, seed=seed
        )
    raise ValueError(f"Unknown compress_method: {compress_method!r}")


@dataclass
class CompressedKVMetadata:
    """Metadata for reading from a compressed KV cache in partial verification."""

    # Per-request compressed context length (number of kept positions).
    seq_lens: np.ndarray  # shape (num_reqs,), int
    # Block table for the compressed buffer: [num_reqs, max_blocks_per_req],
    # physical block indices in the compressed buffer.
    block_table: np.ndarray
    block_size: int
    num_compressed_slots: int


def _copy_full_to_compressed_single_layer(
    full_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compression_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Copy selected slots from full KV cache to a new compressed buffer (one layer).

    Uses the same random token indices (compression_indices) for this layer.
    Assumes cache layout has (num_blocks, block_size, ...) or
    (2, num_blocks, block_size, ...) so that slot s is at [..., s//block_size,
    s%block_size, ...].

    Returns:
        Compressed cache tensor with shape (..., n_blocks_comp, block_size, ...)
        where n_blocks_comp = ceil(len(compression_indices) / block_size).
    """
    n_keep = compression_indices.shape[0]
    if n_keep == 0:
        return full_cache.new_empty(0)

    # Infer block dim: first two dims are often (num_blocks, block_size) or
    # (2, num_blocks, block_size).
    ndim = full_cache.ndim
    if ndim >= 2 and full_cache.shape[0] == 2:
        block_dim = 1
        block_size_dim = 2
    else:
        block_dim = 0
        block_size_dim = 1

    total_slots = full_cache.shape[block_dim] * full_cache.shape[block_size_dim]
    full_slots = slot_mapping[compression_indices]  # (n_keep,) physical slots

    # Clamp to valid range (e.g. -1 padding)
    full_slots = full_slots.clamp(0, total_slots - 1)

    # Flatten slot dimension for indexing
    shape_after = full_cache.shape[block_size_dim + 1 :]
    cache_flat = full_cache.reshape(
        *full_cache.shape[:block_dim], total_slots, *shape_after
    )
    slot_dim = block_dim
    gathered = torch.index_select(
        cache_flat, slot_dim, full_slots
    )  # (..., n_keep, ...)

    n_blocks_comp = cdiv(n_keep, block_size)
    pad_length = n_blocks_comp * block_size - n_keep
    if pad_length > 0:
        pad_shape = list(gathered.shape)
        pad_shape[slot_dim] = pad_length
        padding = gathered.new_zeros(pad_shape, dtype=gathered.dtype)
        gathered_padded = torch.cat([gathered, padding], dim=slot_dim)
    else:
        gathered_padded = gathered

    comp_shape = (
        *gathered.shape[:slot_dim],
        n_blocks_comp,
        block_size,
        *gathered.shape[slot_dim + 1 :],
    )
    return gathered_padded.reshape(comp_shape)


def build_compressed_kv_caches(
    full_kv_caches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    compression_indices: torch.Tensor,
    block_size: int,
) -> list[torch.Tensor]:
    """Build compressed KV cache tensors from full caches (same indices for all layers).

    Args:
        full_kv_caches: One tensor per layer (e.g. from runner kv_caches).
        slot_mapping: [num_tokens] physical slot per batch token.
        compression_indices: [n_keep] batch token indices to keep (sorted).
        block_size: KV cache block size.

    Returns:
        List of compressed cache tensors, one per layer, with the same layout as
        full caches but fewer blocks.
    """
    return [
        _copy_full_to_compressed_single_layer(
            full_cache, slot_mapping, compression_indices, block_size
        )
        for full_cache in full_kv_caches
    ]


def build_compressed_kv_metadata(
    compression_indices: torch.Tensor,
    query_start_loc: np.ndarray,
    num_reqs: int,
    block_size: int,
) -> CompressedKVMetadata:
    """Build block_table and seq_lens for the compressed KV buffer.

    The compressed buffer has |I| slots (I = compression_indices). Slots are
    assigned to requests in order: request 0 gets the first seq_lens[0] slots,
    request 1 the next seq_lens[1], etc.

    Args:
        compression_indices: [n_keep] batch token indices kept (sorted).
        query_start_loc: [num_reqs+1] cumulative token counts per request.
        num_reqs: Number of requests.
        block_size: Block size for the KV cache.

    Returns:
        CompressedKVMetadata with seq_lens, block_table, and num_compressed_slots.
    """
    n_keep = compression_indices.shape[0]
    if n_keep == 0:
        return CompressedKVMetadata(
            seq_lens=np.zeros(num_reqs, dtype=np.int32),
            block_table=np.zeros((num_reqs, 0), dtype=np.int32),
            block_size=block_size,
            num_compressed_slots=0,
        )

    comp_indices_np = compression_indices.cpu().numpy()
    query_start_loc = np.asarray(query_start_loc, dtype=np.int64)
    seq_lens = np.zeros(num_reqs, dtype=np.int32)
    for r in range(num_reqs):
        start = query_start_loc[r]
        end = query_start_loc[r + 1]
        # Count how many of comp_indices_np fall in [start, end)
        seq_lens[r] = int(np.sum((comp_indices_np >= start) & (comp_indices_np < end)))

    num_blocks_per_req = cdiv(seq_lens, block_size)
    max_blocks = int(np.max(num_blocks_per_req))
    block_table = np.zeros((num_reqs, max_blocks), dtype=np.int32)
    block_base = 0
    for r in range(num_reqs):
        n_blocks = num_blocks_per_req[r]
        block_table[r, :n_blocks] = block_base + np.arange(n_blocks, dtype=np.int32)
        block_base += n_blocks

    return CompressedKVMetadata(
        seq_lens=seq_lens,
        block_table=block_table,
        block_size=block_size,
        num_compressed_slots=n_keep,
    )


# ---------------- Block-level compression (no KV memory copy) ----------------

def random_block_indices_for_req(
    num_blocks: int,
    compression_ratio: float,
    rng: random.Random,
) -> list[int]:
    """Select a random subset of block indices for a single request."""
    if num_blocks <= 0:
        return []
    n_keep = max(1, int(num_blocks * compression_ratio))
    n_keep = min(n_keep, num_blocks)
    return sorted(rng.sample(range(num_blocks), n_keep))


def build_block_level_compression_view(
    block_table_np: np.ndarray,
    num_blocks_per_row: np.ndarray,
    compression_ratio: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a block-level compressed view (no KV copy).

    Args:
        block_table_np: [num_reqs, max_num_blocks] block ids.
        num_blocks_per_row: [num_reqs] number of valid blocks per request.
        compression_ratio: fraction of blocks to keep per request.
        seed: RNG seed.

    Returns:
        compressed_block_table: [num_reqs, max_kept_blocks] with -1 padding.
        compressed_blocks_per_row: [num_reqs] kept block counts per request.
    """
    rng = random.Random(seed)
    num_reqs, _ = block_table_np.shape
    compressed: list[list[int]] = []
    compressed_blocks_per_row: list[int] = []
    max_kept = 0

    for r in range(num_reqs):
        nb = int(num_blocks_per_row[r])
        if nb == 0 or compression_ratio >= 1.0:
            kept: list[int] = block_table_np[r, :nb].tolist()
        else:
            kept_ids = random_block_indices_for_req(nb, compression_ratio, rng)
            kept = block_table_np[r, kept_ids].tolist()
        compressed.append(kept)
        compressed_blocks_per_row.append(len(kept))
        max_kept = max(max_kept, len(kept))

    if max_kept == 0:
        return (
            np.full((num_reqs, 0), -1, dtype=np.int32),
            np.zeros(num_reqs, dtype=np.int32),
        )

    out = np.full((num_reqs, max_kept), -1, dtype=np.int32)
    for r, kept in enumerate(compressed):
        if kept:
            out[r, : len(kept)] = np.asarray(kept, dtype=np.int32)

    return out, np.asarray(compressed_blocks_per_row, dtype=np.int32)

