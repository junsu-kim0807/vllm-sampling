# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache compression for hierarchical speculative verification.

When compress_method is "random", we randomly sample a fraction (compression_ratio)
of KV positions to use for partial verification.
"""

import random
import torch

from vllm.config.speculative import HierarchicalVerificationCompressMethod


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
