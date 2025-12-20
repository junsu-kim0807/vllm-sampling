# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Dict, Iterable

import torch

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class StarKVTierStore:
    """A simple two-tier KV store that mirrors blocks from the active (sub) KV cache.

    - sub tier: the vLLM KV cache tensors used by attention (GPU)
    - super tier: a mirrored buffer per layer (GPU if starkv_offload=False,
      CPU pinned if starkv_offload=True)

    This store does not manage block ownership; it only copies data by global
    block IDs along the KV-cache block dimension.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        super_buffers: Dict[str, torch.Tensor],
        sub_kv_caches: Dict[str, torch.Tensor],
    ) -> None:
        self._kv_cache_config = kv_cache_config
        self._super_buffers = super_buffers
        self._sub_kv_caches = sub_kv_caches
        self._group_to_layers: dict[int, list[str]] = {
            idx: list(group.layer_names)
            for idx, group in enumerate(kv_cache_config.kv_cache_groups)
        }

    def _iter_layer_pairs(
        self, kv_group_id: int
    ) -> Iterable[tuple[str, tuple[torch.Tensor, torch.Tensor]]]:
        layer_names = self._group_to_layers.get(kv_group_id, [])
        for layer_name in layer_names:
            sub_tensor = self._sub_kv_caches.get(layer_name)
            super_tensor = self._super_buffers.get(layer_name)
            if sub_tensor is None or super_tensor is None:
                continue
            yield layer_name, (sub_tensor, super_tensor)

    @staticmethod
    def _get_block_dim(tensor: torch.Tensor) -> int:
        # KV cache tensors can be shaped either:
        # - (num_blocks, ...)       -> blocks along dim 0
        # - (2, num_blocks, ...)    -> blocks along dim 1 (K/V in dim 0)
        if tensor.dim() >= 2 and tensor.shape[0] == 2 and tensor.shape[1] != 2:
            return 1
        return 0

    @staticmethod
    def _filter_valid_indices(idx_cpu: torch.Tensor, dim_size: int) -> torch.Tensor:
        return idx_cpu[(idx_cpu >= 0) & (idx_cpu < dim_size)]

    def mirror_blocks(
        self,
        kv_group_id: int,
        block_ids: list[int],
    ) -> None:
        """Copy blocks from sub (active) KV caches into super tier buffers."""
        if not block_ids:
            return
        idx_cpu = torch.tensor(block_ids, dtype=torch.long, device="cpu")
        for layer_name, (sub_tensor, super_tensor) in self._iter_layer_pairs(kv_group_id):
            try:
                block_dim = self._get_block_dim(sub_tensor)
                dim_size = sub_tensor.size(block_dim)
                if dim_size <= 0:
                    continue
                idx_valid_cpu = self._filter_valid_indices(idx_cpu, dim_size)
                if idx_valid_cpu.numel() == 0:
                    continue

                if super_tensor.device.type == "cpu":
                    idx_super = idx_valid_cpu
                    idx_sub = idx_valid_cpu.to(sub_tensor.device)
                    data = torch.index_select(sub_tensor, block_dim, idx_sub).to("cpu")
                    super_tensor.index_copy_(block_dim, idx_super, data)
                else:
                    idx_sub = idx_valid_cpu.to(sub_tensor.device)
                    idx_super = idx_valid_cpu.to(super_tensor.device)
                    data = torch.index_select(sub_tensor, block_dim, idx_sub)
                    super_tensor.index_copy_(block_dim, idx_super, data)
            except Exception:
                logger.exception(
                    "StarKV super-tier mirror failed for layer %s (group %d)",
                    layer_name,
                    kv_group_id,
                )

    def restore_blocks(
        self,
        kv_group_id: int,
        block_ids: list[int],
    ) -> None:
        """Copy blocks from super tier buffers back into sub (active) KV caches."""
        if not block_ids:
            return
        idx_cpu = torch.tensor(block_ids, dtype=torch.long, device="cpu")
        for layer_name, (sub_tensor, super_tensor) in self._iter_layer_pairs(kv_group_id):
            try:
                block_dim = self._get_block_dim(sub_tensor)
                dim_size = sub_tensor.size(block_dim)
                if dim_size <= 0:
                    continue
                idx_valid_cpu = self._filter_valid_indices(idx_cpu, dim_size)
                if idx_valid_cpu.numel() == 0:
                    continue

                if super_tensor.device.type == "cpu":
                    data = torch.index_select(super_tensor, block_dim, idx_valid_cpu).to(
                        sub_tensor.device
                    )
                    idx_sub = idx_valid_cpu.to(sub_tensor.device)
                    sub_tensor.index_copy_(block_dim, idx_sub, data)
                else:
                    idx_super = idx_valid_cpu.to(super_tensor.device)
                    data = torch.index_select(super_tensor, block_dim, idx_super).to(
                        sub_tensor.device
                    )
                    idx_sub = idx_valid_cpu.to(sub_tensor.device)
                    sub_tensor.index_copy_(block_dim, idx_sub, data)
            except Exception:
                logger.exception(
                    "StarKV super-tier restore failed for layer %s (group %d)",
                    layer_name,
                    kv_group_id,
                )


