# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import torch

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


@dataclass
class StarKVOffloadHandle:
    """Serializable metadata describing offloaded StarKV blocks."""

    request_id: str
    kv_group_id: int
    block_ids: list[int]
    handle_id: int


@dataclass
class _StoredEntry:
    request_id: str
    kv_group_id: int
    block_ids: list[int]
    valid: bool = field(default=True)


class StarKVOffloadStore:
    """Worker-local store that mirrors KV blocks into host memory."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        host_buffers: Dict[str, torch.Tensor],
        gpu_kv_caches: Dict[str, torch.Tensor],
    ) -> None:
        self._kv_cache_config = kv_cache_config
        self._host_buffers = host_buffers
        self._layer_kv_caches = gpu_kv_caches
        self._entries: dict[int, _StoredEntry] = {}
        self._next_handle_id = 1
        self._group_to_layers: dict[int, list[str]] = {
            idx: list(group.layer_names)
            for idx, group in enumerate(kv_cache_config.kv_cache_groups)
        }

    def _iter_layer_pairs(
        self, kv_group_id: int
    ) -> Iterable[tuple[str, tuple[torch.Tensor, torch.Tensor]]]:
        layer_names = self._group_to_layers.get(kv_group_id, [])
        for layer_name in layer_names:
            gpu_tensor = self._layer_kv_caches.get(layer_name)
            host_tensor = self._host_buffers.get(layer_name)
            if gpu_tensor is None or host_tensor is None:
                continue
            yield layer_name, (gpu_tensor, host_tensor)

    def offload_blocks(
        self,
        request_id: str,
        kv_group_id: int,
        block_ids: list[int],
    ) -> StarKVOffloadHandle | None:
        if not block_ids:
            return None

        idx_cpu = torch.tensor(block_ids, dtype=torch.long, device="cpu")
        for layer_name, (gpu_tensor, host_tensor) in self._iter_layer_pairs(
            kv_group_id
        ):
            try:
                # KV cache tensors can be shaped either:
                # - (num_blocks, ...)  -> blocks along dim 0
                # - (2, num_blocks, ...) -> blocks along dim 1 (K/V in dim 0)
                # We must offload/restore along the block dimension.
                block_dim = 0
                if gpu_tensor.dim() >= 2 and gpu_tensor.shape[0] == 2 and gpu_tensor.shape[1] != 2:
                    block_dim = 1

                dim_size = gpu_tensor.size(block_dim)
                if dim_size <= 0:
                    continue

                # Filter invalid indices to avoid CUDA device-side asserts.
                idx_valid_cpu = idx_cpu[(idx_cpu >= 0) & (idx_cpu < dim_size)]
                if idx_valid_cpu.numel() == 0:
                    logger.debug(
                        "StarKV offload: all indices out of range for layer %s "
                        "(group %d, block_dim=%d, dim_size=%d). Skipping.",
                        layer_name,
                        kv_group_id,
                        block_dim,
                        dim_size,
                    )
                    continue

                idx_gpu = idx_valid_cpu.to(gpu_tensor.device)
                data = torch.index_select(gpu_tensor, block_dim, idx_gpu).to("cpu")
                host_tensor.index_copy_(block_dim, idx_valid_cpu, data)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "StarKV offload failed for layer %s (group %d)", layer_name, kv_group_id
                )
                return None

        handle_id = self._next_handle_id
        self._next_handle_id += 1
        entry = _StoredEntry(
            request_id=request_id, kv_group_id=kv_group_id, block_ids=list(block_ids)
        )
        self._entries[handle_id] = entry
        return StarKVOffloadHandle(
            request_id=request_id,
            kv_group_id=kv_group_id,
            block_ids=list(block_ids),
            handle_id=handle_id,
        )

    def has_handle(self, handle_id: int) -> bool:
        entry = self._entries.get(handle_id)
        return bool(entry and entry.valid)

    def restore_blocks(self, handle: StarKVOffloadHandle) -> bool:
        entry = self._entries.get(handle.handle_id)
        if not entry or not entry.valid:
            return False
        block_ids = entry.block_ids
        if not block_ids:
            return True

        idx_cpu = torch.tensor(block_ids, dtype=torch.long, device="cpu")
        for layer_name, (gpu_tensor, host_tensor) in self._iter_layer_pairs(entry.kv_group_id):
            try:
                block_dim = 0
                if gpu_tensor.dim() >= 2 and gpu_tensor.shape[0] == 2 and gpu_tensor.shape[1] != 2:
                    block_dim = 1

                dim_size = gpu_tensor.size(block_dim)
                if dim_size <= 0:
                    continue

                idx_valid_cpu = idx_cpu[(idx_cpu >= 0) & (idx_cpu < dim_size)]
                if idx_valid_cpu.numel() == 0:
                    logger.debug(
                        "StarKV restore: all indices out of range for layer %s "
                        "(group %d, block_dim=%d, dim_size=%d). Skipping.",
                        layer_name,
                        entry.kv_group_id,
                        block_dim,
                        dim_size,
                    )
                    continue

                idx_gpu = idx_valid_cpu.to(gpu_tensor.device)
                data = torch.index_select(host_tensor, block_dim, idx_valid_cpu).to(gpu_tensor.device)
                gpu_tensor.index_copy_(block_dim, idx_gpu, data)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "StarKV restore failed for request %s blocks %s",
                    entry.request_id,
                    block_ids,
                )
                return False

        entry.valid = False
        return True
# *** End Patch

