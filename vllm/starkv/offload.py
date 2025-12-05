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
        layer_kv_caches: Dict[str, torch.Tensor],
    ) -> None:
        self._kv_cache_config = kv_cache_config
        self._layer_kv_caches = layer_kv_caches
        self._host_buffers: dict[str, torch.Tensor] = {}
        self._entries: dict[int, _StoredEntry] = {}
        self._next_handle_id = 1
        self._group_to_layers: dict[int, list[str]] = {
            idx: list(group.layer_names)
            for idx, group in enumerate(kv_cache_config.kv_cache_groups)
        }
        self._initialize_host_buffers()

    def _initialize_host_buffers(self) -> None:
        for layer_name, kv_tensor in self._layer_kv_caches.items():
            try:
                host_tensor = torch.empty_like(kv_tensor, device="cpu")
            except RuntimeError:
                logger.exception(
                    "Failed to allocate StarKV host buffer for layer %s", layer_name
                )
                raise
            self._host_buffers[layer_name] = host_tensor

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
            idx_gpu = idx_cpu.to(gpu_tensor.device)
            try:
                data = torch.index_select(gpu_tensor, 0, idx_gpu).to("cpu")
                host_tensor.index_copy_(0, idx_cpu, data)
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
        for _, (gpu_tensor, host_tensor) in self._iter_layer_pairs(entry.kv_group_id):
            idx_gpu = idx_cpu.to(gpu_tensor.device)
            try:
                data = torch.index_select(host_tensor, 0, idx_cpu).to(gpu_tensor.device)
                gpu_tensor.index_copy_(0, idx_gpu, data)
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

