from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StarKVOffloadStore:
    """
    Lightweight request/block index for CPU offload bookkeeping.
    """

    offloaded_blocks_by_request: dict[str, set[int]] = field(default_factory=dict)

    def offload(self, request_id: str, block_ids: list[int]) -> None:
        if not block_ids:
            return
        bucket = self.offloaded_blocks_by_request.setdefault(request_id, set())
        bucket.update(int(bid) for bid in block_ids)

    def restore(self, request_id: str) -> list[int]:
        restored = self.offloaded_blocks_by_request.pop(request_id, set())
        return sorted(restored)

    def clear_request(self, request_id: str) -> None:
        self.offloaded_blocks_by_request.pop(request_id, None)
