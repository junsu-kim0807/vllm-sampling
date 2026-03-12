from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StarKVTierStore:
    """
    Tracks request-level super/sub tier metadata in vLLM-core mode.
    """

    mirrored_blocks_by_request: dict[str, set[int]] = field(default_factory=dict)

    def mirror_blocks(self, request_id: str, block_ids: list[int]) -> None:
        if not block_ids:
            return
        bucket = self.mirrored_blocks_by_request.setdefault(request_id, set())
        bucket.update(int(bid) for bid in block_ids)

    def get_mirrored_blocks(self, request_id: str) -> list[int]:
        return sorted(self.mirrored_blocks_by_request.get(request_id, set()))

    def clear_request(self, request_id: str) -> None:
        self.mirrored_blocks_by_request.pop(request_id, None)

    def clear(self) -> None:
        self.mirrored_blocks_by_request.clear()
