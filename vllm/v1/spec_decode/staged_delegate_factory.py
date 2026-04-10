# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for staged pivot delegate composition."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.spec_decode.staged_eagle import (
    DraftModelProposalEngine,
    EagleHeadProposalEngine,
    IntermediateModelStateProvider,
)


@dataclass(frozen=True)
class PivotStagedDelegates:
    proposal_engine: DraftModelProposalEngine | EagleHeadProposalEngine
    hidden_state_provider: IntermediateModelStateProvider | None


def build_pivot_staged_delegates(
    *,
    proposal_engine_kind: str,
    draft_delegate: object | None,
    eagle_delegate: object | None,
    intermediate_delegate: object | None,
    needs_intermediate_provider: bool,
) -> PivotStagedDelegates:
    if proposal_engine_kind == "draft_model":
        if draft_delegate is None:
            raise ValueError("draft_model proposal engine requires draft delegate.")
        proposal_engine = DraftModelProposalEngine(proposer=draft_delegate)
    elif proposal_engine_kind == "eagle3_head":
        if eagle_delegate is None:
            raise ValueError("eagle3_head proposal engine requires eagle delegate.")
        proposal_engine = EagleHeadProposalEngine(proposer=eagle_delegate)
    else:
        raise ValueError(f"Unsupported pivot proposal engine: {proposal_engine_kind!r}")

    hidden_state_provider = None
    if needs_intermediate_provider:
        if intermediate_delegate is None:
            raise ValueError(
                "intermediate_then_target pipeline requires intermediate delegate."
            )
        hidden_state_provider = IntermediateModelStateProvider(
            proposer=intermediate_delegate
        )

    return PivotStagedDelegates(
        proposal_engine=proposal_engine,
        hidden_state_provider=hidden_state_provider,
    )
