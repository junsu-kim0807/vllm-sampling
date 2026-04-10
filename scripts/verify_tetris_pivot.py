#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke and optional E2E checks for ``method=tetris`` and ``method=pivot``.

Default (no GPU required)
-------------------------
Runs TETRIS tensor logic (needs ``torch-scatter``) and pivot
``get_pivot_runtime_mode()`` contract checks using lightweight stubs.
Run from the repository root (or set ``PYTHONPATH`` to it) so ``import vllm`` works.

.. code-block:: bash

   PYTHONPATH=. python scripts/verify_tetris_pivot.py

Optional engine + generate (CUDA + Hugging Face)
-------------------------------------------------
.. code-block:: bash

   PYTHONPATH=. python scripts/verify_tetris_pivot.py --e2e \\
       --target-model facebook/opt-125m --draft-model facebook/opt-125m

Uses the same small models as several vLLM spec-decode tests. TETRIS still
requires ``torch-scatter`` for the draft path used in the runner.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field


def _pivot_spec_stub(
    *,
    pivot_spechive: bool = False,
    pivot_use_eagle_tree: bool = False,
    pivot_verification_pipeline: str | None = None,
    pivot_proposal_engine: str | None = None,
    intermediate_model: str | None = "hf/intermediate-stub",
) -> object:
    """Minimal SpeculativeConfig stand-in for ``get_pivot_runtime_mode()``."""
    from vllm.config import SpeculativeConfig

    @dataclass
    class _HF:
        model_type: str = "llama"

    @dataclass
    class _DraftCfg:
        model: str = "hf/draft-stub"
        hf_config: _HF = field(default_factory=_HF)

    spec = object.__new__(SpeculativeConfig)
    spec.method = "pivot"
    spec.pivot_proposal_engine = pivot_proposal_engine
    spec.pivot_verification_pipeline = pivot_verification_pipeline
    spec.pivot_spechive = pivot_spechive
    spec.pivot_use_eagle_tree = pivot_use_eagle_tree
    spec.pivot_hidden_state_source = None
    spec.draft_model_config = _DraftCfg()
    spec.intermediate_model = intermediate_model
    return spec


def smoke_tetris() -> bool:
    """Exercise ``apply_tetris`` / ``select_proposals`` on CPU-sized tensors."""
    try:
        import torch

        from vllm.v1.spec_decode.tetris import apply_tetris, select_proposals
    except ImportError as e:
        print(f"[tetris] SKIP: import failed ({e})")
        return True

    device = torch.device("cpu")
    b, k = 3, 4
    logp = torch.randn(b, k, device=device)
    out = select_proposals(capacity=b * k, draft_token_logprobs=logp, num_draft_tokens=[k] * b)
    assert len(out) == b, out
    assert all(0 <= x <= k for x in out), out

    tok = torch.randint(0, 1000, (b, k), device=device)
    trimmed = apply_tetris(
        tok,
        logp,
        num_speculative_tokens=k,
        extra_proposals=0,
        turn_on_batch_size=None,
    )
    assert len(trimmed) == b
    for i, row in enumerate(trimmed):
        assert len(row) <= k
        if row:
            assert row == tok[i, : len(row)].tolist()
    print("[tetris] smoke OK (select_proposals + apply_tetris)")
    return True


def smoke_pivot_modes() -> bool:
    """Pivot runtime mode resolution (target_only vs intermediate pipelines)."""
    s1 = _pivot_spec_stub(pivot_spechive=False, intermediate_model="meta-llama/Llama-3.1-8B-Instruct")
    mode1 = s1.get_pivot_runtime_mode()
    assert mode1.verification_pipeline == "target_only", mode1
    assert mode1.proposal_engine == "draft_model", mode1

    s2 = _pivot_spec_stub(pivot_spechive=True)
    mode2 = s2.get_pivot_runtime_mode()
    assert mode2.verification_pipeline == "intermediate_then_target", mode2

    s3 = _pivot_spec_stub(
        pivot_spechive=True,
        pivot_use_eagle_tree=True,
        pivot_proposal_engine="eagle3_head",
    )
    mode3 = s3.get_pivot_runtime_mode()
    assert mode3.verification_pipeline == "intermediate_tree_then_target_tree", mode3

    s4 = _pivot_spec_stub(pivot_verification_pipeline="target_only", pivot_spechive=True)
    mode4 = s4.get_pivot_runtime_mode()
    assert mode4.verification_pipeline == "target_only", (
        "explicit pivot_verification_pipeline must override pivot_spechive default"
    )

    print("[pivot] smoke OK (get_pivot_runtime_mode contracts)")
    return True


def e2e_tetris(*, target_model: str, draft_model: str) -> bool:
    import torch

    if not torch.cuda.is_available():
        print("[tetris] e2e SKIP: CUDA not available")
        return True

    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(f"[tetris] e2e SKIP: {e}")
        return True

    llm = LLM(
        model=target_model,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.4,
        speculative_config={
            "method": "draft_model",
            "model": draft_model,
            "num_speculative_tokens": 3,
            "max_model_len": 512,
            "tetris": True,
            "tetris_extra_proposals": 0,
            "tetris_turn_on_batch_size": None,
        },
    )
    out = llm.generate("Hello", SamplingParams(max_tokens=12, temperature=0))
    assert out and out[0].outputs, out
    print("[tetris] e2e OK (LLM.generate with tetris draft)")
    return True


def e2e_pivot(*, target_model: str, draft_model: str, intermediate_model: str | None) -> bool:
    import torch

    if not torch.cuda.is_available():
        print("[pivot] e2e SKIP: CUDA not available")
        return True

    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(f"[pivot] e2e SKIP: {e}")
        return True

    spec: dict = {
        "method": "pivot",
        "model": draft_model,
        "num_speculative_tokens": 2,
        "pivot_topk_selection": 2,
        "pivot_expansion_pct": 0.2,
        "pivot_spechive": False,
        "pivot_spechive_num_rounds": 1,
        "max_model_len": 512,
    }
    if intermediate_model:
        spec["intermediate_model"] = intermediate_model

    llm = LLM(
        model=target_model,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.4,
        speculative_config=spec,
    )
    out = llm.generate("Hello", SamplingParams(max_tokens=12, temperature=0))
    assert out and out[0].outputs, out
    sc = llm.llm_engine.vllm_config.speculative_config
    assert sc is not None and sc.method == "pivot"
    mode = sc.get_pivot_runtime_mode()
    assert mode.verification_pipeline == "target_only", mode
    if intermediate_model:
        assert sc.intermediate_model_config is None, (
            "target_only pivot must not build intermediate_model_config "
            "even when intermediate_model id is present"
        )
    print("[pivot] e2e OK (LLM.generate pivot target_only)")
    return True


def main() -> int:
    try:
        import torch  # noqa: F401 — vLLM and this script require PyTorch
    except ImportError:
        print(
            "ERROR: PyTorch is not installed. Use the same environment as vLLM "
            "(see https://docs.vllm.ai/).",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run LLM.generate checks (requires CUDA, HF access, torch-scatter for tetris).",
    )
    parser.add_argument(
        "--target-model",
        default="facebook/opt-125m",
        help="Target model id for --e2e (default: facebook/opt-125m).",
    )
    parser.add_argument(
        "--draft-model",
        default="facebook/opt-125m",
        help="Draft model id for --e2e (default: same as target).",
    )
    parser.add_argument(
        "--pivot-intermediate-model",
        default="facebook/opt-125m",
        help=(
            "Optional intermediate HF id passed into pivot speculative_config during "
            "--e2e to ensure it is ignored for target_only (default: reuse opt-125m)."
        ),
    )
    parser.add_argument(
        "--pivot-skip-extra-intermediate-check",
        action="store_true",
        help="Do not pass intermediate_model into pivot e2e config.",
    )
    args = parser.parse_args()

    ok = True
    try:
        ok = smoke_tetris() and ok
        ok = smoke_pivot_modes() and ok
    except Exception:
        print("[verify] smoke FAILED", file=sys.stderr)
        raise

    if args.e2e:
        try:
            im = None if args.pivot_skip_extra_intermediate_check else args.pivot_intermediate_model
            ok = e2e_tetris(target_model=args.target_model, draft_model=args.draft_model) and ok
            ok = e2e_pivot(
                target_model=args.target_model,
                draft_model=args.draft_model,
                intermediate_model=im,
            ) and ok
        except Exception:
            print("[verify] e2e FAILED", file=sys.stderr)
            raise

    if ok:
        print("[verify] all requested checks passed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
