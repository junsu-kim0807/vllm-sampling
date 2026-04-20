# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AdaptiveSpechive entrypoint.

This module intentionally re-exports the implementation currently living in
`adaptive_cascade.py` so downstream imports can migrate to the new naming
without changing runtime behavior.
"""

from vllm.v1.spec_decode.adaptive_cascade import (  # noqa: F401
    AdaptiveSpechiveProposer,
    _build_hybrid_bundle_from_rows,
    _flatten_prob_rows_for_output,
    _hv_clone_cad,
    verify_intermediate_chunk_with_prefix_prefab,
)
