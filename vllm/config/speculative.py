# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

MagicDecMethod = Literal["streaming"]
AdaptiveSpechiveMode = Literal[
    "draft_target",
    "inter_verification",
    "hierarchical_verification",
]
PivotProposalEngine = Literal["draft_model", "eagle3_head"]
PivotVerificationPipeline = Literal[
    "target_only",
    "intermediate_then_target",
    "intermediate_tree_then_target_tree",
]
PivotHiddenStateSource = Literal["target", "intermediate"]
IntermediateKvMode = Literal["draft_like", "mirror_frontier"]

from pydantic import Field, SkipValidation, model_validator
from typing_extensions import Self

from vllm.config import LoadConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import LazyLoader, has_arctic_inference

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
else:
    PretrainedConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )

logger = init_logger(__name__)

MTPModelTypes = Literal[
    "deepseek_mtp",
    "mimo_mtp",
    "glm4_moe_mtp",
    "glm4_moe_lite_mtp",
    "glm_ocr_mtp",
    "ernie_mtp",
    "nemotron_h_mtp",
    "exaone_moe_mtp",
    "qwen3_next_mtp",
    "qwen3_5_mtp",
    "longcat_flash_mtp",
    "mtp",
    "pangu_ultra_moe_mtp",
    "step3p5_mtp",
]
EagleModelTypes = Literal["eagle", "eagle3", "extract_hidden_states", MTPModelTypes]
SpeculativeMethod = Literal[
    "ngram",
    "medusa",
    "mlp_speculator",
    "draft_model",
    "adaptive_spechive",
    "hierarchical_verification",
    "pivot",
    "suffix",
    EagleModelTypes,
]

# Must stay aligned with vllm.v1.sample.rejection_sampler.MAX_SPEC_LEN
_ADAPTIVE_CASCADE_MAX_SPEC_LEN = 128


def _chain_spec_token_tree(num_tokens: int) -> str:
    return str([(i + 1) * (0,) for i in range(num_tokens)])


@config
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    enforce_eager: bool | None = None
    """Override the default enforce_eager from model_config"""
    # General speculative decoding control
    num_speculative_tokens: int = Field(default=None, gt=0)
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""
    model: str | None = None
    """The name of the draft model, eagle head, or additional weights, if
    provided."""
    method: SpeculativeMethod | None = None
    """The name of the speculative method to use. If users provide and set the
    `model` param, the speculative method type will be detected automatically
    if possible, if `model` param is not provided, the method name must be
    provided.

    If using `ngram` method, the related configuration `prompt_lookup_max` and
    `prompt_lookup_min` should be considered."""
    draft_tensor_parallel_size: int | None = Field(default=None, ge=1)
    """The degree of the tensor parallelism for the draft model. Can only be 1
    or the same as the target model's tensor parallel size."""
    tensor_parallel_size: int | None = None
    """Users should pass "draft_tensor_parallel_size". This parameter's purpose is to
    warn users when they mistakenly provide the wrong argument."""

    # Draft model configuration
    quantization: me_quant.QuantizationMethods | None = None
    """Quantization method that was used to quantize the draft model weights.
    If `None`, we assume the model weights are not quantized. Note that it only
    takes effect when using the draft model-based speculative method."""
    max_model_len: int | None = Field(default=None, ge=1)
    """The maximum model length of the draft model. Used when testing the
    ability to skip speculation for some sequences."""
    revision: str | None = None
    """The specific model version to use for the draft model. It can be a
    branch name, a tag name, or a commit id. If unspecified, will use the
    default version."""
    code_revision: str | None = None
    """The specific revision to use for the draft model code on Hugging Face
    Hub. It can be a branch name, a tag name, or a commit id. If unspecified,
    will use the default version."""

    # Advanced control
    disable_padded_drafter_batch: bool = False
    """Disable input padding for speculative decoding. If set to True,
    speculative input batches can contain sequences of different lengths,
    which may only be supported by certain attention backends. This currently
    only affects the EAGLE method of speculation."""
    use_local_argmax_reduction: bool = False
    """Use vocab-parallel local argmax instead of all-gathering full logits
    for draft token generation. Reduces communication from O(vocab_size) to
    O(2 * tp_size) per token. Only applies to greedy draft selection in
    non-tree speculation."""
    use_draft_probs_in_rejection: bool = False
    """If True, draft-model speculation records per-position proposal distributions
    (when the batch is not all-greedy) and passes them to ``RejectionSampler``.
    Enables the stochastic lossless rejection path; ignored for greedy-only batches."""

    # Ngram proposer configuration
    prompt_lookup_max: int | None = Field(default=None, ge=1)
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: int | None = Field(default=None, ge=1)
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""

    # Alternative drafting strategies
    speculative_token_tree: str | None = None
    """Specifies the tree structure for speculative token generation.
    """
    parallel_drafting: bool = False
    """Enable parallel drafting, where all speculative tokens are generated
    in parallel rather than sequentially. This can improve performance but
    requires the speculative model be trained to support parallel drafting.
    Only compatible with EAGLE and draft model methods."""

    # ---- TETRIS fields -------------------------------------------------------
    tetris: bool = False
    """Enable TETRIS optimal draft token selection.

    TETRIS (ACL 2025) selects which draft tokens across all requests in a
    batch to verify, given a total capacity budget, to maximise expected
    accepted tokens.  It requires logprobs from an EAGLE / draft-model
    proposer; ngram proposers are not supported and TETRIS is silently
    skipped for them.

    Reference: https://arxiv.org/pdf/2502.15197
    """
    tetris_extra_proposals: int = 0
    """Number of *extra* draft tokens the drafter generates beyond ``base_k``.

    When ``tetris=True`` and ``extra_proposals > 0``:
      - ``base_k`` = the user-specified ``num_speculative_tokens``
      - The drafter actually generates ``K = base_k + extra_proposals`` tokens
      - Verification capacity = ``base_k × batch_size``

    The ``__post_init__`` method automatically raises
    ``num_speculative_tokens`` from ``base_k`` to ``K`` so the drafter loop
    runs for the wider length.  The original ``base_k`` is saved to
    ``tetris_base_k`` for use by the selection algorithm.

    Paper experiments used extra_proposals = 1, 2, or 3.
    With ``extra_proposals = 0`` capacity equals the full grid and TETRIS
    is a no-op versus vanilla speculative decoding."""
    tetris_turn_on_batch_size: int | None = None
    """Minimum number of concurrently decoded requests required before TETRIS
    is activated.  When the live batch is smaller than this threshold, plain
    speculative decoding is used (no TETRIS selection).  ``None`` means
    TETRIS is always active when ``tetris=True``."""
    tetris_base_k: int | None = None
    """Populated automatically by ``__post_init__``.  Stores the original
    ``num_speculative_tokens`` value (= base_k) before it was expanded by
    ``tetris_extra_proposals``.  Do not set manually."""
    # --------------------------------------------------------------------------
    # required configuration params passed from engine
    target_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the target model."""
    target_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the target model."""

    # params generated in the post-init stage
    draft_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the draft model initialized internal."""
    draft_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the draft model initialized internal."""
    intermediate_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The intermediate (verifier) model config when using adaptive_spechive."""
    intermediate_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """Parallel config for the intermediate model."""

    # Suffix decoding configuration
    suffix_decoding_max_tree_depth: int = 24
    """The maximum depth of the suffix decoding global and prompt trees. The
    tree depth limits the sum of the prefix match and speculation lengths."""

    suffix_decoding_max_cached_requests: int = 10000
    """The maximum number of requests to cache in the global suffix tree. If
    exceeded, will trigger eviction in FIFO order. If set to 0, the global
    suffix tree is disabled and past responses are not cached (prompt trees
    are still used)."""

    suffix_decoding_max_spec_factor: float = 1.0
    """The maximum spec factor for suffix decoding. The spec factor controls
    speculation lengths based on the prefix match length: max_spec_tokens =
    max_spec_factor * prefix_match_length."""

    suffix_decoding_min_token_prob: float = 0.1
    """The minimum token probability for suffix decoding. Will only speculate
    tokens with estimated probability (based on frequency counts) greater than
    or equal to this value."""

    draft_load_config: LoadConfig | None = None
    """Load config for the draft model. If not specified, will use the load
    config from the target model."""

    # adaptive_spechive: draft (D) + intermediate (I) + target (T)
    intermediate_model: str | None = None
    """HF model id for the intermediate verifier (I) when method is adaptive_spechive."""
    intermediate_revision: str | None = None
    """Optional revision for intermediate_model (defaults to draft revision)."""
    intermediate_kv_mode: IntermediateKvMode = "mirror_frontier"
    """How hierarchical / pivot_spechive intermediate verify aligns attention geometry.

    ``mirror_frontier`` (default): keep a scheduler-mirrored ``InputBatch`` and may
    build mirror ``CommonAttentionMetadata`` for intermediate verify (legacy).

    ``draft_like``: intermediate verify consumes the same per-step speculative
    ``common_attn_metadata`` as the draft path; no persistent intermediate mirror
    frontier on the hot path.
    """
    intermediate_tensor_parallel_size: int | None = Field(default=None, ge=1)
    """TP size for I; defaults to draft_tensor_parallel_size when unset."""
    adaptive_spechive_num_interval_tokens: int = Field(default=1, ge=1)
    """Deprecated compatibility field (hierarchical chunk length now uses num_speculative_tokens)."""
    adaptive_spechive_num_rounds: int = Field(default=1, ge=1)
    """Number of hierarchical verification rounds per outer iteration."""
    adaptive_spechive_mode: AdaptiveSpechiveMode = "draft_target"
    """draft_target: draft→T. inter_verification: I proposes like draft→T. hierarchical_verification: D proposes, I verifies via RejectionSampler, then→T."""
    adaptive_spechive_enable_inter_verification: bool = True
    """Must be True when adaptive_spechive_mode is ``inter_verification``."""
    adaptive_spechive_enable_hierarchical_verification: bool = True
    """Must be True when adaptive_spechive_mode is ``hierarchical_verification``."""
    adaptive_spechive_hidden_state_source: PivotHiddenStateSource | None = None
    """When the draft is Eagle3 and staged hierarchical spechive is active: source
    of hidden states for the eagle head (``intermediate`` verifier forward vs
    ``target``). If unset, defaults to ``intermediate`` (same idea as
    ``pivot_hidden_state_source`` for pivot)."""
    num_hv_rounds: int = Field(default=1, ge=1)
    """Number of D→I verification rounds per outer step when ``method`` is
    ``hierarchical_verification`` (chunk length L remains ``num_speculative_tokens``)."""
    pivot_topk_selection: int = Field(default=5)
    """Pivot first-token expansion width; supported values are 2 or 5."""
    pivot_expansion_pct: float = Field(default=0.2, gt=0.0, le=1.0)
    """Fraction of requests expanded with top-k first-token proposals."""
    pivot_min_batch_for_expansion: int = Field(default=8, ge=1)
    """Top-K expansion is only applied when batch_size >= this threshold."""
    pivot_spechive: bool = False
    """Enable pivot staged D=>I rounds followed by authoritative D=>T verification."""
    pivot_spechive_num_rounds: int = Field(default=1, ge=1)
    """Number of intermediate D=>I rounds per outer pivot_spechive iteration."""
    pivot_proposal_engine: PivotProposalEngine | None = None
    """Optional explicit proposal engine override for pivot runtime."""
    pivot_verification_pipeline: PivotVerificationPipeline | None = None
    """Optional explicit verification pipeline override for pivot runtime."""
    pivot_hidden_state_source: PivotHiddenStateSource | None = None
    """Optional hidden-state source for pivot eagle3-head proposal engine."""
    pivot_use_eagle_tree: bool = False
    """Enable root-expanded Pivot families with Eagle tree layout contracts."""
    pivot_family_collapse_policy: Literal["max_accept_len"] = "max_accept_len"
    """Policy used when collapsing expanded families back to origin rows."""

    # MagicDec: optional draft-only attention metadata rewrite (see
    # vllm/v1/attention/magicdec_streaming_attention.py).
    magicdec: bool = False
    """If True, apply MagicDec draft-side attention behavior when supported."""
    magicdec_method: MagicDecMethod = "streaming"
    """Draft attention rewrite strategy. Currently only \"streaming\" is supported."""
    magicdec_kv_budget: int = Field(default=256, ge=1)
    """Token budget for visible KV in streaming mode (sink + recent), block-aligned."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        # Aux-hidden outputs affect model graph shape for eagle3/extract-hidden and
        # pivot modes that route proposal through eagle3_head.
        uses_aux_hidden_states = self.requires_aux_hidden_state_outputs()
        factors.append(uses_aux_hidden_states)

        factors.append(self.magicdec)
        if self.magicdec:
            factors.append(self.magicdec_method)
            factors.append(self.magicdec_kv_budget)

        if self.method == "adaptive_spechive":
            factors.append(self.intermediate_model)
            factors.append(self.intermediate_kv_mode)
            factors.append(self.adaptive_spechive_num_interval_tokens)
            factors.append(self.adaptive_spechive_num_rounds)
            factors.append(self.adaptive_spechive_mode)
            factors.append(self.adaptive_spechive_enable_inter_verification)
            factors.append(self.adaptive_spechive_enable_hierarchical_verification)
            factors.append(self.adaptive_spechive_hidden_state_source)
            factors.append(self.adaptive_spechive_draft_uses_eagle3_head())
        elif self.method == "hierarchical_verification":
            factors.append(self.intermediate_model)
            factors.append(self.intermediate_kv_mode)
            factors.append(self.num_hv_rounds)
            factors.append(self.num_speculative_tokens)
        elif self.method == "pivot":
            mode = self.get_pivot_runtime_mode()
            factors.append(self.intermediate_model)
            factors.append(self.intermediate_kv_mode)
            factors.append(self.pivot_topk_selection)
            factors.append(self.pivot_expansion_pct)
            factors.append(self.pivot_spechive)
            factors.append(self.pivot_spechive_num_rounds)
            factors.append(self.pivot_use_eagle_tree)
            factors.append(self.pivot_family_collapse_policy)
            factors.append(mode.proposal_engine)
            factors.append(mode.verification_pipeline)
            factors.append(mode.hidden_state_source)

        # The specific layers used also affect the computation graph
        if uses_aux_hidden_states and self.draft_model_config is not None:
            layer_ids = getattr(
                self.draft_model_config.hf_config,
                "eagle_aux_hidden_state_layer_ids",
                None,
            )
            if layer_ids is not None:
                # Convert to tuple to make it hashable
                factors.append(tuple(layer_ids))

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @dataclass(frozen=True)
    class PivotRuntimeMode:
        proposal_engine: PivotProposalEngine
        verification_pipeline: PivotVerificationPipeline
        hidden_state_source: PivotHiddenStateSource | None

    def get_pivot_runtime_mode(self) -> PivotRuntimeMode:
        """Return normalized pivot runtime mode (proposal × verification)."""
        if self.method != "pivot":
            raise ValueError("get_pivot_runtime_mode is only valid for method='pivot'.")

        proposal_engine = self.pivot_proposal_engine
        if proposal_engine is None:
            if self.draft_model_config is not None and (
                getattr(self.draft_model_config.hf_config, "method", None) == "eagle3"
                or "eagle3" in str(self.draft_model_config.model).lower()
            ):
                proposal_engine = "eagle3_head"
            else:
                proposal_engine = "draft_model"

        verification_pipeline = self.pivot_verification_pipeline
        if verification_pipeline is None:
            if self.pivot_use_eagle_tree and self.pivot_spechive:
                verification_pipeline = "intermediate_tree_then_target_tree"
            else:
                verification_pipeline = (
                    "intermediate_then_target" if self.pivot_spechive else "target_only"
                )

        hidden_state_source = self.pivot_hidden_state_source
        if proposal_engine == "draft_model":
            hidden_state_source = None
        elif hidden_state_source is None:
            hidden_state_source = (
                "intermediate"
                if verification_pipeline == "intermediate_then_target"
                else "target"
            )

        return SpeculativeConfig.PivotRuntimeMode(
            proposal_engine=proposal_engine,
            verification_pipeline=verification_pipeline,
            hidden_state_source=hidden_state_source,
        )

    def pivot_packed_batch_size_for_origin_batch(self, origin_batch_size: int) -> int:
        """Verifier packed row count P for linear fixed-capacity pivot.

        ``P = B + ceil(B * pivot_expansion_pct) * (pivot_topk_selection - 1)``.
        Uses config top-k (actual runtime K may be ``min(topk, vocab)``).
        """
        if self.method != "pivot" or origin_batch_size <= 0:
            return 0
        b = int(origin_batch_size)
        k = int(self.pivot_topk_selection)
        extra_per_block = max(0, k - 1)
        num_blocks = int(math.ceil(b * float(self.pivot_expansion_pct)))
        return b + num_blocks * extra_per_block

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
        if hf_config.model_type in ("deepseek_v3", "deepseek_v32", "glm_moe_dsa"):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
            )
        if hf_config.model_type in ("pangu_ultra_moe"):
            hf_config.model_type = "pangu_ultra_moe_mtp"
        if hf_config.model_type == "pangu_ultra_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]}
            )

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["MiMoMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeLiteForCausalLM":
            hf_config.model_type = "glm4_moe_lite_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeLiteMTPModel"],
                }
            )

        if hf_config.architectures[0] == "GlmOcrForConditionalGeneration":
            hf_config.model_type = "glm_ocr_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["GlmOcrMTPModel"],
                }
            )

        if hf_config.model_type == "ernie4_5_moe":
            hf_config.model_type = "ernie_mtp"
        if hf_config.model_type == "ernie_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ErnieMTPModel"]}
            )

        if (
            hf_config.model_type == "nemotron_h"
            and hasattr(hf_config, "num_nextn_predict_layers")
            and hf_config.num_nextn_predict_layers > 0
        ):
            # Check if this is an MTP variant
            hf_config.model_type = "nemotron_h_mtp"
        if hf_config.model_type == "nemotron_h_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["NemotronHMTPModel"]}
            )

        if hf_config.model_type == "qwen3_next":
            hf_config.model_type = "qwen3_next_mtp"
        if hf_config.model_type == "qwen3_next_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}
            )

        if hf_config.model_type == "exaone_moe":
            hf_config.model_type = "exaone_moe_mtp"
        if hf_config.model_type == "exaone_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ExaoneMoeMTP"]}
            )

        if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
            is_moe = hf_config.model_type == "qwen3_5_moe"
            hf_config.model_type = "qwen3_5_mtp"
            n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
                }
            )
        if hf_config.model_type == "longcat_flash":
            hf_config.model_type = "longcat_flash_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]}
            )

        if hf_config.model_type == "step3p5":
            hf_config.model_type = "step3p5_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

        if initial_architecture == "MistralLarge3ForCausalLM":
            hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

        return hf_config

    def __post_init__(self):
        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        # infer method from user args
        if self.method is None:
            if self.model in ("ngram", "[ngram]"):
                self.method = "ngram"
            else:
                self.method = "draft_model"

        if self.method in get_args(MTPModelTypes) and self.method != "mtp":
            logger.warning(
                "method `%s` is deprecated and replaced with mtp.", self.method
            )
            self.method = "mtp"

        if self.model is None and self.num_speculative_tokens is not None:
            if self.method == "mtp":
                if self.target_model_config is None:
                    raise ValueError("target_model_config must be present for mtp")
                if self.target_model_config.hf_text_config.model_type == "deepseek_v32":
                    # FIXME(luccafong): cudagraph with v32 MTP is not supported,
                    # remove this when the issue is fixed.
                    self.enforce_eager = True
                # use the draft model from the same model:
                self.model = self.target_model_config.model
                # Align the quantization of draft model for cases such as
                # --quantization fp8 with a bf16 checkpoint.
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            elif self.method == "suffix":
                self.model = "suffix"
            elif self.method == "extract_hidden_states":
                self.model = "extract_hidden_states"
            else:
                raise ValueError(
                    "num_speculative_tokens was provided but without speculative model."
                )

        if self.method in ("ngram", "[ngram]"):
            # Unified to "ngram" internally
            self.method = "ngram"
            # Set default values if not provided
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                if self.prompt_lookup_max is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                if self.prompt_lookup_min is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}"
                )

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method == "suffix":
            self._validate_suffix_decoding()
        elif self.method == "extract_hidden_states":
            from vllm.transformers_utils.configs.extract_hidden_states import (
                ExtractHiddenStatesConfig,
            )

            # ExtractHiddenStatesModel is instantiated manually in load_model()
            # We just need to store the target model config for KV cache shape info
            self.model = "extract_hidden_states"
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if hasattr(self.draft_model_config, "hf_config"):
                hf_config = self.draft_model_config.hf_config.to_dict()
            elif (
                isinstance(self.draft_model_config, dict)
                and "hf_config" in self.draft_model_config
            ):
                hf_config = self.draft_model_config["hf_config"]
            else:
                hf_config = {}

            self.draft_model_config = copy.copy(self.target_model_config)
            self.draft_model_config.hf_config = ExtractHiddenStatesConfig(
                self.draft_model_config.hf_config, **hf_config
            )
            self.update_arch_()
            self.draft_parallel_config = self.target_parallel_config

        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    runner="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.trust_remote_code,
                    allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                    allowed_media_domains=self.target_model_config.allowed_media_domains,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.tokenizer_revision,
                    spec_target_max_model_len=self.target_model_config.max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                    config_format=self.target_model_config.config_format,
                )

                # Automatically detect the method
                if self.method == "adaptive_spechive":
                    if not self.intermediate_model:
                        raise ValueError(
                            "adaptive_spechive requires `intermediate_model` "
                            "(intermediate I) in addition to `model` (draft D)."
                        )
                elif self.method == "hierarchical_verification":
                    if not self.intermediate_model:
                        raise ValueError(
                            "hierarchical_verification requires `intermediate_model` "
                            "(intermediate I) in addition to `model` (draft D)."
                        )
                elif self.method == "pivot":
                    mode = self.get_pivot_runtime_mode()
                    needs_intermediate = mode.verification_pipeline in (
                        "intermediate_then_target",
                        "intermediate_tree_then_target_tree",
                    )
                    if needs_intermediate and not self.intermediate_model:
                        raise ValueError(
                            "pivot intermediate* pipeline requires `intermediate_model` "
                            "in addition to `model` (draft D)."
                        )
                elif self.method in ("eagle", "eagle3"):
                    pass
                # examples:
                # yuhuili/EAGLE-LLaMA3-Instruct-8B
                # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
                # AngelSlim/Qwen3-8B_eagle3
                elif "eagle-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif "eagle3" in self.draft_model_config.model.lower():
                    self.method = "eagle3"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif self.draft_model_config.hf_config.model_type == "mlp_speculator":
                    self.method = "mlp_speculator"
                elif self.draft_model_config.hf_config.model_type in get_args(
                    MTPModelTypes
                ):
                    self.method = "mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                            "Enabling num_speculative_tokens > 1 will run "
                            "multiple times of forward on same MTP layer"
                            ",which may result in lower acceptance rate"
                        )
                elif self.draft_model_config.hf_config.model_type in (
                    "longcat_flash_mtp"
                ):
                    self.method = "longcat_flash_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                            "LongCat MTP models only have "
                            "one layer. Might need some code changes "
                            "to support multiple layers."
                        )
                elif self.method in ("draft_model", "pivot", "hierarchical_verification"):
                    pass
                else:
                    raise NotImplementedError(
                        f"Unsupported speculative method: '{self.method}'"
                    )

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3"):
                    from vllm.transformers_utils.configs import SpeculatorsConfig
                    from vllm.transformers_utils.configs.eagle import EAGLEConfig

                    if isinstance(
                        self.draft_model_config.hf_config,
                        (EAGLEConfig, SpeculatorsConfig),
                    ):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method,
                            model_type="eagle",
                        )
                        self.draft_model_config.hf_config = eagle_config
                        self.update_arch_()

                if self.num_speculative_tokens is not None and hasattr(
                    self.draft_model_config.hf_config, "num_lookahead_tokens"
                ):
                    self.draft_model_config.hf_config.num_lookahead_tokens = (
                        self.num_speculative_tokens
                    )

                n_predict = getattr(
                    self.draft_model_config.hf_config, "n_predict", None
                )
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif (
                        self.num_speculative_tokens > n_predict
                        and self.num_speculative_tokens % n_predict != 0
                    ):
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}"
                        )

                # --- TETRIS: expand num_speculative_tokens by extra_proposals ---
                # User specifies base_k via num_speculative_tokens; when TETRIS
                # is active we bump the drafter length to K = base_k + extra so
                # the selection algorithm has a wider grid to choose from while
                # the verification budget stays at base_k × B.
                if (
                    self.tetris
                    and self.tetris_extra_proposals > 0
                    and self.num_speculative_tokens is not None
                ):
                    base_k = self.num_speculative_tokens
                    self.tetris_base_k = base_k
                    self.num_speculative_tokens = base_k + self.tetris_extra_proposals
                    logger.info(
                        "TETRIS: expanding num_speculative_tokens from "
                        "base_k=%d to K=%d (extra_proposals=%d).",
                        base_k,
                        self.num_speculative_tokens,
                        self.tetris_extra_proposals,
                    )
                elif self.tetris and self.num_speculative_tokens is not None:
                    self.tetris_base_k = self.num_speculative_tokens
                # --- end TETRIS expansion ---

                if self.speculative_token_tree is None:
                    if self.num_speculative_tokens is None:
                        raise ValueError(
                            "A speculative model was provided, but neither "
                            "`speculative_token_tree` nor `num_speculative_tokens` "
                            "was provided"
                        )

                    # Generate chain of tokens.
                    self.speculative_token_tree = str(
                        [(i + 1) * (0,) for i in range(self.num_speculative_tokens)]
                    )
                else:
                    # Sort the token tree breadth-first.
                    tree_choices = ast.literal_eval(self.speculative_token_tree)
                    self.speculative_token_tree = str(
                        sorted(tree_choices, key=lambda t: (len(t), t))
                    )

                self.draft_tensor_parallel_size = (
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config,
                    )
                )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    )
                )

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config, self.draft_tensor_parallel_size
                    )
                )

                if self.method in ("adaptive_spechive", "hierarchical_verification"):
                    assert self.intermediate_model is not None
                    int_rev = (
                        self.intermediate_revision
                        if self.intermediate_revision is not None
                        else self.revision
                    )
                    self.intermediate_model_config = ModelConfig(
                        model=self.intermediate_model,
                        runner="draft",
                        tokenizer=self.target_model_config.tokenizer,
                        tokenizer_mode=self.target_model_config.tokenizer_mode,
                        trust_remote_code=self.target_model_config.trust_remote_code,
                        allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                        allowed_media_domains=self.target_model_config.allowed_media_domains,
                        dtype=self.target_model_config.dtype,
                        seed=self.target_model_config.seed,
                        revision=int_rev,
                        code_revision=self.code_revision,
                        tokenizer_revision=self.target_model_config.tokenizer_revision,
                        spec_target_max_model_len=self.target_model_config.max_model_len,
                        quantization=self.quantization,
                        enforce_eager=self.target_model_config.enforce_eager,
                        max_logprobs=self.target_model_config.max_logprobs,
                        hf_overrides=SpeculativeConfig.hf_config_override,
                        config_format=self.target_model_config.config_format,
                    )
                    int_tp = (
                        self.intermediate_tensor_parallel_size
                        or self.draft_tensor_parallel_size
                    )
                    self.intermediate_tensor_parallel_size = (
                        SpeculativeConfig._verify_and_get_draft_tp(
                            self.target_parallel_config,
                            int_tp,
                            self.intermediate_model_config.hf_config,
                        )
                    )
                    self.intermediate_model_config.max_model_len = (
                        SpeculativeConfig._maybe_override_draft_max_model_len(
                            self.max_model_len,
                            self.intermediate_model_config.max_model_len,
                            self.target_model_config.max_model_len,
                        )
                    )
                    self.intermediate_parallel_config = (
                        SpeculativeConfig.create_draft_parallel_config(
                            self.target_parallel_config,
                            self.intermediate_tensor_parallel_size,
                        )
                    )
                elif self.method == "pivot" and self.intermediate_model is not None:
                    # target_only pivot uses draft top-k expansion only; ignore
                    # intermediate_model unless verification uses an I stage.
                    if self.get_pivot_runtime_mode().verification_pipeline in (
                        "intermediate_then_target",
                        "intermediate_tree_then_target_tree",
                    ):
                        int_rev = (
                            self.intermediate_revision
                            if self.intermediate_revision is not None
                            else self.revision
                        )
                        self.intermediate_model_config = ModelConfig(
                            model=self.intermediate_model,
                            runner="draft",
                            tokenizer=self.target_model_config.tokenizer,
                            tokenizer_mode=self.target_model_config.tokenizer_mode,
                            trust_remote_code=self.target_model_config.trust_remote_code,
                            allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                            allowed_media_domains=self.target_model_config.allowed_media_domains,
                            dtype=self.target_model_config.dtype,
                            seed=self.target_model_config.seed,
                            revision=int_rev,
                            code_revision=self.code_revision,
                            tokenizer_revision=self.target_model_config.tokenizer_revision,
                            spec_target_max_model_len=self.target_model_config.max_model_len,
                            quantization=self.quantization,
                            enforce_eager=self.target_model_config.enforce_eager,
                            max_logprobs=self.target_model_config.max_logprobs,
                            hf_overrides=SpeculativeConfig.hf_config_override,
                            config_format=self.target_model_config.config_format,
                        )
                        int_tp = (
                            self.intermediate_tensor_parallel_size
                            or self.draft_tensor_parallel_size
                        )
                        self.intermediate_tensor_parallel_size = (
                            SpeculativeConfig._verify_and_get_draft_tp(
                                self.target_parallel_config,
                                int_tp,
                                self.intermediate_model_config.hf_config,
                            )
                        )
                        self.intermediate_model_config.max_model_len = (
                            SpeculativeConfig._maybe_override_draft_max_model_len(
                                self.max_model_len,
                                self.intermediate_model_config.max_model_len,
                                self.target_model_config.max_model_len,
                            )
                        )
                        self.intermediate_parallel_config = (
                            SpeculativeConfig.create_draft_parallel_config(
                                self.target_parallel_config,
                                self.intermediate_tensor_parallel_size,
                            )
                        )
        return self

    def _validate_suffix_decoding(self):
        if not has_arctic_inference():
            raise ImportError(
                "Arctic Inference is required for suffix decoding. "
                "Install via `pip install arctic-inference==0.1.1`."
            )
        if self.num_speculative_tokens is None:
            # Suffix decoding decides the actual number of speculative tokens
            # dynamically and treats num_speculative_tokens as a maximum limit.
            self.num_speculative_tokens = self.suffix_decoding_max_tree_depth
            logger.warning(
                "Defaulted num_speculative_tokens to %s for suffix decoding.",
                self.num_speculative_tokens,
            )
        # Validate values
        if self.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{self.suffix_decoding_max_tree_depth} must be >= 1"
            )
        if self.suffix_decoding_max_cached_requests < 0:
            raise ValueError(
                f"suffix_decoding_max_cached_requests="
                f"{self.suffix_decoding_max_cached_requests} must be >= 0"
            )
        if self.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{self.suffix_decoding_max_spec_factor} must be >= 0"
            )
        if not 0 <= self.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{self.suffix_decoding_min_token_prob} must be in [0, 1]"
            )

    @staticmethod
    def _maybe_override_draft_max_model_len(
        speculative_max_model_len: int | None,
        draft_max_model_len: int,
        target_max_model_len: int,
    ) -> int:
        """Determine the max sequence len for the draft model. This is usually
        the draft_max_model_len, but may be the target_max_model_len if it is
        less than the draft_max_model_len, or may be speculative_max_model_len
        if it is specified.

        This is necessary so that sequences do not exceed the capacity of the
        draft model or the target model.

        speculative_max_model_len is mainly used for testing that sequences can
        skip speculation.
        """

        if speculative_max_model_len is not None:
            if speculative_max_model_len > draft_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {draft_max_model_len=}"
                )

            if speculative_max_model_len > target_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {target_max_model_len=}"
                )

            return speculative_max_model_len

        return min(
            draft_max_model_len,
            target_max_model_len,
        )

    @staticmethod
    def _verify_and_get_draft_tp(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int | None,
        draft_hf_config: PretrainedConfig,
    ) -> int:
        """
        Verifies and adjusts the tensor parallel size for a draft model
        specified using speculative_draft_tensor_parallel_size.
        """
        # If speculative_draft_tensor_parallel_size is unset then set it
        # appropriately else verify that it is set correctly.
        if speculative_draft_tensor_parallel_size is None:
            if draft_hf_config.model_type == "mlp_speculator":
                speculative_draft_tensor_parallel_size = 1
                if target_parallel_config.tensor_parallel_size > 1:
                    logger.warning(
                        "%s cannot currently be run with tp>1; "
                        "setting speculative_draft_tensor_parallel_size=1",
                        draft_hf_config.model_type,
                    )
            else:
                speculative_draft_tensor_parallel_size = (
                    target_parallel_config.tensor_parallel_size
                )
        elif speculative_draft_tensor_parallel_size not in (
            1,
            target_parallel_config.tensor_parallel_size,
        ):
            raise ValueError(
                f"{speculative_draft_tensor_parallel_size=} cannot be "
                f"other value than 1 or target model tensor_parallel_size"
            )
        return speculative_draft_tensor_parallel_size

    def update_arch_(self):
        """
        EagleConfig and ExtractHiddenStatesConfig update architectures, so update all
        architectures-related fields in self.draft_model_config
        """
        self.draft_model_config.hf_text_config = get_hf_text_config(
            self.draft_model_config.hf_config
        )
        self.draft_model_config.model_arch_config = (
            self.draft_model_config.get_model_arch_config()
        )
        model_info, arch = self.draft_model_config.registry.inspect_model_cls(
            self.draft_model_config.architectures,
            self.draft_model_config,
        )
        self.draft_model_config._model_info = model_info
        self.draft_model_config._architecture = arch

    @staticmethod
    def create_draft_parallel_config(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int,
    ) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.

        This is mostly a copy of the target parallel config, except the tp_size.
        """
        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,
            tensor_parallel_size=speculative_draft_tensor_parallel_size,
            distributed_executor_backend=target_parallel_config.distributed_executor_backend,
            max_parallel_loading_workers=target_parallel_config.max_parallel_loading_workers,
            disable_custom_all_reduce=target_parallel_config.disable_custom_all_reduce,
            ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        if self.tensor_parallel_size is not None:
            raise ValueError(
                "'tensor_parallel_size' is not a valid argument in the "
                "speculative_config. Please pass 'draft_tensor_parallel_size' instead."
            )

        if self.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative model unless the draft model config contains an "
                "n_predict parameter."
            )

        if self.num_speculative_tokens <= 0:
            raise ValueError(
                "Expected num_speculative_tokens to be greater "
                f"than zero ({self.num_speculative_tokens})."
            )

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )

        if (
            self.method in ("adaptive_spechive", "pivot", "hierarchical_verification")
            and self.intermediate_model_config is not None
            and self.intermediate_parallel_config is not None
        ):
            self.intermediate_model_config.verify_with_parallel_config(
                self.intermediate_parallel_config
            )

        if self.method == "adaptive_spechive":
            if self.adaptive_spechive_mode not in (
                "draft_target",
                "inter_verification",
                "hierarchical_verification",
            ):
                raise ValueError(
                    f"Invalid adaptive_spechive_mode={self.adaptive_spechive_mode!r}; "
                    "expected 'draft_target', 'inter_verification', or 'hierarchical_verification'."
                )
            if (
                self.adaptive_spechive_mode == "inter_verification"
                and not self.adaptive_spechive_enable_inter_verification
            ):
                raise ValueError(
                    "adaptive_spechive_mode='inter_verification' requires adaptive_spechive_enable_inter_verification=True."
                )
            if (
                self.adaptive_spechive_mode == "hierarchical_verification"
                and not self.adaptive_spechive_enable_hierarchical_verification
            ):
                raise ValueError(
                    "adaptive_spechive_mode='hierarchical_verification' requires adaptive_spechive_enable_hierarchical_verification=True."
                )
            if self.adaptive_spechive_mode in (
                "inter_verification",
                "hierarchical_verification",
            ):
                assert self.draft_model_config is not None
                assert self.intermediate_model_config is not None
                d_h = self.draft_model_config.get_hidden_size()
                i_h = self.intermediate_model_config.get_hidden_size()
                if d_h != i_h:
                    raise ValueError(
                        "adaptive_spechive inter/hierarchical verification requires draft and intermediate models "
                        f"to share the same hidden size (got draft={d_h}, "
                        f"intermediate={i_h})."
                    )
            chunk_len = self.num_speculative_tokens
            rounds = self.adaptive_spechive_num_rounds
            outer_upper = chunk_len + rounds * (chunk_len + 1)
            if outer_upper > _ADAPTIVE_CASCADE_MAX_SPEC_LEN:
                raise ValueError(
                    f"adaptive_spechive: implied max outer verify length "
                    f"{outer_upper} = chunk_len + rounds * (chunk_len + 1) "
                    f"exceeds {_ADAPTIVE_CASCADE_MAX_SPEC_LEN} (sampler limit). "
                    "Reduce num_speculative_tokens or adaptive_spechive_num_rounds."
                )
            if self.parallel_drafting:
                raise ValueError(
                    "parallel_drafting is not supported with adaptive_spechive."
                )
        if self.method == "hierarchical_verification":
            assert self.draft_model_config is not None
            assert self.intermediate_model_config is not None
            d_h = self.draft_model_config.get_hidden_size()
            i_h = self.intermediate_model_config.get_hidden_size()
            if d_h != i_h:
                raise ValueError(
                    "hierarchical_verification requires draft and intermediate models "
                    f"to share the same hidden size (got draft={d_h}, "
                    f"intermediate={i_h})."
                )
            outer_upper = self.hv_max_spec_len()
            if outer_upper > _ADAPTIVE_CASCADE_MAX_SPEC_LEN:
                raise ValueError(
                    f"hierarchical_verification: implied max target verify length "
                    f"{outer_upper} = hv_max_spec_len() exceeds "
                    f"{_ADAPTIVE_CASCADE_MAX_SPEC_LEN} (sampler limit). "
                    "Reduce num_speculative_tokens or num_hv_rounds."
                )
            if self.parallel_drafting:
                raise ValueError(
                    "parallel_drafting is not supported with hierarchical_verification."
                )
            dc = self.draft_model_config
            if getattr(dc.hf_config, "method", None) == "eagle3" or "eagle3" in str(
                dc.model
            ).lower():
                raise ValueError(
                    "hierarchical_verification (standalone) does not support Eagle3 "
                    "draft yet; use a draft_model method='draft_model' checkpoint."
                )
            if self.target_model_config.is_multimodal_model():
                raise ValueError(
                    "hierarchical_verification (v1) does not support multimodal "
                    "target models."
                )
        if self.method == "pivot" and self.parallel_drafting:
            raise ValueError("parallel_drafting is not supported with pivot.")
        if self.method == "pivot":
            mode = self.get_pivot_runtime_mode()
            if self.pivot_topk_selection not in (2, 5):
                raise ValueError(
                    "pivot_topk_selection must be one of {2, 5}."
                )
            if (
                mode.verification_pipeline == "intermediate_then_target"
                or mode.verification_pipeline == "intermediate_tree_then_target_tree"
            ):
                chunk_len = self.num_speculative_tokens
                rounds = self.pivot_spechive_num_rounds
                outer_upper = chunk_len + rounds * (chunk_len + 1)
                if outer_upper > _ADAPTIVE_CASCADE_MAX_SPEC_LEN:
                    raise ValueError(
                        f"pivot_spechive: implied max outer verify length "
                        f"{outer_upper} = chunk_len + rounds * (chunk_len + 1) "
                        f"exceeds {_ADAPTIVE_CASCADE_MAX_SPEC_LEN} (sampler limit). "
                        "Reduce num_speculative_tokens or pivot_spechive_num_rounds."
                    )
            if (
                (
                    mode.verification_pipeline == "intermediate_then_target"
                    or mode.verification_pipeline == "intermediate_tree_then_target_tree"
                )
                and self.intermediate_model_config is None
            ):
                raise ValueError(
                    "pivot intermediate* pipeline requires intermediate_model."
                )
            if (
                self.pivot_use_eagle_tree
                and mode.proposal_engine != "eagle3_head"
            ):
                raise ValueError(
                    "pivot_use_eagle_tree=True requires pivot_proposal_engine='eagle3_head'."
                )
            if (
                not self.pivot_use_eagle_tree
                and mode.verification_pipeline == "intermediate_tree_then_target_tree"
            ):
                raise ValueError(
                    "intermediate_tree_then_target_tree is only valid when "
                    "pivot_use_eagle_tree=True."
                )
            if (
                mode.proposal_engine == "eagle3_head"
                and (
                    mode.verification_pipeline == "intermediate_then_target"
                    or mode.verification_pipeline == "intermediate_tree_then_target_tree"
                )
                and mode.hidden_state_source != "intermediate"
            ):
                raise ValueError(
                    "pivot eagle3_head + intermediate* pipeline requires "
                    "pivot_hidden_state_source='intermediate'."
                )
            if (
                mode.proposal_engine == "eagle3_head"
                and mode.verification_pipeline == "target_only"
                and mode.hidden_state_source != "target"
            ):
                raise ValueError(
                    "pivot eagle3_head + target_only requires "
                    "pivot_hidden_state_source='target'."
                )
            if (
                not self.pivot_use_eagle_tree
                and self.speculative_token_tree is not None
                and self.speculative_token_tree != _chain_spec_token_tree(self.num_speculative_tokens)
            ):
                raise ValueError(
                    "linear pivot only supports root-only expansion with chain semantics "
                    "after root; non-chain speculative_token_tree is not supported."
                )
            if (
                self.pivot_family_collapse_policy != "max_accept_len"
            ):
                raise ValueError(
                    "Unsupported pivot_family_collapse_policy. "
                    "Only 'max_accept_len' is supported."
                )
            if (
                mode.proposal_engine == "eagle3_head"
                and self.disable_padded_drafter_batch
            ):
                raise ValueError(
                    "pivot + eagle3_head currently requires padded drafter batch "
                    "(disable_padded_drafter_batch=False)."
                )

        aux_hidden_states_supported = [
            "llama",
            "qwen",
            "minicpm",
            "gpt_oss",
            "hunyuan_vl",
            "hunyuan_v1_dense",
            "afmoe",
            "nemotron_h",
        ]
        if (
            self.method in ("eagle3", "extract_hidden_states")
            and self.target_model_config
            and not any(
                supported_model in self.target_model_config.hf_text_config.model_type
                for supported_model in aux_hidden_states_supported
            )
        ):
            raise ValueError(
                f"{self.method} is only supported for {aux_hidden_states_supported}"
                f" models. Got {self.target_model_config.hf_text_config.model_type=}"
            )
        self.verify_equal_vocab_size_if_draft_model()
        if self.magicdec:
            if self.method != "draft_model":
                raise ValueError(
                    "magicdec is only supported with speculative method "
                    "'draft_model' (draft-model proposer)."
                )
            if self.magicdec_method != "streaming":
                raise ValueError(
                    f"Unsupported magicdec_method={self.magicdec_method!r}; "
                    "only 'streaming' is implemented."
                )
        return self

    def verify_equal_vocab_size_if_draft_model(self):
        if (
            self.method in ("draft_model", "adaptive_spechive", "hierarchical_verification")
            and self.target_model_config is not None
            and self.draft_model_config is not None
        ):
            target_vocab_size = self.target_model_config.get_vocab_size()
            draft_vocab_size = self.draft_model_config.get_vocab_size()
            if target_vocab_size != draft_vocab_size:
                raise ValueError(
                    f"Target and draft model should have the same vocabulary size. "
                    f"Target model vocab_size={target_vocab_size}. "
                    f"Draft model vocab_size={draft_vocab_size}. "
                    f"Using models with different tokenizers can cause out-of-bounds "
                    f"errors during speculative decoding."
                )
        if (
            self.method == "pivot"
            and self.target_model_config is not None
            and self.draft_model_config is not None
            and self.get_pivot_runtime_mode().proposal_engine == "draft_model"
        ):
            target_vocab_size = self.target_model_config.get_vocab_size()
            draft_vocab_size = self.draft_model_config.get_vocab_size()
            if target_vocab_size != draft_vocab_size:
                raise ValueError(
                    f"Target and draft model should have the same vocabulary size. "
                    f"Target model vocab_size={target_vocab_size}. "
                    f"Draft model vocab_size={draft_vocab_size}. "
                    f"Using models with different tokenizers can cause out-of-bounds "
                    f"errors during speculative decoding."
                )
        if self.target_model_config is None or self.intermediate_model_config is None:
            return
        if self.method in ("adaptive_spechive", "hierarchical_verification"):
            target_vocab_size = self.target_model_config.get_vocab_size()
            i_vocab = self.intermediate_model_config.get_vocab_size()
            if target_vocab_size != i_vocab:
                raise ValueError(
                    "Target and intermediate (verifier) model must share the same "
                    f"vocabulary size. Target vocab_size={target_vocab_size}, "
                    f"intermediate vocab_size={i_vocab}."
                )
        if self.method == "pivot":
            target_vocab_size = self.target_model_config.get_vocab_size()
            i_vocab = self.intermediate_model_config.get_vocab_size()
            if (
                target_vocab_size != i_vocab
                and not self._pivot_intermediate_uses_shared_output_semantics()
            ):
                raise ValueError(
                    "pivot requires intermediate output compatibility with target. "
                    "Either use an intermediate model with matching vocabulary size "
                    "(direct token-id emission), or an EAGLE-style intermediate model "
                    "that declares shared output semantics."
                )

    def _pivot_intermediate_uses_shared_output_semantics(self) -> bool:
        if self.intermediate_model_config is None:
            return False
        hf_config = self.intermediate_model_config.hf_config
        model_type = getattr(hf_config, "model_type", "")
        if model_type in ("eagle", "speculators"):
            return True
        for arch in getattr(hf_config, "architectures", []) or []:
            if isinstance(arch, str) and "Eagle" in arch:
                return True
        return False

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        """
        Calculate the maximum number of new slots that might be added to the batch
        when drafting.
        """
        slots_per_req = 0  # for serial non-draft-model methods, no change needed
        if self.parallel_drafting:
            # For parallel drafting, we need one new slot per 'masked' token
            slots_per_req = self.num_speculative_tokens - 1
        if self.uses_model_based_drafter():
            # For draft model-based speculation, we need one new slot per request
            # Since we do not slice the draft tokens
            slots_per_req += 1
        return slots_per_req

    def hv_chunk_len(self) -> int:
        """Per-round draft proposal chunk length L (``num_speculative_tokens``)."""
        return int(self.num_speculative_tokens)

    def hv_max_spec_len(self) -> int:
        """Exact upper bound on accepted speculative length before final target verify.

        For R = ``num_hv_rounds`` and L = chunk length, each inner round contributes at
        most L+1 tokens; the final tail adds at most L: ``R * (L + 1) + L``.
        """
        l = int(self.num_speculative_tokens)
        r = int(self.num_hv_rounds)
        return r * (l + 1) + l

    def runner_num_speculative_tokens(self) -> int:
        """Tensor width aligned with the **authoritative target verification** contract.

        For most methods this equals ``num_speculative_tokens``. For staged
        hierarchical modes (adaptive/pivot intermediate pipelines, or standalone
        ``hierarchical_verification``), this is the **final padded width** ``K_final``
        (``hv_max_spec_len()`` for the standalone method)—**not** the per-round chunk
        length L. Scheduler lookahead, ``GPUModelRunner.num_spec_tokens``, and hybrid
        bundle sanitization all use this value.
        """
        if self.method == "hierarchical_verification":
            return self.hv_max_spec_len()
        if (
            self.method == "adaptive_spechive"
            and self.adaptive_spechive_mode == "hierarchical_verification"
        ):
            chunk_len = self.num_speculative_tokens
            rounds = self.adaptive_spechive_num_rounds
            return chunk_len + rounds * (chunk_len + 1)
        if (
            self.method == "pivot"
            and self.get_pivot_runtime_mode().verification_pipeline
            in ("intermediate_then_target", "intermediate_tree_then_target_tree")
        ):
            chunk_len = self.num_speculative_tokens
            rounds = self.pivot_spechive_num_rounds
            return chunk_len + rounds * (chunk_len + 1)
        return self.num_speculative_tokens

    def runner_num_partial_speculative_tokens(self) -> int:
        """Budget axis for **inner-round draft proposals**, not final target width.

        For hierarchical D→I rounds this is ``R * L`` (number of inner rounds times
        chunk length): the cumulative draft-proposal token count across intermediate
        verification rounds **before** the tail and **before** the single authoritative
        target pass. This differs from ``runner_num_speculative_tokens()`` which is
        ``K_final`` for the same configs—do not mix the two in scheduler or metrics code.
        """
        if self.method == "hierarchical_verification":
            return int(self.num_hv_rounds) * int(self.num_speculative_tokens)
        if (
            self.method == "adaptive_spechive"
            and self.adaptive_spechive_mode == "hierarchical_verification"
        ):
            return self.adaptive_spechive_num_rounds * self.num_speculative_tokens
        if (
            self.method == "pivot"
            and self.get_pivot_runtime_mode().verification_pipeline
            in ("intermediate_then_target", "intermediate_tree_then_target_tree")
        ):
            return self.pivot_spechive_num_rounds * self.num_speculative_tokens
        return 0

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "mtp")

    def uses_draft_model(self) -> bool:
        if self.method == "pivot":
            return self.get_pivot_runtime_mode().proposal_engine == "draft_model"
        return self.method in ("draft_model", "adaptive_spechive", "hierarchical_verification")

    def uses_extract_hidden_states(self) -> bool:
        return self.method == "extract_hidden_states"

    def uses_model_based_drafter(self) -> bool:
        if self.method == "pivot":
            return True
        return self.use_eagle() or self.uses_draft_model()

    def uses_gpu_sampled_tokens_for_drafting(self) -> bool:
        if self.method == "pivot":
            return True
        return (
            self.use_eagle()
            or self.uses_draft_model()
            or self.uses_extract_hidden_states()
        )

    def adaptive_spechive_draft_uses_eagle3_head(self) -> bool:
        """True when adaptive_spechive uses an Eagle3 head as draft (staged spechive fast path)."""
        if self.method != "adaptive_spechive" or self.draft_model_config is None:
            return False
        if getattr(self.draft_model_config.hf_config, "method", None) == "eagle3":
            return True
        return "eagle3" in str(self.draft_model_config.model).lower()

    def requires_aux_hidden_state_outputs(self) -> bool:
        if self.method in ("eagle3", "extract_hidden_states"):
            return True
        if self.method == "pivot":
            return self.get_pivot_runtime_mode().proposal_engine == "eagle3_head"
        if self.method == "adaptive_spechive":
            return self.adaptive_spechive_draft_uses_eagle3_head()
        return False

    def __repr__(self) -> str:
        method = self.method
        model = (
            None
            if method in ("ngram", "suffix", "extract_hidden_states")
            else self.draft_model_config.model
        )
        num_spec_tokens = self.num_speculative_tokens
        if method == "adaptive_spechive":
            im = self.intermediate_model
            acm = self.adaptive_spechive_mode
            rounds = self.adaptive_spechive_num_rounds
            return f"SpeculativeConfig({method=}, {model=}, {im=}, {acm=}, {num_spec_tokens=}, {rounds=})"
        if method == "hierarchical_verification":
            im = self.intermediate_model
            hv_r = self.num_hv_rounds
            return f"SpeculativeConfig({method=}, {model=}, {im=}, {num_spec_tokens=}, num_hv_rounds={hv_r})"
        if method == "pivot":
            mode = self.get_pivot_runtime_mode()
            im = self.intermediate_model
            topk = self.pivot_topk_selection
            pct = self.pivot_expansion_pct
            min_b = self.pivot_min_batch_for_expansion
            psp = self.pivot_spechive
            rounds = self.pivot_spechive_num_rounds
            use_tree = self.pivot_use_eagle_tree
            return (
                "SpeculativeConfig("
                f"{method=}, {model=}, {im=}, {num_spec_tokens=}, "
                f"{topk=}, {pct=}, {min_b=}, {psp=}, {rounds=}, "
                f"{use_tree=}, "
                f"proposal_engine={mode.proposal_engine}, "
                f"verification_pipeline={mode.verification_pipeline}, "
                f"hidden_state_source={mode.hidden_state_source})"
            )
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"
