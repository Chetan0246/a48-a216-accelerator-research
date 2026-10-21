"""Mixture-of-Experts (MoE) analytical expert-union and host-offload modeling.

IMPORTANT MODELING NOTICE:
    This module implements an analytical first-order throughput bound, NOT a validated
    MoE performance predictor.

Assumptions and architectural boundaries:
    1. Equal expert sizes: Routed expert parameters are assumed distributed uniformly across experts.
    2. Explicit expert selection distribution: Tokens select top-k distinct experts either uniformly
       or according to an explicit validated probability distribution.
    3. Correct routing-skew semantics: Skew concentrates tokens on fewer popular experts, which
       reduces the distinct active expert union (E[U_skew] <= E[U_uniform]). Load imbalance affects
       per-expert compute queue depths, not the distinct set of weights streamed.
    4. Two offload execution bounds:
       - Overlapped (optimistic / pipelined upper bound): min(card_tps, host_tps).
       - Serialized (conservative non-overlapped bound): B / (t_card + t_host).
    5. Fixed residency: A fixed subset of experts is resident in accelerator memory.
    6. Resident shared weights: All attention, embedding, and shared MLP parameters reside on-chip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .capacity import calculate_stored_weights_gb


def calculate_expected_expert_union(
    expert_count: int,
    top_k: int,
    batch_size: int,
    expert_inclusion_probs: list[float] | None = None,
) -> float:
    """Calculates expected distinct experts activated across a batch.

    Mathematical formulation:
    - Under uniform independent routing:
      For any expert, probability of NOT being selected by one token is (1 - top_k / expert_count).
      Across a batch of B independent tokens, probability of never being selected is (1 - top_k / expert_count)^B.
      Expected unique experts: expert_count * [1.0 - (1.0 - top_k / expert_count)^B].

    - Under explicit marginal per-token expert inclusion probabilities (expert_inclusion_probs = [q_1, ..., q_E]):
      Each entry q_i in [0.0, 1.0] represents the marginal probability that one token selects expert i.
      Since each token selects top_k experts in total, the sum of inclusion probabilities must satisfy:
          sum_{i=1}^E q_i == top_k
      Assuming independence between tokens across a batch of B tokens, the probability that expert i is never
      selected by any token in the batch is (1.0 - q_i)^B.
      The expected number of distinct activated experts across the batch is:
          E[U] = sum_{i=1}^E [1.0 - (1.0 - q_i)^B]

    Args:
        expert_count: Total number of routed experts (E).
        top_k: Number of experts activated per token (k).
        batch_size: Number of concurrent tokens/sequences in the batch (B).
        expert_inclusion_probs: Optional list of marginal per-token inclusion probabilities q_i,
            where each q_i in [0.0, 1.0] and sum(q_i) == top_k.

    Returns:
        Expected number of distinct active experts.
    """
    for name, val in [("expert_count", expert_count), ("top_k", top_k), ("batch_size", batch_size)]:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val}")

    if expert_count <= 0:
        raise ValueError("expert_count must be positive")
    if top_k <= 0 or top_k > expert_count:
        raise ValueError("top_k must satisfy 1 <= top_k <= expert_count")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if expert_inclusion_probs is None:
        prob_not_selected_single = 1.0 - (top_k / float(expert_count))
        prob_not_selected_batch = prob_not_selected_single ** batch_size
        return float(expert_count * (1.0 - prob_not_selected_batch))

    # Explicit marginal inclusion probabilities validation
    if len(expert_inclusion_probs) != expert_count:
        raise ValueError(
            f"expert_inclusion_probs length ({len(expert_inclusion_probs)}) must match expert_count ({expert_count})"
        )
    for idx, q in enumerate(expert_inclusion_probs):
        if not math.isfinite(q):
            raise ValueError(f"expert_inclusion_probs[{idx}] must be a finite number, got {q}")
        if not (0.0 <= q <= 1.0):
            raise ValueError(
                f"expert_inclusion_probs[{idx}] must be between 0 and 1, got {q}"
            )

    sum_q = sum(expert_inclusion_probs)
    if abs(sum_q - float(top_k)) > 1e-4:
        raise ValueError(
            f"expert_inclusion_probs must sum to top_k ({top_k}), got {sum_q:.6f}"
        )

    expected_union = sum(1.0 - (1.0 - q) ** batch_size for q in expert_inclusion_probs)
    return float(expected_union)


@dataclass(frozen=True)
class MoEOffloadResult:
    total_experts: int
    resident_experts: int
    nonresident_experts: int
    top_k: int
    batch_size: int
    expected_total_union: float
    expected_resident_union: float
    expected_nonresident_union: float
    card_weight_traffic_gb: float
    host_weight_traffic_gb: float
    card_step_time_ms: float
    host_step_time_ms: float
    card_aggregate_tps: float
    host_aggregate_tps: float
    overlapped_aggregate_tps: float
    serialized_aggregate_tps: float
    combined_aggregate_tps: float
    per_sequence_tps: float
    offload_mode: str
    limiting_factor: str
    card_cache_traffic_gb: float = 0.0
    streamed_shared_params_b: float | None = None
    backbone_weight_traffic_gb: float = 0.0
    output_head_traffic_gb: float = 0.0
    streamed_shared_traffic_gb: float = 0.0
    routed_weight_traffic_gb: float = 0.0
    embedding_lookup_traffic_gb: float = 0.0
    embedding_precision_bits: float = 16.0
    vision_weight_traffic_gb: float = 0.0
    execute_vision_encoder: bool = False
    mode: str = "text_only"
    model_type: str = "first_order_analytical_bound"
    is_validated_predictor: bool = False
    assumptions_note: str = (
        "First-order analytical bound under explicit residency and transfer assumptions; "
        "not a cycle-accurate or validated end-to-end performance predictor."
    )


def evaluate_moe_decode(
    total_experts: int,
    resident_experts: int,
    top_k: int,
    shared_params_b: float,
    total_params_b: float,
    bits_per_param: float,
    batch_size: int,
    kv_cache_per_sequence_gb: float,
    card_sustained_bw_tb_s: float,
    host_bw_tb_s: float = 0.0,
    host_latency_ms: float = 0.0,
    expert_inclusion_probs: list[float] | None = None,
    offload_mode: str = "serialized",
    step_cache_traffic_gb: float | None = None,
    streamed_shared_params_b: float | None = None,
    shared_backbone_params_b: float | None = None,
    embedding_params_b: float | None = None,
    output_head_params_b: float | None = None,
    vision_params_b: float | None = None,
    hidden_size: int | None = None,
    mode: str = "text_only",
    execute_vision_encoder: bool = False,
    embedding_precision_bits: float = 16.0,
) -> MoEOffloadResult:
    """Evaluates MoE decode throughput ceiling with on-card and host offloading.

    CAUTION:
        This returns a first-order analytical bound, not a cycle-accurate or validated
        end-to-end execution prediction.

    Bounds:
        - Overlapped mode: min(card_tps, host_tps) assuming perfect pipelining / DMA prefetch.
        - Serialized mode: B / (t_card + t_host) assuming non-overlapped execution.

    Args:
        total_experts: Total routed experts in model.
        resident_experts: Experts stored in accelerator memory (0 <= resident <= total).
        top_k: Experts activated per token.
        shared_params_b: Billions of shared (non-expert) parameters.
        total_params_b: Billions of total model parameters.
        bits_per_param: Weight precision including quantization metadata.
        batch_size: Concurrency / batch size.
        kv_cache_per_sequence_gb: KV cache read per sequence in decimal GB.
        card_sustained_bw_tb_s: Accelerator sustained memory bandwidth in TB/s.
        host_bw_tb_s: Effective host offload bandwidth in TB/s (0 disables host throughput model).
        host_latency_ms: Fixed per-decode-step host offload dispatch latency in ms.
        expert_inclusion_probs: Optional marginal inclusion probability distribution across experts (summing to top_k).
        offload_mode: 'serialized' (conservative) or 'overlapped' (optimistic ceiling).
        step_cache_traffic_gb: Optional total cache step traffic in decimal GB.
        streamed_shared_params_b: Optional shared weights streamed every decode step (text backbone + output LM head).
        shared_backbone_params_b: Optional text-backbone shared parameters in billions.
        embedding_params_b: Optional input embedding table parameters in billions.
        output_head_params_b: Optional output LM head projection parameters in billions.
        vision_params_b: Optional vision tower / multimodal parameters in billions.
        hidden_size: Optional transformer hidden dimension for embedding lookup traffic modeling.
        mode: Execution mode ('text_only' or 'multimodal').
        execute_vision_encoder: Explicitly indicates whether the vision tower runs during this step (default False).
        embedding_precision_bits: Explicit precision in bits used for embedding table row lookup (default 16.0).

    Returns:
        MoEOffloadResult with complete breakdown and ceilings for both execution bounds.
    """
    for name, val in [
        ("total_experts", total_experts),
        ("resident_experts", resident_experts),
        ("top_k", top_k),
        ("shared_params_b", shared_params_b),
        ("total_params_b", total_params_b),
        ("bits_per_param", bits_per_param),
        ("batch_size", batch_size),
        ("kv_cache_per_sequence_gb", kv_cache_per_sequence_gb),
        ("card_sustained_bw_tb_s", card_sustained_bw_tb_s),
        ("host_bw_tb_s", host_bw_tb_s),
        ("host_latency_ms", host_latency_ms),
        ("embedding_precision_bits", embedding_precision_bits),
    ]:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val}")

    if total_params_b <= 0 or bits_per_param <= 0:
        raise ValueError("total_params_b and bits_per_param must be positive")
    if shared_params_b < 0 or shared_params_b >= total_params_b:
        raise ValueError("shared_params_b must be >= 0 and < total_params_b")
    if not (0 <= resident_experts <= total_experts):
        raise ValueError("resident_experts must satisfy 0 <= resident_experts <= total_experts")
    if card_sustained_bw_tb_s <= 0:
        raise ValueError("card_sustained_bw_tb_s must be positive")
    if host_bw_tb_s < 0 or host_latency_ms < 0:
        raise ValueError("host_bw_tb_s and host_latency_ms cannot be negative")
    if kv_cache_per_sequence_gb < 0:
        raise ValueError("kv_cache_per_sequence_gb cannot be negative")
    if step_cache_traffic_gb is not None and (
        not math.isfinite(step_cache_traffic_gb) or step_cache_traffic_gb < 0
    ):
        raise ValueError("step_cache_traffic_gb must be a non-negative finite number")
    if embedding_precision_bits <= 0:
        raise ValueError("embedding_precision_bits must be positive")
    valid_modes = ("serialized", "overlapped")
    if offload_mode not in valid_modes:
        raise ValueError(f"offload_mode must be one of {valid_modes}, got '{offload_mode}'")
    valid_exec_modes = ("text_only", "multimodal")
    if mode not in valid_exec_modes:
        raise ValueError(f"mode must be one of {valid_exec_modes}, got '{mode}'")

    for opt_name, opt_val in [
        ("streamed_shared_params_b", streamed_shared_params_b),
        ("shared_backbone_params_b", shared_backbone_params_b),
        ("embedding_params_b", embedding_params_b),
        ("output_head_params_b", output_head_params_b),
        ("vision_params_b", vision_params_b),
    ]:
        if opt_val is not None:
            if not math.isfinite(opt_val):
                raise ValueError(f"{opt_name} must be a finite number, got {opt_val}")
            if opt_val < 0:
                raise ValueError(f"{opt_name} cannot be negative")

    if hidden_size is not None and (not math.isfinite(hidden_size) or hidden_size <= 0):
        raise ValueError("hidden_size must be a positive integer")

    routed_total_b = total_params_b - shared_params_b
    expert_b = routed_total_b / float(total_experts)
    nonresident_experts = total_experts - resident_experts

    # Calculate expected expert unions
    if expert_inclusion_probs is None:
        expected_total_union = calculate_expected_expert_union(
            expert_count=total_experts,
            top_k=top_k,
            batch_size=batch_size,
        )
        inclusion_ratio = expected_total_union / float(total_experts)
        expected_resident_union = float(resident_experts) * inclusion_ratio
        expected_nonresident_union = float(nonresident_experts) * inclusion_ratio
    else:
        # Validate and compute using exact per-expert inclusion probabilities
        calculate_expected_expert_union(
            expert_count=total_experts,
            top_k=top_k,
            batch_size=batch_size,
            expert_inclusion_probs=expert_inclusion_probs,
        )
        expected_resident_union = float(
            sum(1.0 - (1.0 - q) ** batch_size for q in expert_inclusion_probs[:resident_experts])
        )
        expected_nonresident_union = float(
            sum(1.0 - (1.0 - q) ** batch_size for q in expert_inclusion_probs[resident_experts:])
        )
        expected_total_union = expected_resident_union + expected_nonresident_union

    # Determine streamed shared parameters
    if streamed_shared_params_b is not None:
        effective_streamed_shared_b = streamed_shared_params_b
    elif shared_backbone_params_b is not None:
        effective_streamed_shared_b = shared_backbone_params_b + (output_head_params_b or 0.0)
    else:
        effective_streamed_shared_b = shared_params_b

    # Mutually exclusive component calculations
    if shared_backbone_params_b is not None:
        backbone_weight_traffic_gb = calculate_stored_weights_gb(shared_backbone_params_b, bits_per_param)
        output_head_traffic_gb = (
            calculate_stored_weights_gb(output_head_params_b, bits_per_param)
            if output_head_params_b is not None
            else 0.0
        )
    else:
        backbone_weight_traffic_gb = calculate_stored_weights_gb(effective_streamed_shared_b, bits_per_param)
        output_head_traffic_gb = 0.0

    streamed_shared_traffic_gb = calculate_stored_weights_gb(effective_streamed_shared_b, bits_per_param)

    # Active routed expert traffic on card
    card_routed_params_b = expected_resident_union * expert_b
    routed_weight_traffic_gb = calculate_stored_weights_gb(card_routed_params_b, bits_per_param)

    # Embedding lookup traffic: in text decode, only 1 token per sequence is fetched
    if hidden_size is not None:
        lookup_dim = hidden_size
    elif embedding_params_b is not None and embedding_params_b > 0:
        lookup_dim = 2048
    else:
        lookup_dim = 0

    if lookup_dim > 0 and (embedding_params_b is not None or streamed_shared_params_b is not None):
        bytes_per_dim = embedding_precision_bits / 8.0
        embedding_lookup_bytes = batch_size * lookup_dim * bytes_per_dim
        embedding_lookup_traffic_gb = embedding_lookup_bytes / 1e9
    else:
        embedding_lookup_traffic_gb = 0.0

    # Vision weights traffic:
    # Autoregressive decoding generates tokens from KV cache and text backbone;
    # it does NOT run the vision encoder tower unless explicitly requested for that step.
    if execute_vision_encoder and vision_params_b is not None and vision_params_b > 0:
        vision_weight_traffic_gb = calculate_stored_weights_gb(vision_params_b, bits_per_param)
    else:
        vision_weight_traffic_gb = 0.0

    # Total card weight traffic is the exact sum of the mutually exclusive components:
    card_weight_traffic_gb = (
        backbone_weight_traffic_gb
        + output_head_traffic_gb
        + routed_weight_traffic_gb
        + embedding_lookup_traffic_gb
        + vision_weight_traffic_gb
    )
    card_cache_traffic_gb = (
        step_cache_traffic_gb if step_cache_traffic_gb is not None else (kv_cache_per_sequence_gb * batch_size)
    )
    card_step_traffic_gb = card_weight_traffic_gb + card_cache_traffic_gb
    card_sustained_bw_gb_s = card_sustained_bw_tb_s * 1000.0
    card_step_time_s = card_step_traffic_gb / card_sustained_bw_gb_s if card_sustained_bw_gb_s > 0 else math.inf
    card_tps = batch_size / card_step_time_s if card_step_time_s > 0 else 0.0

    # Host offload transfer traffic & step time
    if nonresident_experts > 0:
        host_active_params_b = expected_nonresident_union * expert_b
        host_weight_traffic_gb = calculate_stored_weights_gb(host_active_params_b, bits_per_param)
        if host_bw_tb_s > 0 and host_weight_traffic_gb > 0:
            transfer_s = host_weight_traffic_gb / (host_bw_tb_s * 1000.0)
            latency_s = host_latency_ms / 1000.0
            host_step_time_s = transfer_s + latency_s
            host_tps = batch_size / host_step_time_s if host_step_time_s > 0 else 0.0
        else:
            host_step_time_s = math.inf
            host_tps = 0.0
    else:
        host_weight_traffic_gb = 0.0
        host_step_time_s = 0.0
        host_tps = math.inf

    # Overlapped bound: min(card_tps, host_tps)
    if host_tps == 0.0:
        overlapped_tps = 0.0
    else:
        overlapped_tps = min(card_tps, host_tps)

    # Serialized bound: batch_size / (card_step_time_s + host_step_time_s)
    if math.isinf(host_step_time_s) or host_tps == 0.0:
        serialized_tps = 0.0
    else:
        total_serial_time_s = card_step_time_s + host_step_time_s
        serialized_tps = batch_size / total_serial_time_s if total_serial_time_s > 0 else 0.0

    # Combined throughput selected according to offload_mode
    combined_tps = serialized_tps if offload_mode == "serialized" else overlapped_tps

    # Limiting factor determination
    if nonresident_experts > 0 and (host_bw_tb_s == 0.0 or math.isinf(host_step_time_s)):
        limiting = "host offload unspecified"
    elif offload_mode == "serialized":
        limiting = "serialized on-card and host transfer" if nonresident_experts > 0 else "on-card memory bandwidth"
    else:  # overlapped
        limiting = "host offload link" if host_tps < card_tps else "on-card memory bandwidth"

    per_seq_tps = combined_tps / batch_size if combined_tps > 0 else 0.0

    return MoEOffloadResult(
        total_experts=total_experts,
        resident_experts=resident_experts,
        nonresident_experts=nonresident_experts,
        top_k=top_k,
        batch_size=batch_size,
        expected_total_union=expected_total_union,
        expected_resident_union=expected_resident_union,
        expected_nonresident_union=expected_nonresident_union,
        card_weight_traffic_gb=card_weight_traffic_gb,
        host_weight_traffic_gb=host_weight_traffic_gb,
        card_step_time_ms=card_step_time_s * 1000.0 if not math.isinf(card_step_time_s) else math.inf,
        host_step_time_ms=host_step_time_s * 1000.0 if not math.isinf(host_step_time_s) else math.inf,
        card_aggregate_tps=card_tps,
        host_aggregate_tps=host_tps,
        overlapped_aggregate_tps=overlapped_tps,
        serialized_aggregate_tps=serialized_tps,
        combined_aggregate_tps=combined_tps,
        per_sequence_tps=per_seq_tps,
        offload_mode=offload_mode,
        limiting_factor=limiting,
        card_cache_traffic_gb=card_cache_traffic_gb,
        streamed_shared_params_b=effective_streamed_shared_b,
        backbone_weight_traffic_gb=backbone_weight_traffic_gb,
        output_head_traffic_gb=output_head_traffic_gb,
        streamed_shared_traffic_gb=streamed_shared_traffic_gb,
        routed_weight_traffic_gb=routed_weight_traffic_gb,
        embedding_lookup_traffic_gb=embedding_lookup_traffic_gb,
        embedding_precision_bits=embedding_precision_bits,
        vision_weight_traffic_gb=vision_weight_traffic_gb,
        execute_vision_encoder=execute_vision_encoder,
        mode=mode,
        model_type="first_order_analytical_bound",
        is_validated_predictor=False,
        assumptions_note=(
            "First-order analytical bound under explicit residency and transfer assumptions; "
            "not a cycle-accurate or validated end-to-end performance predictor."
        ),
    )

