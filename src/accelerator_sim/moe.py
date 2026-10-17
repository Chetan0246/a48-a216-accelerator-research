"""Mixture-of-Experts (MoE) analytical expert-union and host-offload modeling.

Assumptions and modeling contract:
    1. Equal expert sizes: Routed expert parameters are distributed equally across experts.
    2. Uniform independent routing: Tokens select top-k distinct experts independently.
    3. Fixed residency: A fixed subset of experts is resident in accelerator memory.
    4. Resident shared weights: All attention, embedding, and shared MLP parameters reside on-chip.
    5. Host offload tier: Non-resident expert weights are streamed across a host link per step.
    6. Analytical bound: This is a first-order throughput ceiling, not a cycle-accurate
       simulation of PCIe framing, DMA packetization, prefetch overlapping, or kernel jitter.
"""

import math
from dataclasses import dataclass
from .capacity import calculate_stored_weights_gb


def calculate_expected_expert_union(
    expert_count: int,
    top_k: int,
    batch_size: int,
) -> float:
    """Calculates expected distinct experts activated across a batch under uniform independent routing.

    Formula:
        For any expert, the probability of NOT being selected by one token is (1 - top_k / expert_count).
        Across a batch of B independent tokens, the probability of never being selected is (1 - top_k / expert_count)^B.
        Expected total active experts: expert_count * [1.0 - (1.0 - top_k / expert_count)^B].

    Args:
        expert_count: Total number of routed experts (E).
        top_k: Number of experts activated per token (k).
        batch_size: Number of concurrent tokens/sequences in the batch (B).

    Returns:
        Expected number of distinct active experts.
    """
    if expert_count <= 0:
        raise ValueError("expert_count must be positive")
    if top_k <= 0 or top_k > expert_count:
        raise ValueError("top_k must satisfy 1 <= top_k <= expert_count")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    prob_not_selected_single = 1.0 - (top_k / float(expert_count))
    prob_not_selected_batch = prob_not_selected_single ** batch_size
    return float(expert_count * (1.0 - prob_not_selected_batch))


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
    card_aggregate_tps: float
    host_aggregate_tps: float
    combined_aggregate_tps: float
    per_sequence_tps: float
    limiting_factor: str


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
) -> MoEOffloadResult:
    """Evaluates MoE decode throughput ceiling with on-card and host offloading.

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

    Returns:
        MoEOffloadResult with complete breakdown and ceilings.
    """
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

    routed_total_b = total_params_b - shared_params_b
    expert_b = routed_total_b / float(total_experts)

    # Calculate expected unions
    expected_total_union = calculate_expected_expert_union(total_experts, top_k, batch_size)
    per_expert_inclusion = 1.0 - (1.0 - (top_k / float(total_experts))) ** batch_size
    expected_resident_union = resident_experts * per_expert_inclusion
    nonresident_experts = total_experts - resident_experts
    expected_nonresident_union = nonresident_experts * per_expert_inclusion

    # On-card weight traffic
    card_active_params_b = shared_params_b + expected_resident_union * expert_b
    card_weight_traffic_gb = calculate_stored_weights_gb(card_active_params_b, bits_per_param)

    # On-card decode throughput
    card_step_traffic_gb = card_weight_traffic_gb + kv_cache_per_sequence_gb * batch_size
    card_sustained_bw_gb_s = card_sustained_bw_tb_s * 1000.0
    card_tps = (card_sustained_bw_gb_s * batch_size) / card_step_traffic_gb if card_step_traffic_gb > 0 else math.inf

    # Host offload throughput
    if nonresident_experts > 0:
        host_active_params_b = expected_nonresident_union * expert_b
        host_weight_traffic_gb = calculate_stored_weights_gb(host_active_params_b, bits_per_param)
        if host_bw_tb_s > 0 and host_weight_traffic_gb > 0:
            transfer_s = host_weight_traffic_gb / (host_bw_tb_s * 1000.0)
            latency_s = host_latency_ms / 1000.0
            host_tps = batch_size / (transfer_s + latency_s)
        else:
            host_tps = 0.0
    else:
        host_weight_traffic_gb = 0.0
        host_tps = math.inf

    # Combined throughput determination
    if host_tps == 0.0:
        combined_tps = 0.0
        limiting = "host offload unspecified"
    else:
        combined_tps = min(card_tps, host_tps)
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
        card_aggregate_tps=card_tps,
        host_aggregate_tps=host_tps,
        combined_aggregate_tps=combined_tps,
        per_sequence_tps=per_seq_tps,
        limiting_factor=limiting,
    )
