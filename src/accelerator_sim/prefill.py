"""Prefill (prompt processing) compute requirements and latency estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillComputeBreakdown:
    prompt_tokens: int
    batch_size: int
    mode: str
    shared_backbone_tflop: float
    routed_expert_tflop: float
    output_head_tflop: float
    embedding_lookup_tflop: float  # 0.0 (sparse index lookup, not dense GEMM)
    vision_tflop: float  # 0.0 in text_only, or dense GEMM if multimodal
    total_tflop: float
    active_compute_params_b: float
    logit_positions: int = 1


@dataclass(frozen=True)
class VisionEncodingStageResult:
    vision_params_b: float
    image_tokens_per_seq: int
    batch_size: int
    total_image_tokens: int
    bits_per_param: float
    vision_weight_traffic_gb: float
    vision_compute_tflop: float


def calculate_vision_encoding_stage(
    vision_params_b: float,
    image_tokens_per_seq: int,
    bits_per_param: float,
    batch_size: int = 1,
) -> VisionEncodingStageResult:
    """Calculates weight traffic and compute work for the one-time vision encoding stage.

    The vision tower encodes input image patches once per prompt sequence, producing
    image token representations. It does NOT run during autoregressive token decode.
    """
    for name, val in [
        ("vision_params_b", vision_params_b),
        ("image_tokens_per_seq", image_tokens_per_seq),
        ("bits_per_param", bits_per_param),
        ("batch_size", batch_size),
    ]:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val}")

    if vision_params_b < 0:
        raise ValueError("vision_params_b cannot be negative")
    if image_tokens_per_seq < 0:
        raise ValueError("image_tokens_per_seq cannot be negative")
    if bits_per_param <= 0:
        raise ValueError("bits_per_param must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    from .capacity import calculate_stored_weights_gb

    traffic_gb = calculate_stored_weights_gb(vision_params_b, bits_per_param) if vision_params_b > 0 else 0.0
    compute_tflop = (2.0 * vision_params_b * image_tokens_per_seq * batch_size) / 1000.0

    return VisionEncodingStageResult(
        vision_params_b=vision_params_b,
        image_tokens_per_seq=image_tokens_per_seq,
        batch_size=batch_size,
        total_image_tokens=image_tokens_per_seq * batch_size,
        bits_per_param=bits_per_param,
        vision_weight_traffic_gb=traffic_gb,
        vision_compute_tflop=compute_tflop,
    )


@dataclass(frozen=True)
class PrefillEstimate:
    params_b: float
    prompt_tokens_per_seq: int
    batch_size: int
    total_prompt_tokens: int
    dense_equivalent_tflop: float
    target_ttft_seconds: float
    required_effective_tflops: float
    peak_tflops: float | None = None
    compute_utilization: float | None = None
    idealized_ttft_seconds: float | None = None
    mode: str = "text_only"
    breakdown: PrefillComputeBreakdown | None = None


def calculate_prefill_compute_breakdown(
    prompt_tokens: int,
    batch_size: int,
    shared_backbone_params_b: float,
    routed_expert_params_b: float,
    num_experts: int,
    experts_per_token: int,
    output_head_params_b: float = 0.0,
    embedding_params_b: float = 0.0,
    vision_params_b: float = 0.0,
    mode: str = "text_only",
    vision_tokens_per_seq: int = 0,
    logit_positions: int = 1,
) -> PrefillComputeBreakdown:
    """Calculates granular prefill FLOP breakdown across parameter categories.

    Explicit accounting:
        - Text backbone shared weights: Dense GEMMs across prompt length.
        - Active routed experts: k * (routed_params / E) dense GEMMs across prompt length.
        - Output LM head: Dense projection to vocabulary for logit_positions tokens
          (e.g., logit_positions=1 for next-token TTFT prediction matching Qwen logits_to_keep=1;
           logit_positions=prompt_tokens for full sequence logits).
        - Input embedding table: Sparse index lookup (0.0 dense matrix FLOPs).
        - Vision tower: Multimodal encoder GEMMs only charged when mode == 'multimodal'.

    Args:
        prompt_tokens: Text prompt tokens per sequence.
        batch_size: Concurrency / batch size.
        shared_backbone_params_b: Text-backbone shared parameters in billions.
        routed_expert_params_b: Total routed expert parameters in billions.
        num_experts: Total expert count in MoE.
        experts_per_token: Top-k experts activated per token.
        output_head_params_b: Output LM head parameters in billions.
        embedding_params_b: Input embedding table parameters in billions.
        vision_params_b: Vision tower parameters in billions.
        mode: 'text_only' or 'multimodal'.
        vision_tokens_per_seq: Image patch tokens per sequence for multimodal prefill.
        logit_positions: Number of prompt positions for which output logits are computed (1 to prompt_tokens).

    Returns:
        PrefillComputeBreakdown detailing TFLOPs per category.
    """
    for name, val in [
        ("prompt_tokens", prompt_tokens),
        ("batch_size", batch_size),
        ("shared_backbone_params_b", shared_backbone_params_b),
        ("routed_expert_params_b", routed_expert_params_b),
        ("num_experts", num_experts),
        ("experts_per_token", experts_per_token),
        ("output_head_params_b", output_head_params_b),
        ("embedding_params_b", embedding_params_b),
        ("vision_params_b", vision_params_b),
        ("vision_tokens_per_seq", vision_tokens_per_seq),
        ("logit_positions", logit_positions),
    ]:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val}")

    if prompt_tokens <= 0 or batch_size <= 0:
        raise ValueError("prompt_tokens and batch_size must be positive")
    if not (1 <= logit_positions <= prompt_tokens):
        raise ValueError(
            f"logit_positions ({logit_positions}) must satisfy 1 <= logit_positions <= prompt_tokens ({prompt_tokens})"
        )
    if shared_backbone_params_b < 0 or routed_expert_params_b < 0:
        raise ValueError("shared_backbone_params_b and routed_expert_params_b cannot be negative")
    if num_experts < 1 or not (1 <= experts_per_token <= num_experts):
        raise ValueError("num_experts and experts_per_token must be valid")
    if output_head_params_b < 0 or embedding_params_b < 0 or vision_params_b < 0:
        raise ValueError("output_head_params_b, embedding_params_b, and vision_params_b cannot be negative")
    if vision_tokens_per_seq < 0:
        raise ValueError("vision_tokens_per_seq cannot be negative")
    if mode not in ("text_only", "multimodal"):
        raise ValueError(f"mode must be 'text_only' or 'multimodal', got '{mode}'")

    routed_active_params_b = experts_per_token * (routed_expert_params_b / float(num_experts))
    active_compute_params_b = shared_backbone_params_b + routed_active_params_b + output_head_params_b

    backbone_tflop = (2.0 * shared_backbone_params_b * prompt_tokens * batch_size) / 1000.0
    routed_tflop = (2.0 * routed_active_params_b * prompt_tokens * batch_size) / 1000.0
    head_tflop = (2.0 * output_head_params_b * logit_positions * batch_size) / 1000.0
    embedding_tflop = 0.0  # Sparse index lookup, not matrix multiplication

    if mode == "multimodal" and vision_params_b > 0 and vision_tokens_per_seq > 0:
        vision_tflop = (2.0 * vision_params_b * vision_tokens_per_seq * batch_size) / 1000.0
    else:
        vision_tflop = 0.0

    total_tflop = backbone_tflop + routed_tflop + head_tflop + embedding_tflop + vision_tflop

    return PrefillComputeBreakdown(
        prompt_tokens=prompt_tokens,
        batch_size=batch_size,
        mode=mode,
        shared_backbone_tflop=backbone_tflop,
        routed_expert_tflop=routed_tflop,
        output_head_tflop=head_tflop,
        embedding_lookup_tflop=embedding_tflop,
        vision_tflop=vision_tflop,
        total_tflop=total_tflop,
        active_compute_params_b=active_compute_params_b,
        logit_positions=logit_positions,
    )


def calculate_prefill_tflop(
    params_b: float,
    prompt_tokens: int,
    batch_size: int = 1,
) -> float:
    """Calculates first-order dense-equivalent prefill FLOPs in TFLOP (10^12 FLOPs).

    Formula:
        (2 * (params_b * 1e9) * prompt_tokens * batch_size) / 1e12
        = (2 * params_b * prompt_tokens * batch_size) / 1000.0

    Args:
        params_b: Model parameters (or active parameters for MoE) in billions.
        prompt_tokens: Prompt token length per sequence.
        batch_size: Number of concurrent prompt sequences.

    Returns:
        Total prefill FLOPs in TFLOP.
    """
    for name, val in [("params_b", params_b), ("prompt_tokens", prompt_tokens), ("batch_size", batch_size)]:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val}")

    if params_b <= 0 or prompt_tokens <= 0 or batch_size <= 0:
        raise ValueError("params_b, prompt_tokens, and batch_size must be positive")
    return (2.0 * params_b * prompt_tokens * batch_size) / 1000.0


def calculate_required_prefill_throughput(
    prefill_tflop: float,
    target_ttft_seconds: float,
) -> float:
    """Calculates required effective compute throughput in TFLOP/s for target TTFT budget.

    Formula:
        prefill_tflop / target_ttft_seconds

    Args:
        prefill_tflop: Total prefill work in TFLOP.
        target_ttft_seconds: Time budget in seconds to process the prompt.

    Returns:
        Required effective compute throughput in TFLOP/s.
    """
    for name, val in [("prefill_tflop", prefill_tflop), ("target_ttft_seconds", target_ttft_seconds)]:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val}")

    if prefill_tflop <= 0 or target_ttft_seconds <= 0:
        raise ValueError("prefill_tflop and target_ttft_seconds must be positive")
    return prefill_tflop / target_ttft_seconds


def estimate_prefill(
    params_b: float,
    prompt_tokens: int,
    target_ttft_seconds: float = 3.0,
    batch_size: int = 1,
    peak_tflops: float = 0.0,
    compute_utilization: float = 0.5,
    mode: str = "text_only",
    breakdown: PrefillComputeBreakdown | None = None,
) -> PrefillEstimate:
    """Estimates prefill compute work and required/idealized throughput.

    Note:
        This is a first-order analytical model based on dense matrix multiplications (2 * P * S * B).
        Attention quadratics, logits, batching overheads, and kernel scheduling are omitted.
    """
    for name, val in [
        ("params_b", params_b),
        ("prompt_tokens", prompt_tokens),
        ("target_ttft_seconds", target_ttft_seconds),
        ("batch_size", batch_size),
        ("peak_tflops", peak_tflops),
        ("compute_utilization", compute_utilization),
    ]:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val}")

    if peak_tflops < 0:
        raise ValueError("peak_tflops cannot be negative")
    if peak_tflops > 0 and not (0 < compute_utilization <= 1.0):
        raise ValueError("compute_utilization must be in (0, 1.0]")
    if mode not in ("text_only", "multimodal"):
        raise ValueError(f"mode must be 'text_only' or 'multimodal', got '{mode}'")

    tflop = breakdown.total_tflop if breakdown is not None else calculate_prefill_tflop(params_b, prompt_tokens, batch_size)
    req_tflops = calculate_required_prefill_throughput(tflop, target_ttft_seconds)

    idealized_sec = None
    if peak_tflops > 0:
        effective_compute = peak_tflops * compute_utilization
        idealized_sec = tflop / effective_compute

    return PrefillEstimate(
        params_b=params_b,
        prompt_tokens_per_seq=prompt_tokens,
        batch_size=batch_size,
        total_prompt_tokens=prompt_tokens * batch_size,
        dense_equivalent_tflop=tflop,
        target_ttft_seconds=target_ttft_seconds,
        required_effective_tflops=req_tflops,
        peak_tflops=peak_tflops if peak_tflops > 0 else None,
        compute_utilization=compute_utilization if peak_tflops > 0 else None,
        idealized_ttft_seconds=idealized_sec,
        mode=mode,
        breakdown=breakdown,
    )
