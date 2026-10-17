"""Prefill (prompt processing) compute requirements and latency estimation."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PrefillEstimate:
    params_b: float
    prompt_tokens_per_seq: int
    batch_size: int
    total_prompt_tokens: int
    dense_equivalent_tflop: float
    target_ttft_seconds: float
    required_effective_tflops: float
    peak_tflops: Optional[float] = None
    compute_utilization: Optional[float] = None
    idealized_ttft_seconds: Optional[float] = None


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
) -> PrefillEstimate:
    """Estimates prefill compute work and required/idealized throughput.

    Note:
        This is a first-order analytical model based on dense matrix multiplications (2 * P * S * B).
        Attention quadratics, logits, batching overheads, and kernel scheduling are omitted.
    """
    if peak_tflops < 0:
        raise ValueError("peak_tflops cannot be negative")
    if peak_tflops > 0 and not (0 < compute_utilization <= 1.0):
        raise ValueError("compute_utilization must be in (0, 1.0]")

    tflop = calculate_prefill_tflop(params_b, prompt_tokens, batch_size)
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
    )
