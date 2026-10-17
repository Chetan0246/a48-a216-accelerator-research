"""Decode phase memory-bandwidth ceiling and token throughput modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeCeiling:
    sustained_bandwidth_gb_s: float
    weight_traffic_gb: float
    kv_traffic_per_sequence_gb: float
    kv_traffic_batch_gb: float
    total_step_traffic_gb: float
    aggregate_tps: float
    per_sequence_tps: float
    limiting_factor: str
    card_cache_traffic_gb: float = 0.0


def calculate_decode_bandwidth_ceiling(
    peak_bandwidth_tb_s: float,
    effective_bandwidth_fraction: float,
    weight_traffic_gb: float,
    kv_cache_per_sequence_gb: float,
    batch_size: int,
    step_cache_traffic_gb: float | None = None,
) -> DecodeCeiling:
    """Calculates decode memory-bandwidth throughput ceilings.

    Traffic semantics:
        - Weight reuse: Shared model weights are read once per batch step (weight_traffic_gb).
        - KV cache: Each sequence in the batch reads its KV context per step (batch_size * kv_cache_per_sequence_gb).
        - Total step traffic: weight_traffic_gb + batch_size * kv_cache_per_sequence_gb.

    Throughput equations:
        sustained_bw_gb_s = peak_bandwidth_tb_s * 1000.0 * effective_bandwidth_fraction
        aggregate_tps     = (sustained_bw_gb_s * batch_size) / total_step_traffic_gb
        per_sequence_tps  = aggregate_tps / batch_size

    Args:
        peak_bandwidth_tb_s: Theoretical peak memory bandwidth in TB/s.
        effective_bandwidth_fraction: Achievable sustained fraction of peak bandwidth (0, 1.0].
        weight_traffic_gb: Weights read per decode step in decimal GB.
        kv_cache_per_sequence_gb: KV cache read per sequence in decimal GB.
        batch_size: Number of concurrent generation sequences.

    Returns:
        DecodeCeiling containing aggregate and per-sequence tokens/s.
    """
    for name, val in [
        ("peak_bandwidth_tb_s", peak_bandwidth_tb_s),
        ("effective_bandwidth_fraction", effective_bandwidth_fraction),
        ("weight_traffic_gb", weight_traffic_gb),
        ("kv_cache_per_sequence_gb", kv_cache_per_sequence_gb),
        ("batch_size", batch_size),
    ]:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val}")

    if peak_bandwidth_tb_s <= 0:
        raise ValueError("peak_bandwidth_tb_s must be positive")
    if not (0.0 < effective_bandwidth_fraction <= 1.0):
        raise ValueError("effective_bandwidth_fraction must be in (0, 1.0]")
    if weight_traffic_gb < 0 or kv_cache_per_sequence_gb < 0:
        raise ValueError("weight_traffic_gb and kv_cache_per_sequence_gb cannot be negative")
    if step_cache_traffic_gb is not None and (
        not math.isfinite(step_cache_traffic_gb) or step_cache_traffic_gb < 0
    ):
        raise ValueError("step_cache_traffic_gb must be a non-negative finite number")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    sustained_bw_gb_s = peak_bandwidth_tb_s * 1000.0 * effective_bandwidth_fraction
    kv_batch_gb = kv_cache_per_sequence_gb * batch_size
    card_cache_traffic_gb = step_cache_traffic_gb if step_cache_traffic_gb is not None else kv_batch_gb
    step_traffic_gb = weight_traffic_gb + card_cache_traffic_gb

    if step_traffic_gb <= 0:
        return DecodeCeiling(
            sustained_bandwidth_gb_s=sustained_bw_gb_s,
            weight_traffic_gb=weight_traffic_gb,
            kv_traffic_per_sequence_gb=kv_cache_per_sequence_gb,
            kv_traffic_batch_gb=kv_batch_gb,
            total_step_traffic_gb=0.0,
            aggregate_tps=math.inf,
            per_sequence_tps=math.inf,
            limiting_factor="zero traffic",
            card_cache_traffic_gb=0.0,
        )

    aggregate_tps = (sustained_bw_gb_s * batch_size) / step_traffic_gb
    per_seq_tps = aggregate_tps / batch_size

    return DecodeCeiling(
        sustained_bandwidth_gb_s=sustained_bw_gb_s,
        weight_traffic_gb=weight_traffic_gb,
        kv_traffic_per_sequence_gb=kv_cache_per_sequence_gb,
        kv_traffic_batch_gb=kv_batch_gb,
        total_step_traffic_gb=step_traffic_gb,
        aggregate_tps=aggregate_tps,
        per_sequence_tps=per_seq_tps,
        limiting_factor="on-card memory bandwidth",
        card_cache_traffic_gb=card_cache_traffic_gb,
    )
