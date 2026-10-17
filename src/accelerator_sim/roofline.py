"""Roofline model implementation for AI accelerators."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RooflinePoint:
    arithmetic_intensity: float  # FLOPs / Byte
    achievable_tflops: float
    is_memory_bound: bool
    efficiency: float


class RooflineModel:
    """Analytical Roofline model for compute vs. memory bandwidth limits."""

    def __init__(self, peak_tflops: float, memory_bandwidth_gbs: float):
        """
        Args:
            peak_tflops: Theoretical peak throughput in TFLOPS (1e12 FLOPs/s).
            memory_bandwidth_gbs: Memory bandwidth in GB/s (1e9 Bytes/s).
        """
        self.peak_tflops = float(peak_tflops)
        self.memory_bandwidth_gbs = float(memory_bandwidth_gbs)
        # Ridge point in FLOPs/byte:
        # (peak_tflops * 1e12) / (memory_bandwidth_gbs * 1e9) = (peak_tflops / memory_bandwidth_gbs) * 1000
        self.ridge_point = (self.peak_tflops / self.memory_bandwidth_gbs) * 1000.0

    def evaluate(self, arithmetic_intensity: float) -> RooflinePoint:
        """Evaluate achievable performance given operational intensity."""
        if arithmetic_intensity <= 0:
            return RooflinePoint(0.0, 0.0, True, 0.0)

        # Performance limit from bandwidth: (intensity * BW) / 1000 in TFLOPS
        bw_limited_tflops = (arithmetic_intensity * self.memory_bandwidth_gbs) / 1000.0
        achievable_tflops = min(self.peak_tflops, bw_limited_tflops)
        is_mem_bound = arithmetic_intensity < self.ridge_point
        efficiency = achievable_tflops / self.peak_tflops if self.peak_tflops > 0 else 0.0

        return RooflinePoint(
            arithmetic_intensity=arithmetic_intensity,
            achievable_tflops=achievable_tflops,
            is_memory_bound=is_mem_bound,
            efficiency=efficiency,
        )

    def estimate_latency_ms(self, total_flops: float, total_bytes: float) -> float:
        """
        Estimate ideal execution time in milliseconds.
        Latency = max(total_flops / peak_flops_per_ms, total_bytes / peak_bytes_per_ms)
        """
        if total_flops <= 0 and total_bytes <= 0:
            return 0.0

        # peak_tflops * 1e12 / 1000 = peak FLOPs / ms
        compute_ms = (total_flops / (self.peak_tflops * 1e9)) if self.peak_tflops > 0 else float("inf")
        # memory_bandwidth_gbs * 1e9 / 1000 = bytes / ms
        memory_ms = (total_bytes / (self.memory_bandwidth_gbs * 1e6)) if self.memory_bandwidth_gbs > 0 else float("inf")

        return max(compute_ms, memory_ms)

    def summary(self) -> Dict[str, Any]:
        return {
            "peak_tflops": self.peak_tflops,
            "memory_bandwidth_gbs": self.memory_bandwidth_gbs,
            "ridge_point_flops_per_byte": round(self.ridge_point, 2),
        }
