"""Data models and configurations for accelerator research simulation."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SystemConfig:
    key: str
    name: str
    memory_capacity_gb: float
    peak_bandwidth_tb_s: float
    memory_bandwidth_gbs: float
    effective_bandwidth_fraction: float = 0.8
    runtime_reserve_gb: float = 4.0
    sram_mib: float = 0.0
    topology: str = ""
    peak_fp16_tflops: float = 0.0
    peak_fp8_tflops: float = 0.0
    peak_fp4_tflops: float = 0.0

    def __post_init__(self):
        if self.memory_capacity_gb <= 0:
            raise ValueError(f"memory_capacity_gb ({self.memory_capacity_gb}) must be positive")
        if self.peak_bandwidth_tb_s <= 0 or self.memory_bandwidth_gbs <= 0:
            raise ValueError("Bandwidth specifications must be positive")
        if self.runtime_reserve_gb < 0:
            raise ValueError("runtime_reserve_gb cannot be negative")
        if self.runtime_reserve_gb >= self.memory_capacity_gb:
            raise ValueError(
                f"runtime_reserve_gb ({self.runtime_reserve_gb} GB) must be strictly less than "
                f"memory_capacity_gb ({self.memory_capacity_gb} GB)"
            )
        if not (0.0 < self.effective_bandwidth_fraction <= 1.0):
            raise ValueError("effective_bandwidth_fraction must be in (0, 1.0]")


@dataclass(frozen=True)
class WorkloadConfig:
    key: str
    name: str
    total_parameters_b: float
    active_parameters_b: float
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    is_moe: bool = False
    num_experts: int = 1
    experts_per_token: int = 1
    default_context_len: int = 2048
    default_gen_len: int = 256
    bits_per_param: float = 4.5

    def __post_init__(self):
        if self.total_parameters_b <= 0 or self.active_parameters_b <= 0:
            raise ValueError("Parameters count must be positive")
        if self.active_parameters_b > self.total_parameters_b:
            raise ValueError("active_parameters_b cannot exceed total_parameters_b")
        if self.num_layers <= 0 or self.hidden_size <= 0:
            raise ValueError("num_layers and hidden_size must be positive")
        if self.num_attention_heads <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("Attention head dimensions must be positive")
        if self.bits_per_param <= 0:
            raise ValueError("bits_per_param must be positive")
        if self.is_moe:
            if self.num_experts < 1:
                raise ValueError("num_experts must be >= 1 for MoE")
            if not (1 <= self.experts_per_token <= self.num_experts):
                raise ValueError(
                    f"experts_per_token ({self.experts_per_token}) must satisfy "
                    f"1 <= experts_per_token <= num_experts ({self.num_experts})"
                )
