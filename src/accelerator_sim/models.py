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
