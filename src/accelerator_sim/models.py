"""Data models and configurations for accelerator research simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

ROUNDING_TOLERANCE_B: float = 0.08  # Parameter reconciliation tolerance (80M params on billion-scale specs)


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
    description: str = ""
    peak_fp16_tflops: float = 0.0
    peak_fp8_tflops: float = 0.0
    peak_fp4_tflops: float = 0.0

    def __post_init__(self) -> None:
        numeric_fields = {
            "memory_capacity_gb": self.memory_capacity_gb,
            "peak_bandwidth_tb_s": self.peak_bandwidth_tb_s,
            "memory_bandwidth_gbs": self.memory_bandwidth_gbs,
            "effective_bandwidth_fraction": self.effective_bandwidth_fraction,
            "runtime_reserve_gb": self.runtime_reserve_gb,
            "sram_mib": self.sram_mib,
            "peak_fp16_tflops": self.peak_fp16_tflops,
            "peak_fp8_tflops": self.peak_fp8_tflops,
            "peak_fp4_tflops": self.peak_fp4_tflops,
        }
        for name, val in numeric_fields.items():
            if not math.isfinite(val):
                raise ValueError(f"{name} must be a finite number, got {val}")

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
    intermediate_size: int | None = None
    vocab_size: int | None = None
    is_moe: bool = False
    num_experts: int = 1
    experts_per_token: int = 1
    default_context_len: int = 2048
    default_gen_len: int = 256
    bits_per_param: float = 4.5
    attention_type: str = "gqa"
    shared_params_b: float | None = None
    routed_expert_params_b: float | None = None
    shared_backbone_params_b: float | None = None
    embedding_params_b: float | None = None
    output_head_params_b: float | None = None
    vision_params_b: float | None = None
    streamed_shared_params_b: float | None = None
    embedding_precision_bits: float = 16.0
    mtp_params_b: float = 0.0
    is_complete: bool = True
    status: str = "supported"
    hybrid_attention: dict[str, Any] | None = None
    notes: str = ""

    @property
    def effective_streamed_shared_params_b(self) -> float:
        """Returns the shared parameter count streamed per decode step.

        For models with detailed parameter breakdown, this includes only the text
        backbone and output LM head, strictly excluding the vision tower and avoiding
        dense streaming of the input embedding table.
        """
        if self.streamed_shared_params_b is not None:
            return self.streamed_shared_params_b
        if self.shared_backbone_params_b is not None:
            return self.shared_backbone_params_b + (self.output_head_params_b or 0.0)
        return self.shared_params_b if self.shared_params_b is not None else 0.0

    def __post_init__(self) -> None:
        numeric_fields = {
            "total_parameters_b": self.total_parameters_b,
            "active_parameters_b": self.active_parameters_b,
            "bits_per_param": self.bits_per_param,
            "embedding_precision_bits": self.embedding_precision_bits,
            "mtp_params_b": self.mtp_params_b,
        }
        for name, val in numeric_fields.items():
            if not math.isfinite(val):
                raise ValueError(f"{name} must be a finite number, got {val}")

        if self.total_parameters_b <= 0 or self.active_parameters_b <= 0:
            raise ValueError("Parameters count must be positive")
        if self.active_parameters_b > self.total_parameters_b:
            raise ValueError("active_parameters_b cannot exceed total_parameters_b")
        if self.num_layers <= 0 or self.hidden_size <= 0:
            raise ValueError("num_layers and hidden_size must be positive")
        if self.num_attention_heads <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("Attention head dimensions must be positive")
        if self.intermediate_size is not None and self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.bits_per_param <= 0:
            raise ValueError("bits_per_param must be positive")
        if self.embedding_precision_bits <= 0:
            raise ValueError("embedding_precision_bits must be positive")
        if self.mtp_params_b < 0:
            raise ValueError("mtp_params_b cannot be negative")

        if self.shared_params_b is not None:
            if not math.isfinite(self.shared_params_b):
                raise ValueError(f"shared_params_b must be finite, got {self.shared_params_b}")
            if not (0.0 <= self.shared_params_b < self.total_parameters_b):
                raise ValueError("shared_params_b must be >= 0 and < total_parameters_b")

        if self.routed_expert_params_b is not None:
            if not math.isfinite(self.routed_expert_params_b):
                raise ValueError(f"routed_expert_params_b must be finite, got {self.routed_expert_params_b}")
            if not (0.0 < self.routed_expert_params_b <= self.total_parameters_b):
                raise ValueError("routed_expert_params_b must be > 0 and <= total_parameters_b")

        # Validate detailed parameter categories if provided
        detailed_fields = {
            "shared_backbone_params_b": self.shared_backbone_params_b,
            "embedding_params_b": self.embedding_params_b,
            "output_head_params_b": self.output_head_params_b,
            "vision_params_b": self.vision_params_b,
            "streamed_shared_params_b": self.streamed_shared_params_b,
        }
        for name, val in detailed_fields.items():
            if val is not None:
                if not math.isfinite(val):
                    raise ValueError(f"{name} must be finite, got {val}")
                if val < 0.0 or val > self.total_parameters_b:
                    raise ValueError(f"{name} must satisfy 0.0 <= {name} <= total_parameters_b")

        # Check detailed category split against shared_params_b if detailed categories are given
        if self.shared_params_b is not None and self.shared_backbone_params_b is not None:
            detailed_sum = (
                self.shared_backbone_params_b
                + (self.embedding_params_b or 0.0)
                + (self.output_head_params_b or 0.0)
                + (self.vision_params_b or 0.0)
            )
            if abs(detailed_sum - self.shared_params_b) > ROUNDING_TOLERANCE_B:
                raise ValueError(
                    f"Inconsistent detailed parameter accounting for '{self.key}': "
                    f"shared_backbone ({self.shared_backbone_params_b}) + "
                    f"embedding ({self.embedding_params_b or 0.0}) + "
                    f"output_head ({self.output_head_params_b or 0.0}) + "
                    f"vision ({self.vision_params_b or 0.0}) = {detailed_sum:.4f} "
                    f"does not match shared_params_b ({self.shared_params_b}) "
                    f"within tolerance {ROUNDING_TOLERANCE_B}B."
                )

        if self.is_moe:
            if self.num_experts < 1:
                raise ValueError("num_experts must be >= 1 for MoE")
            if not (1 <= self.experts_per_token <= self.num_experts):
                raise ValueError(
                    f"experts_per_token ({self.experts_per_token}) must satisfy "
                    f"1 <= experts_per_token <= num_experts ({self.num_experts})"
                )

            # Complete MoE workloads require explicit parameter decomposition
            if self.is_complete and (self.shared_params_b is None or self.routed_expert_params_b is None):
                raise ValueError(
                    f"MoE workload '{self.key}' is marked complete but requires explicit "
                    "shared_params_b and routed_expert_params_b"
                )

            # Validate split consistency if both are given
            if self.shared_params_b is not None and self.routed_expert_params_b is not None:
                split_total = self.shared_params_b + self.routed_expert_params_b
                if abs(split_total - self.total_parameters_b) > ROUNDING_TOLERANCE_B:
                    raise ValueError(
                        f"Inconsistent parameter accounting for '{self.key}': "
                        f"shared_params_b ({self.shared_params_b}) + "
                        f"routed_expert_params_b ({self.routed_expert_params_b}) = {split_total:.4f} "
                        f"does not match total_parameters_b ({self.total_parameters_b}) "
                        f"within tolerance {ROUNDING_TOLERANCE_B}B."
                    )

                # Validate active parameter consistency against decomposition
                routed_per_token = self.experts_per_token * (
                    self.routed_expert_params_b / float(self.num_experts)
                )
                if self.shared_backbone_params_b is not None:
                    # In text inference, active compute parameters consist of text backbone,
                    # active routed experts, and output LM head (excluding sparse embedding lookup & multimodal vision)
                    expected_active = (
                        self.shared_backbone_params_b
                        + routed_per_token
                        + (self.output_head_params_b or 0.0)
                    )
                else:
                    expected_active = self.shared_params_b + routed_per_token

                if abs(self.active_parameters_b - expected_active) > ROUNDING_TOLERANCE_B:
                    raise ValueError(
                        f"Inconsistent active parameter count for '{self.key}': "
                        f"configured active_parameters_b ({self.active_parameters_b}) does not match "
                        f"reconciled active compute decomposition ({expected_active:.4f}) "
                        f"within tolerance {ROUNDING_TOLERANCE_B}B."
                    )


def _find_default_config_path(filename: str) -> str:
    import os

    cwd = os.getcwd()
    candidate = os.path.join(cwd, "configs", filename)
    if os.path.isfile(candidate):
        return candidate
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(base, "configs", filename)
    if os.path.isfile(candidate):
        return candidate
    return os.path.join("configs", filename)


def load_systems_config(config_source: str | dict[str, Any] | None = None) -> dict[str, SystemConfig]:
    """Loads and validates accelerator system configurations through SystemConfig."""
    import json
    import os

    if config_source is None:
        config_source = _find_default_config_path("systems.json")

    if isinstance(config_source, str):
        if not os.path.isfile(config_source):
            raise FileNotFoundError(f"Systems configuration file not found: {config_source}")
        with open(config_source, "r", encoding="utf-8") as f:
            raw_dict = json.load(f)
    elif isinstance(config_source, dict):
        raw_dict = config_source
    else:
        raise TypeError(f"config_source must be a file path or dict, got {type(config_source)}")

    systems: dict[str, SystemConfig] = {}
    for key, data in raw_dict.items():
        data_clean = dict(data)
        data_clean.pop("key", None)
        systems[key] = SystemConfig(key=key, **data_clean)
    return systems


def load_workloads_config(config_source: str | dict[str, Any] | None = None) -> dict[str, WorkloadConfig]:
    """Loads and validates workload configurations through WorkloadConfig."""
    import json
    import os

    if config_source is None:
        config_source = _find_default_config_path("workloads.json")

    if isinstance(config_source, str):
        if not os.path.isfile(config_source):
            raise FileNotFoundError(f"Workloads configuration file not found: {config_source}")
        with open(config_source, "r", encoding="utf-8") as f:
            raw_dict = json.load(f)
    elif isinstance(config_source, dict):
        raw_dict = config_source
    else:
        raise TypeError(f"config_source must be a file path or dict, got {type(config_source)}")

    workloads: dict[str, WorkloadConfig] = {}
    for key, data in raw_dict.items():
        data_clean = dict(data)
        data_clean.pop("key", None)
        workloads[key] = WorkloadConfig(key=key, **data_clean)
    return workloads

