"""Accelerator Simulation Framework for A48 and A216 Research."""

__version__ = "0.1.0"

from .models import SystemConfig, WorkloadConfig
from .capacity import (
    calculate_stored_weights_gb,
    calculate_usable_memory_gb,
    evaluate_capacity,
    CapacityEvaluation,
)
from .kv_cache import (
    KVCacheShape,
    KVCacheModel,
    calculate_kv_bytes_per_token,
    calculate_kv_cache_per_sequence_gb,
    calculate_kv_cache_total_gb,
)
from .prefill import (
    calculate_prefill_tflop,
    calculate_required_prefill_throughput,
    estimate_prefill,
    PrefillEstimate,
)
from .decode import (
    calculate_decode_bandwidth_ceiling,
    DecodeCeiling,
)
from .moe import (
    calculate_expected_expert_union,
    evaluate_moe_decode,
    MoEOffloadResult,
)

__all__ = [
    "SystemConfig",
    "WorkloadConfig",
    "calculate_stored_weights_gb",
    "calculate_usable_memory_gb",
    "evaluate_capacity",
    "CapacityEvaluation",
    "KVCacheShape",
    "KVCacheModel",
    "calculate_kv_bytes_per_token",
    "calculate_kv_cache_per_sequence_gb",
    "calculate_kv_cache_total_gb",
    "calculate_prefill_tflop",
    "calculate_required_prefill_throughput",
    "estimate_prefill",
    "PrefillEstimate",
    "calculate_decode_bandwidth_ceiling",
    "DecodeCeiling",
    "calculate_expected_expert_union",
    "evaluate_moe_decode",
    "MoEOffloadResult",
]
