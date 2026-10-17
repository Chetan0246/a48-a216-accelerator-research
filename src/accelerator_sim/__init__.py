"""Accelerator Simulation Framework for A48 and A216 Research."""

__version__ = "0.1.0"

from .capacity import (
    CapacityEvaluation,
    calculate_stored_weights_gb,
    calculate_usable_memory_gb,
    evaluate_capacity,
)
from .decode import (
    DecodeCeiling,
    calculate_decode_bandwidth_ceiling,
)
from .kv_cache import (
    HybridCacheBreakdown,
    HybridCacheConfig,
    HybridCacheModel,
    KVCacheModel,
    KVCacheShape,
    UnsupportedAttentionTypeError,
    calculate_hybrid_cache_breakdown,
    calculate_kv_bytes_per_token,
    calculate_kv_cache_per_sequence_gb,
    calculate_kv_cache_total_gb,
)
from .models import (
    ROUNDING_TOLERANCE_B,
    SystemConfig,
    WorkloadConfig,
    load_systems_config,
    load_workloads_config,
)
from .moe import (
    MoEOffloadResult,
    calculate_expected_expert_union,
    evaluate_moe_decode,
)
from .prefill import (
    PrefillEstimate,
    calculate_prefill_tflop,
    calculate_required_prefill_throughput,
    estimate_prefill,
)

__all__ = [
    "ROUNDING_TOLERANCE_B",
    "CapacityEvaluation",
    "DecodeCeiling",
    "HybridCacheBreakdown",
    "HybridCacheConfig",
    "HybridCacheModel",
    "KVCacheModel",
    "KVCacheShape",
    "MoEOffloadResult",
    "PrefillEstimate",
    "SystemConfig",
    "UnsupportedAttentionTypeError",
    "WorkloadConfig",
    "calculate_decode_bandwidth_ceiling",
    "calculate_expected_expert_union",
    "calculate_hybrid_cache_breakdown",
    "calculate_kv_bytes_per_token",
    "calculate_kv_cache_per_sequence_gb",
    "calculate_kv_cache_total_gb",
    "calculate_prefill_tflop",
    "calculate_required_prefill_throughput",
    "calculate_stored_weights_gb",
    "calculate_usable_memory_gb",
    "estimate_prefill",
    "evaluate_capacity",
    "evaluate_moe_decode",
    "load_systems_config",
    "load_workloads_config",
]
