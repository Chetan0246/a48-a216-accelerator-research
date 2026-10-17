"""Memory capacity modeling for model weights, KV cache, and usable headroom."""

from dataclasses import dataclass


def calculate_stored_weights_gb(params_b: float, bits_per_param: float = 4.5) -> float:
    """Calculates model weights footprint in decimal GB.

    Args:
        params_b: Total parameter count in billions.
        bits_per_param: Effective precision in bits per parameter, including
            scales and quantization metadata.

    Returns:
        Stored weight memory in decimal GB (1e9 bytes).
    """
    if params_b <= 0 or bits_per_param <= 0:
        raise ValueError("params_b and bits_per_param must be positive")
    return (params_b * bits_per_param) / 8.0


def calculate_usable_memory_gb(physical_memory_gb: float, runtime_reserve_gb: float = 4.0) -> float:
    """Calculates usable memory capacity after subtracting runtime reserve.

    Args:
        physical_memory_gb: Total physical device memory in decimal GB.
        runtime_reserve_gb: Configurable reserve for driver, OS buffers, and runtime context.

    Returns:
        Usable memory in decimal GB.
    """
    if physical_memory_gb <= 0:
        raise ValueError("physical_memory_gb must be positive")
    if runtime_reserve_gb < 0:
        raise ValueError("runtime_reserve_gb cannot be negative")
    return physical_memory_gb - runtime_reserve_gb


@dataclass(frozen=True)
class CapacityEvaluation:
    weights_gb: float
    kv_cache_gb: float
    model_footprint_gb: float
    runtime_reserve_gb: float
    physical_memory_gb: float
    usable_memory_gb: float
    headroom_gb: float
    fits_in_memory: bool


def evaluate_capacity(
    physical_memory_gb: float,
    params_b: float,
    kv_cache_gb: float,
    bits_per_param: float = 4.5,
    runtime_reserve_gb: float = 4.0,
) -> CapacityEvaluation:
    """Evaluates whether model weights and KV cache fit within usable device memory.

    Accounting rules (no double-counting):
        model_footprint_gb = weights_gb + kv_cache_gb
        usable_memory_gb   = physical_memory_gb - runtime_reserve_gb
        headroom_gb        = usable_memory_gb - model_footprint_gb
        fits_in_memory     = headroom_gb >= 0.0

    Args:
        physical_memory_gb: Physical memory capacity in decimal GB.
        params_b: Total parameter count in billions.
        kv_cache_gb: Precomputed total KV cache memory for the batch in decimal GB.
        bits_per_param: Effective weight bits per parameter.
        runtime_reserve_gb: Reserved memory for runtime/driver overhead.

    Returns:
        CapacityEvaluation breakdown.
    """
    if kv_cache_gb < 0:
        raise ValueError("kv_cache_gb cannot be negative")

    weights_gb = calculate_stored_weights_gb(params_b, bits_per_param)
    usable_gb = calculate_usable_memory_gb(physical_memory_gb, runtime_reserve_gb)
    footprint_gb = weights_gb + kv_cache_gb
    headroom_gb = usable_gb - footprint_gb
    fits = headroom_gb >= 0.0

    return CapacityEvaluation(
        weights_gb=weights_gb,
        kv_cache_gb=kv_cache_gb,
        model_footprint_gb=footprint_gb,
        runtime_reserve_gb=runtime_reserve_gb,
        physical_memory_gb=physical_memory_gb,
        usable_memory_gb=usable_gb,
        headroom_gb=headroom_gb,
        fits_in_memory=fits,
    )
