"""KV Cache memory modeling for Transformer architectures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheShape:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    precision_bits: float = 16.0  # FP16/BF16 = 16.0, FP8/INT8 = 8.0
    overhead_percent: float = 0.0  # PagedAttention or block allocation overhead


def calculate_kv_bytes_per_token(shape: KVCacheShape) -> float:
    """Calculates bytes required for 1 token across all layers for a single sequence.

    Formula:
        2 * num_layers * num_kv_heads * head_dim * (precision_bits / 8) * (1 + overhead_percent / 100)
    """
    if shape.num_layers <= 0 or shape.num_kv_heads <= 0 or shape.head_dim <= 0 or shape.precision_bits <= 0:
        raise ValueError("num_layers, num_kv_heads, head_dim, and precision_bits must be positive")
    if shape.overhead_percent < 0:
        raise ValueError("overhead_percent cannot be negative")

    bytes_per_elem = shape.precision_bits / 8.0
    base_bytes = 2.0 * shape.num_layers * shape.num_kv_heads * shape.head_dim * bytes_per_elem
    return base_bytes * (1.0 + shape.overhead_percent / 100.0)


def calculate_kv_cache_per_sequence_gb(shape: KVCacheShape, context_length: int) -> float:
    """Calculates K+V cache memory for a single sequence in decimal GB (1e9 bytes)."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    total_bytes = context_length * calculate_kv_bytes_per_token(shape)
    return total_bytes / 1e9


def calculate_kv_cache_total_gb(
    shape: KVCacheShape,
    context_length: int,
    batch_size: int = 1,
) -> float:
    """Calculates total K+V cache memory for a batch in decimal GB (1e9 bytes)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return calculate_kv_cache_per_sequence_gb(shape, context_length) * batch_size


class KVCacheModel:
    """Object-oriented wrapper around KVCacheShape calculations."""

    def __init__(self, shape: KVCacheShape):
        self.shape = shape

    @property
    def bytes_per_token(self) -> float:
        return calculate_kv_bytes_per_token(self.shape)

    def per_sequence_gb(self, context_length: int) -> float:
        return calculate_kv_cache_per_sequence_gb(self.shape, context_length)

    def total_gb(self, context_length: int, batch_size: int = 1) -> float:
        return calculate_kv_cache_total_gb(self.shape, context_length, batch_size)
