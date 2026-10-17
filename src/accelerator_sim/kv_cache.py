"""KV Cache memory modeling for Transformer architectures."""

from __future__ import annotations

import math
from dataclasses import dataclass


class UnsupportedAttentionTypeError(NotImplementedError):
    """Raised when an attention/cache mechanism is not supported by standard K/V formulas."""


@dataclass(frozen=True)
class KVCacheShape:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    precision_bits: float = 16.0  # FP16/BF16 = 16.0, FP8/INT8 = 8.0
    overhead_percent: float = 0.0  # Configurable allocation/paging overhead (baseline: 0.0%)
    attention_type: str = "gqa"  # "mha", "gqa", "mqa", or "mla"

    def __post_init__(self) -> None:
        numeric_fields = {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "precision_bits": self.precision_bits,
            "overhead_percent": self.overhead_percent,
        }
        for name, val in numeric_fields.items():
            if not math.isfinite(val):
                raise ValueError(f"{name} must be a finite number, got {val}")

        if self.num_layers <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0 or self.precision_bits <= 0:
            raise ValueError("num_layers, num_kv_heads, head_dim, and precision_bits must be positive")
        if self.overhead_percent < 0:
            raise ValueError("overhead_percent cannot be negative")


def calculate_kv_bytes_per_token(shape: KVCacheShape) -> float:
    """Calculates bytes required for 1 token across all layers for a single sequence.

    Formula:
        2 * num_layers * num_kv_heads * head_dim * (precision_bits / 8) * (1 + overhead_percent / 100)

    Raises:
        UnsupportedAttentionTypeError: If shape.attention_type is 'mla' or any unsupported format.
        ValueError: If shape parameters are invalid.
    """
    att = shape.attention_type.strip().lower()
    if att in ("mla", "latent"):
        raise UnsupportedAttentionTypeError(
            f"Attention type '{shape.attention_type}' (Multi-Head Latent Attention) is not supported "
            "by the standard full K/V-cache formula. MLA uses compressed latent KV vectors and decoupled "
            "RoPE keys. Evaluation is deferred to avoid misleading capacity and bandwidth estimates."
        )
    if att not in ("mha", "gqa", "mqa"):
        raise UnsupportedAttentionTypeError(
            f"Unsupported attention type '{shape.attention_type}'. "
            "Supported standard cache representations are 'mha', 'gqa', 'mqa'."
        )

    bytes_per_elem = shape.precision_bits / 8.0
    base_bytes = 2.0 * shape.num_layers * shape.num_kv_heads * shape.head_dim * bytes_per_elem
    return base_bytes * (1.0 + shape.overhead_percent / 100.0)


def calculate_kv_cache_per_sequence_gb(shape: KVCacheShape, context_length: int) -> float:
    """Calculates K+V cache memory for a single sequence in decimal GB (1e9 bytes)."""
    if not math.isfinite(context_length) or context_length <= 0:
        raise ValueError("context_length must be a positive finite integer")
    total_bytes = context_length * calculate_kv_bytes_per_token(shape)
    return total_bytes / 1e9


def calculate_kv_cache_total_gb(
    shape: KVCacheShape,
    context_length: int,
    batch_size: int = 1,
) -> float:
    """Calculates total K+V cache memory for a batch in decimal GB (1e9 bytes)."""
    if not math.isfinite(batch_size) or batch_size <= 0:
        raise ValueError("batch_size must be a positive finite integer")
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
