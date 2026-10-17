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
    if "hybrid" in att or att in ("linear_attention", "gated_deltanet"):
        raise UnsupportedAttentionTypeError(
            f"Attention type '{shape.attention_type}' (Hybrid Linear/Gated Attention) is not supported "
            "by the standard full K/V-cache formula. Qwen3.6-35B-A3B combines 30 Gated DeltaNet linear attention "
            "layers (fixed recurrent state) with 10 standard gated attention layers. Cache capacity and "
            "decode bandwidth projections are deferred until dual recurrent-state and KV-cache modeling is implemented."
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


@dataclass(frozen=True)
class HybridCacheConfig:
    """Configuration for hybrid architectures combining standard attention and linear attention.

    Published parameters (e.g. Qwen3.6-35B-A3B / Qwen3.5-35B-A3B):
      - 40 total layers (10 standard gated-attention, 30 Gated DeltaNet linear attention).
      - Gated Attention: 2 KV heads, head_dim 256.
      - Gated DeltaNet: 16 key heads (dim 128), 32 value heads (dim 128), conv kernel size 4.
      - Recurrent state precision: FP32 (32 bits, per mamba_ssm_dtype in published config.json).
      - Conv channels: 8192 (2 * key_dim + value_dim = 2 * 2048 + 4096).

    Simulator assumptions:
      - Conv state precision: FP16 (16 bits) baseline, configurable.
      - PagedAttention / allocation overhead: configurable percent (baseline 0.0%).
    """

    total_layers: int = 40
    gated_attention_layers: int = 10
    gated_deltanet_layers: int = 30
    attn_kv_heads: int = 2
    attn_head_dim: int = 256
    attn_precision_bits: float = 16.0  # FP16/BF16
    attn_overhead_percent: float = 0.0
    linear_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_heads: int = 32
    linear_value_head_dim: int = 128
    linear_state_precision_bits: float = 32.0  # FP32 recurrent state
    conv_kernel_size: int = 4
    conv_channels: int = 8192
    conv_precision_bits: float = 16.0

    def __post_init__(self) -> None:
        numeric_fields = {
            "total_layers": self.total_layers,
            "gated_attention_layers": self.gated_attention_layers,
            "gated_deltanet_layers": self.gated_deltanet_layers,
            "attn_kv_heads": self.attn_kv_heads,
            "attn_head_dim": self.attn_head_dim,
            "attn_precision_bits": self.attn_precision_bits,
            "attn_overhead_percent": self.attn_overhead_percent,
            "linear_key_heads": self.linear_key_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_heads": self.linear_value_heads,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_state_precision_bits": self.linear_state_precision_bits,
            "conv_kernel_size": self.conv_kernel_size,
            "conv_channels": self.conv_channels,
            "conv_precision_bits": self.conv_precision_bits,
        }
        for name, val in numeric_fields.items():
            if not math.isfinite(val):
                raise ValueError(f"{name} must be a finite number, got {val}")

        if (
            self.total_layers <= 0
            or self.gated_attention_layers <= 0
            or self.gated_deltanet_layers <= 0
        ):
            raise ValueError("Layer counts must be positive integers")

        if self.gated_attention_layers + self.gated_deltanet_layers != self.total_layers:
            raise ValueError(
                f"Layer split inconsistent: gated_attention_layers ({self.gated_attention_layers}) + "
                f"gated_deltanet_layers ({self.gated_deltanet_layers}) != total_layers ({self.total_layers})"
            )

        if self.attn_kv_heads <= 0 or self.attn_head_dim <= 0 or self.attn_precision_bits <= 0:
            raise ValueError("Attention parameters must be positive")

        if (
            self.linear_key_heads <= 0
            or self.linear_key_head_dim <= 0
            or self.linear_value_heads <= 0
            or self.linear_value_head_dim <= 0
            or self.linear_state_precision_bits <= 0
        ):
            raise ValueError("Linear attention parameters must be positive")

        if self.conv_kernel_size < 1 or self.conv_channels <= 0 or self.conv_precision_bits <= 0:
            raise ValueError("Convolution parameters must be positive")

        if self.attn_overhead_percent < 0:
            raise ValueError("attn_overhead_percent cannot be negative")


@dataclass(frozen=True)
class HybridCacheBreakdown:
    """Detailed memory and step traffic breakdown for hybrid attention models."""

    # Per-sequence memory in bytes
    attn_kv_bytes_per_seq: float
    recurrent_state_bytes_per_seq: float
    conv_state_bytes_per_seq: float
    total_state_bytes_per_seq: float

    # Per-sequence memory in decimal GB (1e9 bytes)
    attn_kv_gb_per_seq: float
    recurrent_state_gb_per_seq: float
    conv_state_gb_per_seq: float
    total_state_gb_per_seq: float

    # Batch memory in decimal GB (1e9 bytes)
    attn_kv_batch_gb: float
    recurrent_state_batch_gb: float
    conv_state_batch_gb: float
    total_state_batch_gb: float

    # Decode step traffic breakdown in decimal GB per step
    step_attn_kv_read_gb: float
    step_attn_kv_write_gb: float
    step_recurrent_state_read_gb: float
    step_recurrent_state_write_gb: float
    step_recurrent_state_traffic_gb: float
    step_conv_state_traffic_gb: float
    total_step_cache_traffic_gb: float

    # Metadata
    context_length: int
    batch_size: int
    config: HybridCacheConfig


def calculate_hybrid_cache_breakdown(
    config: HybridCacheConfig,
    context_length: int,
    batch_size: int = 1,
) -> HybridCacheBreakdown:
    """Calculates memory footprint and per-step decode memory traffic for hybrid attention.

    Formulas:
      1. Standard Gated Attention (O(S) context scaling):
         bytes_per_token = 2 * gated_attention_layers * attn_kv_heads * attn_head_dim * (attn_precision_bits / 8) * (1 + overhead)
         attn_kv_bytes_per_seq = context_length * bytes_per_token
         step_attn_kv_read_gb  = batch_size * (attn_kv_bytes_per_seq / 1e9)
         step_attn_kv_write_gb = batch_size * (bytes_per_token / 1e9)

      2. Gated DeltaNet Recurrent State (O(1) context-independent persistent state):
         State matrix per layer per value head: S_h in R^{d_k x d_v}
         elements_per_layer = linear_value_heads * linear_key_head_dim * linear_value_head_dim
         recurrent_state_bytes_per_seq = gated_deltanet_layers * elements_per_layer * (linear_state_precision_bits / 8)
         During decode, updating S_t = S_{t-1}(I - beta*k*k^T) + v*k^T requires reading S_{t-1} and writing S_t:
         step_recurrent_state_read_gb  = batch_size * (recurrent_state_bytes_per_seq / 1e9)
         step_recurrent_state_write_gb = batch_size * (recurrent_state_bytes_per_seq / 1e9)
         step_recurrent_state_traffic_gb = step_recurrent_state_read_gb + step_recurrent_state_write_gb

      3. Causal Convolution State (O(1) context-independent persistent state):
         Holds (conv_kernel_size - 1) history tokens across conv_channels:
         conv_state_bytes_per_seq = gated_deltanet_layers * (conv_kernel_size - 1) * conv_channels * (conv_precision_bits / 8)
         step_conv_state_traffic_gb = 2 * batch_size * (conv_state_bytes_per_seq / 1e9) (read buffer + write shifted buffer)

      4. Total Persistent Inference State:
         total_state_bytes_per_seq = attn_kv_bytes_per_seq + recurrent_state_bytes_per_seq + conv_state_bytes_per_seq
         total_state_batch_gb = batch_size * (total_state_bytes_per_seq / 1e9)
         total_step_cache_traffic_gb = (
             step_attn_kv_read_gb + step_attn_kv_write_gb + step_recurrent_state_traffic_gb + step_conv_state_traffic_gb
         )
    """
    for name, val in [("context_length", context_length), ("batch_size", batch_size)]:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val}")
    if context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    # 1. Gated Attention KV
    attn_bytes_per_elem = config.attn_precision_bits / 8.0
    attn_bytes_per_tok = (
        2.0
        * config.gated_attention_layers
        * config.attn_kv_heads
        * config.attn_head_dim
        * attn_bytes_per_elem
        * (1.0 + config.attn_overhead_percent / 100.0)
    )
    attn_kv_bytes_per_seq = float(context_length) * attn_bytes_per_tok
    attn_kv_gb_per_seq = attn_kv_bytes_per_seq / 1e9
    attn_kv_batch_gb = attn_kv_gb_per_seq * float(batch_size)

    # 2. Gated DeltaNet Recurrent State
    linear_bytes_per_elem = config.linear_state_precision_bits / 8.0
    elements_per_layer = (
        config.linear_value_heads * config.linear_key_head_dim * config.linear_value_head_dim
    )
    recurrent_state_bytes_per_seq = (
        float(config.gated_deltanet_layers) * float(elements_per_layer) * linear_bytes_per_elem
    )
    recurrent_state_gb_per_seq = recurrent_state_bytes_per_seq / 1e9
    recurrent_state_batch_gb = recurrent_state_gb_per_seq * float(batch_size)

    # 3. Causal Convolution State
    conv_bytes_per_elem = config.conv_precision_bits / 8.0
    conv_history_len = max(0, config.conv_kernel_size - 1)
    conv_state_bytes_per_seq = (
        float(config.gated_deltanet_layers)
        * float(conv_history_len)
        * float(config.conv_channels)
        * conv_bytes_per_elem
    )
    conv_state_gb_per_seq = conv_state_bytes_per_seq / 1e9
    conv_state_batch_gb = conv_state_gb_per_seq * float(batch_size)

    # 4. Total Persistent State
    total_state_bytes_per_seq = (
        attn_kv_bytes_per_seq + recurrent_state_bytes_per_seq + conv_state_bytes_per_seq
    )
    total_state_gb_per_seq = total_state_bytes_per_seq / 1e9
    total_state_batch_gb = total_state_gb_per_seq * float(batch_size)

    # 5. Decode Step Traffic Breakdown
    step_attn_kv_read_gb = attn_kv_batch_gb
    step_attn_kv_write_gb = (float(batch_size) * attn_bytes_per_tok) / 1e9
    step_recurrent_state_read_gb = recurrent_state_batch_gb
    step_recurrent_state_write_gb = recurrent_state_batch_gb
    step_recurrent_state_traffic_gb = step_recurrent_state_read_gb + step_recurrent_state_write_gb
    step_conv_state_traffic_gb = 2.0 * conv_state_batch_gb
    total_step_cache_traffic_gb = (
        step_attn_kv_read_gb
        + step_attn_kv_write_gb
        + step_recurrent_state_traffic_gb
        + step_conv_state_traffic_gb
    )

    return HybridCacheBreakdown(
        attn_kv_bytes_per_seq=attn_kv_bytes_per_seq,
        recurrent_state_bytes_per_seq=recurrent_state_bytes_per_seq,
        conv_state_bytes_per_seq=conv_state_bytes_per_seq,
        total_state_bytes_per_seq=total_state_bytes_per_seq,
        attn_kv_gb_per_seq=attn_kv_gb_per_seq,
        recurrent_state_gb_per_seq=recurrent_state_gb_per_seq,
        conv_state_gb_per_seq=conv_state_gb_per_seq,
        total_state_gb_per_seq=total_state_gb_per_seq,
        attn_kv_batch_gb=attn_kv_batch_gb,
        recurrent_state_batch_gb=recurrent_state_batch_gb,
        conv_state_batch_gb=conv_state_batch_gb,
        total_state_batch_gb=total_state_batch_gb,
        step_attn_kv_read_gb=step_attn_kv_read_gb,
        step_attn_kv_write_gb=step_attn_kv_write_gb,
        step_recurrent_state_read_gb=step_recurrent_state_read_gb,
        step_recurrent_state_write_gb=step_recurrent_state_write_gb,
        step_recurrent_state_traffic_gb=step_recurrent_state_traffic_gb,
        step_conv_state_traffic_gb=step_conv_state_traffic_gb,
        total_step_cache_traffic_gb=total_step_cache_traffic_gb,
        context_length=context_length,
        batch_size=batch_size,
        config=config,
    )


class HybridCacheModel:
    """Object-oriented wrapper around HybridCacheConfig calculations."""

    def __init__(self, config: HybridCacheConfig):
        self.config = config

    def evaluate(self, context_length: int, batch_size: int = 1) -> HybridCacheBreakdown:
        return calculate_hybrid_cache_breakdown(self.config, context_length, batch_size)

