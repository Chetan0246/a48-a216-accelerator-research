import pytest

from accelerator_sim.kv_cache import (
    KVCacheModel,
    KVCacheShape,
    calculate_kv_bytes_per_token,
    calculate_kv_cache_per_sequence_gb,
    calculate_kv_cache_total_gb,
)


def test_reference_standard_kv_cache():
    # Standard K/V cache: 48 layers, 8 KV heads, 128 dimensions per head, FP16
    shape = KVCacheShape(num_layers=48, num_kv_heads=8, head_dim=128, precision_bits=16.0)

    # Exact bytes per token = 2 * 48 * 8 * 128 * 2 = 196,608 bytes
    bytes_per_tok = calculate_kv_bytes_per_token(shape)
    assert bytes_per_tok == 196608.0

    # Decimal GB for 8K context (B=1) = 8192 * 196,608 / 1e9 = 1.610612736 GB
    kv_8k_seq = calculate_kv_cache_per_sequence_gb(shape, context_length=8192)
    assert pytest.approx(kv_8k_seq, rel=1e-5) == 1.610612736

    # Decimal GB for 8K context (B=8) = 8 * 1.610612736 = 12.884901888 GB
    kv_8k_b8 = calculate_kv_cache_total_gb(shape, context_length=8192, batch_size=8)
    assert pytest.approx(kv_8k_b8, rel=1e-5) == 12.884901888

    # Decimal GB for 4K context (B=1) = 4096 * 196,608 / 1e9 = 0.805306368 GB
    kv_4k_seq = calculate_kv_cache_per_sequence_gb(shape, context_length=4096)
    assert pytest.approx(kv_4k_seq, rel=1e-5) == 0.805306368


def test_kv_cache_model_wrapper_and_overhead():
    shape_with_overhead = KVCacheShape(
        num_layers=48,
        num_kv_heads=8,
        head_dim=128,
        precision_bits=16.0,
        overhead_percent=5.0,
    )
    model = KVCacheModel(shape_with_overhead)
    expected_bytes = 196608.0 * 1.05
    assert pytest.approx(model.bytes_per_token, rel=1e-5) == expected_bytes
    assert pytest.approx(model.per_sequence_gb(8192), rel=1e-5) == (8192 * expected_bytes) / 1e9
    assert pytest.approx(model.total_gb(8192, batch_size=4), rel=1e-5) == (4 * 8192 * expected_bytes) / 1e9


def test_kv_cache_invalid_inputs():
    with pytest.raises(ValueError):
        KVCacheShape(num_layers=0, num_kv_heads=8, head_dim=128)
    with pytest.raises(ValueError):
        KVCacheShape(num_layers=48, num_kv_heads=8, head_dim=128, overhead_percent=-1.0)
    with pytest.raises(ValueError, match="finite number"):
        KVCacheShape(num_layers=48, num_kv_heads=8, head_dim=128, precision_bits=float("nan"))
    with pytest.raises(ValueError, match="positive finite integer"):
        calculate_kv_cache_per_sequence_gb(KVCacheShape(num_layers=48, num_kv_heads=8, head_dim=128), context_length=-1)
    with pytest.raises(ValueError, match="positive finite integer"):
        calculate_kv_cache_per_sequence_gb(KVCacheShape(num_layers=48, num_kv_heads=8, head_dim=128), context_length=float("nan"))
    with pytest.raises(ValueError, match="positive finite integer"):
        calculate_kv_cache_total_gb(KVCacheShape(num_layers=48, num_kv_heads=8, head_dim=128), context_length=2048, batch_size=0)
    with pytest.raises(ValueError, match="positive finite integer"):
        calculate_kv_cache_total_gb(KVCacheShape(num_layers=48, num_kv_heads=8, head_dim=128), context_length=2048, batch_size=float("inf"))


def test_kv_cache_mla_and_unsupported_attention_rejection():
    from accelerator_sim.kv_cache import UnsupportedAttentionTypeError

    mla_shape = KVCacheShape(num_layers=28, num_kv_heads=16, head_dim=128, attention_type="mla")
    with pytest.raises(UnsupportedAttentionTypeError, match="Multi-Head Latent Attention"):
        calculate_kv_bytes_per_token(mla_shape)

    latent_shape = KVCacheShape(num_layers=28, num_kv_heads=16, head_dim=128, attention_type="latent")
    with pytest.raises(UnsupportedAttentionTypeError, match="Multi-Head Latent Attention"):
        calculate_kv_bytes_per_token(latent_shape)

    weird_shape = KVCacheShape(num_layers=28, num_kv_heads=16, head_dim=128, attention_type="unknown_attention")
    with pytest.raises(UnsupportedAttentionTypeError, match="Unsupported attention type"):
        calculate_kv_bytes_per_token(weird_shape)


def test_kv_cache_qwen_hybrid_attention_rejection():
    """Qwen3.6-35B-A3B hybrid linear/gated attention cache must be deferred."""
    from accelerator_sim.capacity import calculate_stored_weights_gb
    from accelerator_sim.kv_cache import UnsupportedAttentionTypeError

    # Qwen 3.6 35B A3B: 40 layers, 2 KV heads, 256 head dim
    hybrid_shape = KVCacheShape(
        num_layers=40,
        num_kv_heads=2,
        head_dim=256,
        attention_type="hybrid_gated_deltanet",
    )

    # 1. Prohibit standard KV cache calculation for hybrid attention
    with pytest.raises(UnsupportedAttentionTypeError, match="Hybrid Linear/Gated Attention"):
        calculate_kv_bytes_per_token(hybrid_shape)

    with pytest.raises(UnsupportedAttentionTypeError, match="Hybrid Linear/Gated Attention"):
        calculate_kv_cache_per_sequence_gb(hybrid_shape, context_length=8192)

    with pytest.raises(UnsupportedAttentionTypeError, match="Hybrid Linear/Gated Attention"):
        calculate_kv_cache_total_gb(hybrid_shape, context_length=8192, batch_size=4)

    # 2. Parameter storage calculations that do not depend on cache architecture remain fully supported
    qwen_weights_gb = calculate_stored_weights_gb(params_b=35.0, bits_per_param=4.5)
    assert pytest.approx(qwen_weights_gb, rel=1e-5) == 19.6875


