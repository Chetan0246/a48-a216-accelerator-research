import pytest

from accelerator_sim.kv_cache import (
    HybridCacheBreakdown,
    HybridCacheConfig,
    HybridCacheModel,
    KVCacheModel,
    KVCacheShape,
    calculate_hybrid_cache_breakdown,
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


def test_reference_hybrid_cache_breakdown():
    """Validates exact reference calculations for Qwen3.6-35B-A3B hybrid architecture."""
    cfg = HybridCacheConfig()

    # 1. Evaluate at 8,192 tokens context, batch 1
    res = calculate_hybrid_cache_breakdown(cfg, context_length=8192, batch_size=1)

    # Gated-attention KV: 10 layers, 2 KV heads, 256 head dim, FP16
    # Exact bytes per token = 2 * 10 * 2 * 256 * (16/8) = 20,480 bytes
    # Exact bytes for 8192 context = 8192 * 20,480 = 167,772,160 bytes
    assert res.attn_kv_bytes_per_seq == 167772160.0
    # In decimal GB: 167,772,160 / 1e9 = 0.16777216 GB (~0.168 decimal GB)
    assert pytest.approx(res.attn_kv_gb_per_seq, rel=1e-5) == 0.16777216
    assert pytest.approx(res.attn_kv_gb_per_seq, abs=0.001) == 0.168

    # Gated DeltaNet recurrent state: 30 layers, 32 value heads, 128 dk, 128 dv, FP32
    # Elements per layer = 32 * 128 * 128 = 524,288
    # Total elements across 30 layers = 30 * 524,288 = 15,728,640
    # Exact bytes in FP32 (4 bytes/elem) = 15,728,640 * 4 = 62,914,560 bytes
    assert res.recurrent_state_bytes_per_seq == 62914560.0
    # In decimal GB: 62,914,560 / 1e9 = 0.06291456 GB (~0.063 decimal GB)
    assert pytest.approx(res.recurrent_state_gb_per_seq, rel=1e-5) == 0.06291456

    # Causal Conv1D state: 30 layers, kernel size 4 (3 history steps), 8192 channels, FP16
    # Total elements = 30 * 3 * 8192 = 737,280
    # Exact bytes in FP16 (2 bytes/elem) = 737,280 * 2 = 1,474,560 bytes
    assert res.conv_state_bytes_per_seq == 1474560.0
    # In decimal GB: 1,474,560 / 1e9 = 0.00147456 GB (~0.0015 decimal GB)
    assert pytest.approx(res.conv_state_gb_per_seq, rel=1e-5) == 0.00147456

    # Total persistent inference state per sequence:
    # 167,772,160 + 62,914,560 + 1,474,560 = 232,161,280 bytes
    assert res.total_state_bytes_per_seq == 232161280.0
    assert pytest.approx(res.total_state_gb_per_seq, rel=1e-5) == 0.23216128
    assert pytest.approx(res.total_state_batch_gb, rel=1e-5) == 0.23216128


def test_hybrid_context_scaling_properties():
    """Full-attention KV cache must scale linearly O(S); recurrent state must be constant O(1) w.r.t context."""
    cfg = HybridCacheConfig()

    res_2k = calculate_hybrid_cache_breakdown(cfg, context_length=2048, batch_size=1)
    res_8k = calculate_hybrid_cache_breakdown(cfg, context_length=8192, batch_size=1)
    res_32k = calculate_hybrid_cache_breakdown(cfg, context_length=32768, batch_size=1)

    # Context length increased by 4x (2048 -> 8192) and 16x (2048 -> 32768)
    assert res_8k.attn_kv_bytes_per_seq == pytest.approx(res_2k.attn_kv_bytes_per_seq * 4.0, rel=1e-6)
    assert res_32k.attn_kv_bytes_per_seq == pytest.approx(res_2k.attn_kv_bytes_per_seq * 16.0, rel=1e-6)

    # Recurrent state MUST remain strictly constant regardless of context length S
    assert res_2k.recurrent_state_bytes_per_seq == res_8k.recurrent_state_bytes_per_seq
    assert res_2k.recurrent_state_bytes_per_seq == res_32k.recurrent_state_bytes_per_seq
    assert res_8k.recurrent_state_bytes_per_seq == 62914560.0

    # Causal conv state MUST remain strictly constant regardless of context length S
    assert res_2k.conv_state_bytes_per_seq == res_8k.conv_state_bytes_per_seq
    assert res_2k.conv_state_bytes_per_seq == res_32k.conv_state_bytes_per_seq
    assert res_8k.conv_state_bytes_per_seq == 1474560.0


def test_hybrid_batch_scaling_properties():
    """All persistent state components must scale linearly with batch size B."""
    cfg = HybridCacheConfig()

    res_b1 = calculate_hybrid_cache_breakdown(cfg, context_length=8192, batch_size=1)
    res_b4 = calculate_hybrid_cache_breakdown(cfg, context_length=8192, batch_size=4)
    res_b8 = calculate_hybrid_cache_breakdown(cfg, context_length=8192, batch_size=8)

    assert pytest.approx(res_b4.attn_kv_batch_gb, rel=1e-6) == res_b1.attn_kv_batch_gb * 4.0
    assert pytest.approx(res_b8.attn_kv_batch_gb, rel=1e-6) == res_b1.attn_kv_batch_gb * 8.0

    assert pytest.approx(res_b4.recurrent_state_batch_gb, rel=1e-6) == res_b1.recurrent_state_batch_gb * 4.0
    assert pytest.approx(res_b8.recurrent_state_batch_gb, rel=1e-6) == res_b1.recurrent_state_batch_gb * 8.0

    assert pytest.approx(res_b4.conv_state_batch_gb, rel=1e-6) == res_b1.conv_state_batch_gb * 4.0
    assert pytest.approx(res_b8.conv_state_batch_gb, rel=1e-6) == res_b1.conv_state_batch_gb * 8.0

    assert pytest.approx(res_b4.total_state_batch_gb, rel=1e-6) == res_b1.total_state_batch_gb * 4.0
    assert pytest.approx(res_b8.total_state_batch_gb, rel=1e-6) == res_b1.total_state_batch_gb * 8.0


def test_hybrid_precision_scaling():
    """Precision changes must proportionally scale byte counts."""
    # FP32 vs FP16 recurrent state
    cfg_fp32 = HybridCacheConfig(linear_state_precision_bits=32.0)
    cfg_fp16 = HybridCacheConfig(linear_state_precision_bits=16.0)

    res_fp32 = calculate_hybrid_cache_breakdown(cfg_fp32, context_length=4096, batch_size=1)
    res_fp16 = calculate_hybrid_cache_breakdown(cfg_fp16, context_length=4096, batch_size=1)

    assert res_fp32.recurrent_state_bytes_per_seq == 62914560.0
    assert res_fp16.recurrent_state_bytes_per_seq == 31457280.0
    assert res_fp16.recurrent_state_bytes_per_seq == res_fp32.recurrent_state_bytes_per_seq / 2.0

    # FP8 attention KV cache
    cfg_fp8_attn = HybridCacheConfig(attn_precision_bits=8.0)
    res_fp8_attn = calculate_hybrid_cache_breakdown(cfg_fp8_attn, context_length=4096, batch_size=1)
    res_fp16_attn = calculate_hybrid_cache_breakdown(cfg_fp32, context_length=4096, batch_size=1)
    assert res_fp8_attn.attn_kv_bytes_per_seq == res_fp16_attn.attn_kv_bytes_per_seq / 2.0


def test_hybrid_decode_step_traffic_separation():
    """Decode traffic must strictly separate weight reads, KV reads, state updates, and conv shifts."""
    cfg = HybridCacheConfig()
    res = calculate_hybrid_cache_breakdown(cfg, context_length=8192, batch_size=4)

    # 1. KV reads scale with context length S: 4 * 0.16777216 = 0.67108864 GB
    assert pytest.approx(res.step_attn_kv_read_gb, rel=1e-5) == 0.67108864

    # 2. KV write is only for the newly generated token: 4 * 20,480 / 1e9 = 0.00008192 GB
    assert pytest.approx(res.step_attn_kv_write_gb, rel=1e-5) == 0.00008192

    # 3. Recurrent state traffic = read S_{t-1} + write S_t: 2 * (4 * 62,914,560 / 1e9) = 0.50331648 GB
    assert pytest.approx(res.step_recurrent_state_read_gb, rel=1e-5) == 0.25165824
    assert pytest.approx(res.step_recurrent_state_write_gb, rel=1e-5) == 0.25165824
    assert pytest.approx(res.step_recurrent_state_traffic_gb, rel=1e-5) == 0.50331648

    # 4. Conv state traffic = read buffer + write shifted: 2 * (4 * 1,474,560 / 1e9) = 0.01179648 GB
    assert pytest.approx(res.step_conv_state_traffic_gb, rel=1e-5) == 0.01179648

    # 5. Total step cache traffic must match exact sum of distinct components
    expected_sum = (
        res.step_attn_kv_read_gb
        + res.step_attn_kv_write_gb
        + res.step_recurrent_state_traffic_gb
        + res.step_conv_state_traffic_gb
    )
    assert pytest.approx(res.total_step_cache_traffic_gb, rel=1e-6) == expected_sum


def test_hybrid_cache_invalid_inputs():
    """Rejects malformed, non-finite, and inconsistent hybrid configurations."""
    # Mismatched layer split (10 + 29 != 40)
    with pytest.raises(ValueError, match="Layer split inconsistent"):
        HybridCacheConfig(total_layers=40, gated_attention_layers=10, gated_deltanet_layers=29)

    # Zero / negative layer counts
    with pytest.raises(ValueError, match="Layer counts must be positive"):
        HybridCacheConfig(total_layers=40, gated_attention_layers=0, gated_deltanet_layers=40)

    # Non-finite values
    with pytest.raises(ValueError, match="finite number"):
        HybridCacheConfig(linear_state_precision_bits=float("nan"))

    # Invalid context length or batch size in calculate_hybrid_cache_breakdown
    cfg = HybridCacheConfig()
    with pytest.raises(ValueError, match="context_length must be a positive integer"):
        calculate_hybrid_cache_breakdown(cfg, context_length=-1)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        calculate_hybrid_cache_breakdown(cfg, context_length=2048, batch_size=0)
    with pytest.raises(ValueError, match="finite number"):
        calculate_hybrid_cache_breakdown(cfg, context_length=2048, batch_size=float("inf"))


def test_hybrid_cache_model_wrapper():
    """HybridCacheModel wrapper correctly forwards evaluate call."""
    cfg = HybridCacheConfig()
    model = HybridCacheModel(cfg)
    breakdown = model.evaluate(context_length=8192, batch_size=2)
    assert isinstance(breakdown, HybridCacheBreakdown)
    assert pytest.approx(breakdown.attn_kv_batch_gb, rel=1e-5) == 0.16777216 * 2.0



