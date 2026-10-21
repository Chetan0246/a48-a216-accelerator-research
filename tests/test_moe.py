import pytest

from accelerator_sim.moe import (
    calculate_expected_expert_union,
    evaluate_moe_decode,
)


def test_reference_expert_union_expectations():
    # 64-expert model with top-2 routing:
    # Formula: 64 * [1 - (62/64)^B]
    e = 64
    k = 2

    # Batch 1: exactly 2.0000
    u1 = calculate_expected_expert_union(e, k, batch_size=1)
    assert pytest.approx(u1, rel=1e-4) == 2.0000

    # Batch 4: 7.632751... (approx 7.6328)
    u4 = calculate_expected_expert_union(e, k, batch_size=4)
    assert pytest.approx(u4, rel=1e-4) == 7.6328

    # Batch 8: 14.355143... (approx 14.3552)
    u8 = calculate_expected_expert_union(e, k, batch_size=8)
    assert pytest.approx(u8, rel=1e-4) == 14.3552

    # Batch 32: 40.828483... (approx 40.8285)
    u32 = calculate_expected_expert_union(e, k, batch_size=32)
    assert pytest.approx(u32, rel=1e-4) == 40.8285


def test_moe_all_resident_decode():
    # 40B total params, 8B shared, 32B routed across 64 experts -> 0.5B per expert
    # 64 resident experts (100% on card)
    res = evaluate_moe_decode(
        total_experts=64,
        resident_experts=64,
        top_k=2,
        shared_params_b=8.0,
        total_params_b=40.0,
        bits_per_param=4.5,
        batch_size=8,
        kv_cache_per_sequence_gb=1.610612736,
        card_sustained_bw_tb_s=1.4336,
    )
    assert res.nonresident_experts == 0
    assert res.expected_nonresident_union == 0.0
    assert res.host_weight_traffic_gb == 0.0
    assert res.card_aggregate_tps > 0.0
    assert res.combined_aggregate_tps == res.card_aggregate_tps
    assert res.limiting_factor == "on-card memory bandwidth"
    # Verification of analytical bound status
    assert res.model_type == "first_order_analytical_bound"
    assert res.is_validated_predictor is False


def test_moe_expert_distribution_skew_semantics():
    e = 64
    k = 2
    b = 8

    # 1. Uniform explicit marginal inclusion vector matches analytical closed form
    # Under uniform routing, each expert has marginal inclusion probability q_i = k / E
    uniform_inclusion = [float(k) / float(e)] * e
    assert pytest.approx(sum(uniform_inclusion), rel=1e-6) == float(k)
    u_analytical = calculate_expected_expert_union(e, k, batch_size=b)
    u_uniform_dist = calculate_expected_expert_union(e, k, batch_size=b, expert_inclusion_probs=uniform_inclusion)
    assert pytest.approx(u_uniform_dist, rel=1e-5) == u_analytical

    # 2. Skewed inclusion probabilities:
    # Top 4 experts have q_i = 0.25 (tokens frequently activate them; sum for top-4 = 1.0)
    # Remaining 60 experts share the remaining 1.0 inclusion probability (q_i = 1.0 / 60)
    # Total sum = 4 * 0.25 + 60 * (1/60) = 2.0 = top_k
    skewed_inclusion = [0.25] * 4 + [1.0 / 60.0] * 60
    assert pytest.approx(sum(skewed_inclusion), rel=1e-6) == float(k)
    u_skewed = calculate_expected_expert_union(e, k, batch_size=b, expert_inclusion_probs=skewed_inclusion)

    # Correct skew semantics: Token concentration on popular experts mathematically reduces
    # the expected number of distinct activated experts across the batch (Jensen's inequality on concave 1 - (1-q)^B).
    assert u_skewed < u_analytical

    # Also verify evaluate_moe_decode with expert_inclusion_probs
    res_uniform = evaluate_moe_decode(
        total_experts=e,
        resident_experts=e,
        top_k=k,
        shared_params_b=8.0,
        total_params_b=40.0,
        bits_per_param=4.5,
        batch_size=b,
        kv_cache_per_sequence_gb=1.6,
        card_sustained_bw_tb_s=1.4336,
        expert_inclusion_probs=uniform_inclusion,
    )
    res_skewed = evaluate_moe_decode(
        total_experts=e,
        resident_experts=e,
        top_k=k,
        shared_params_b=8.0,
        total_params_b=40.0,
        bits_per_param=4.5,
        batch_size=b,
        kv_cache_per_sequence_gb=1.6,
        card_sustained_bw_tb_s=1.4336,
        expert_inclusion_probs=skewed_inclusion,
    )
    assert pytest.approx(res_uniform.expected_total_union, rel=1e-5) == u_analytical
    assert res_skewed.expected_total_union < res_uniform.expected_total_union
    assert res_skewed.card_weight_traffic_gb < res_uniform.card_weight_traffic_gb

    # 3. Validation rejections for malformed inclusion vectors
    with pytest.raises(ValueError, match="length"):
        calculate_expected_expert_union(e, k, b, expert_inclusion_probs=[0.5, 0.5])
    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_expected_expert_union(e, k, b, expert_inclusion_probs=[-0.1] + [2.1 / 63] * 63)
    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_expected_expert_union(e, k, b, expert_inclusion_probs=[1.5] + [0.5 / 63] * 63)
    with pytest.raises(ValueError, match="must sum to top_k"):
        calculate_expected_expert_union(e, k, b, expert_inclusion_probs=[0.01] * 64)
    with pytest.raises(ValueError, match="finite number"):
        calculate_expected_expert_union(e, k, b, expert_inclusion_probs=[float("nan")] * 64)


def test_moe_host_offloading_serialized_and_overlapped_bounds():
    # 32 resident, 32 non-resident
    # When host BW is 0, combined TPS is 0 and status is unspecified
    res_no_host = evaluate_moe_decode(
        total_experts=64,
        resident_experts=32,
        top_k=2,
        shared_params_b=8.0,
        total_params_b=40.0,
        bits_per_param=4.5,
        batch_size=8,
        kv_cache_per_sequence_gb=1.610612736,
        card_sustained_bw_tb_s=1.4336,
        host_bw_tb_s=0.0,
        offload_mode="serialized",
    )
    assert res_no_host.expected_nonresident_union > 0.0
    assert res_no_host.host_weight_traffic_gb > 0.0
    assert res_no_host.combined_aggregate_tps == 0.0
    assert res_no_host.limiting_factor == "host offload unspecified"

    # With PCIe host offload link (0.032 TB/s = 32 GB/s, 1.0 ms dispatch latency)
    b = 8
    res_serialized = evaluate_moe_decode(
        total_experts=64,
        resident_experts=32,
        top_k=2,
        shared_params_b=8.0,
        total_params_b=40.0,
        bits_per_param=4.5,
        batch_size=b,
        kv_cache_per_sequence_gb=1.610612736,
        card_sustained_bw_tb_s=1.4336,
        host_bw_tb_s=0.032,
        host_latency_ms=1.0,
        offload_mode="serialized",
    )
    # Verify stage times are positive
    assert res_serialized.card_step_time_ms > 0.0
    assert res_serialized.host_step_time_ms > 0.0

    # Numerical verification of Serialized mode: B / (t_card + t_host)
    t_card_s = res_serialized.card_step_time_ms / 1000.0
    t_host_s = res_serialized.host_step_time_ms / 1000.0
    expected_serial_tps = b / (t_card_s + t_host_s)
    assert pytest.approx(res_serialized.serialized_aggregate_tps, rel=1e-5) == expected_serial_tps
    assert res_serialized.combined_aggregate_tps == res_serialized.serialized_aggregate_tps
    assert res_serialized.limiting_factor == "serialized on-card and host transfer"

    # Numerical verification of Overlapped mode: min(TPS_card, TPS_host)
    res_overlapped = evaluate_moe_decode(
        total_experts=64,
        resident_experts=32,
        top_k=2,
        shared_params_b=8.0,
        total_params_b=40.0,
        bits_per_param=4.5,
        batch_size=b,
        kv_cache_per_sequence_gb=1.610612736,
        card_sustained_bw_tb_s=1.4336,
        host_bw_tb_s=0.032,
        host_latency_ms=1.0,
        offload_mode="overlapped",
    )
    expected_overlap_tps = min(res_overlapped.card_aggregate_tps, res_overlapped.host_aggregate_tps)
    assert pytest.approx(res_overlapped.overlapped_aggregate_tps, rel=1e-5) == expected_overlap_tps
    assert res_overlapped.combined_aggregate_tps == res_overlapped.overlapped_aggregate_tps
    assert res_overlapped.limiting_factor == "host offload link"

    # Mathematical certainty: Serialized throughput is strictly less than overlapped throughput
    assert res_serialized.serialized_aggregate_tps < res_serialized.overlapped_aggregate_tps


def test_moe_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_expected_expert_union(expert_count=0, top_k=2, batch_size=1)
    with pytest.raises(ValueError):
        calculate_expected_expert_union(expert_count=64, top_k=100, batch_size=1)
    with pytest.raises(ValueError):
        calculate_expected_expert_union(expert_count=64, top_k=2, batch_size=0)
    with pytest.raises(ValueError):
        evaluate_moe_decode(
            total_experts=64,
            resident_experts=70,  # resident > total
            top_k=2,
            shared_params_b=8.0,
            total_params_b=40.0,
            bits_per_param=4.5,
            batch_size=8,
            kv_cache_per_sequence_gb=1.6,
            card_sustained_bw_tb_s=1.4336,
        )
    with pytest.raises(ValueError, match="offload_mode"):
        evaluate_moe_decode(
            total_experts=64,
            resident_experts=64,
            top_k=2,
            shared_params_b=8.0,
            total_params_b=40.0,
            bits_per_param=4.5,
            batch_size=8,
            kv_cache_per_sequence_gb=1.6,
            card_sustained_bw_tb_s=1.4336,
            offload_mode="invalid_mode",
        )
    with pytest.raises(ValueError, match="finite number"):
        calculate_expected_expert_union(expert_count=64, top_k=2, batch_size=float("nan"))
    with pytest.raises(ValueError, match="finite number"):
        evaluate_moe_decode(
            total_experts=64,
            resident_experts=64,
            top_k=2,
            shared_params_b=float("nan"),
            total_params_b=40.0,
            bits_per_param=4.5,
            batch_size=8,
            kv_cache_per_sequence_gb=1.6,
            card_sustained_bw_tb_s=1.4336,
        )


def test_evaluate_moe_decode_with_explicit_cache_traffic():
    """evaluate_moe_decode correctly uses step_cache_traffic_gb when provided."""
    res = evaluate_moe_decode(
        total_experts=64,
        resident_experts=64,
        top_k=2,
        shared_params_b=8.0,
        total_params_b=40.0,
        bits_per_param=4.5,
        batch_size=4,
        kv_cache_per_sequence_gb=0.0,
        card_sustained_bw_tb_s=1.0,
        step_cache_traffic_gb=1.5,
    )
    # card_cache_traffic_gb should be 1.5 GB
    assert pytest.approx(res.card_cache_traffic_gb, rel=1e-5) == 1.5
    # Total card step traffic = weight_traffic + 1.5
    assert pytest.approx(res.card_weight_traffic_gb + 1.5, rel=1e-5) == (res.card_weight_traffic_gb + res.card_cache_traffic_gb)


def test_moe_decode_weight_streaming_categories():
    """Separation of weight storage from decode weight streaming.

    Text-only decode streams shared text backbone + active routed experts + output LM head,
    strictly excludes vision tower weights, and avoids dense streaming of the embedding table.
    """
    # Qwen-like parameters:
    # 35.0B total, 32.2B routed (256 experts, top-8) -> 0.1258B/expert, 1.006B active/token
    # Non-routed shared pool: 2.8B = 1.43B backbone + 0.51B embedding + 0.51B head + 0.35B vision
    # Streamed shared parameters: 1.43B backbone + 0.51B head = 1.94B
    b = 4
    res_text = evaluate_moe_decode(
        total_experts=256,
        resident_experts=256,
        top_k=8,
        shared_params_b=2.8,
        total_params_b=35.0,
        bits_per_param=4.5,
        batch_size=b,
        kv_cache_per_sequence_gb=0.232,
        card_sustained_bw_tb_s=1.4336,
        streamed_shared_params_b=1.94,
        shared_backbone_params_b=1.43,
        embedding_params_b=0.51,
        output_head_params_b=0.51,
        vision_params_b=0.35,
        hidden_size=2048,
        mode="text_only",
    )

    # 1. Streamed shared traffic corresponds to 1.94B parameters: 1.94 * 4.5 / 8 = 1.09125 GB
    assert pytest.approx(res_text.streamed_shared_traffic_gb, rel=1e-4) == (1.94 * 4.5 / 8.0)
    assert res_text.streamed_shared_params_b == 1.94

    # 2. Output head traffic corresponds to 0.51B parameters: 0.51 * 4.5 / 8 = 0.286875 GB
    assert pytest.approx(res_text.output_head_traffic_gb, rel=1e-4) == (0.51 * 4.5 / 8.0)

    # 3. Vision tower weights are strictly 0 in text-only decode
    assert res_text.vision_weight_traffic_gb == 0.0

    # 4. Input embedding table is NOT streamed densely (saving 0.51 * 4.5 / 8 = 0.287 GB);
    # only sparse lookup traffic: 4 tokens * 2048 dim * 2 bytes = 16,384 bytes = 0.000016384 GB
    expected_lookup_gb = (b * 2048 * 2.0) / 1e9
    assert pytest.approx(res_text.embedding_lookup_traffic_gb, rel=1e-4) == expected_lookup_gb
    assert res_text.embedding_lookup_traffic_gb < 1e-4  # negligible compared to GB weight streams

    # 5. Routed weight traffic: E[U(4)] unique experts * expert_size * bits / 8
    expected_union = calculate_expected_expert_union(256, 8, batch_size=b)
    expert_size_b = 32.2 / 256.0
    expected_routed_gb = (expected_union * expert_size_b * 4.5) / 8.0
    assert pytest.approx(res_text.routed_weight_traffic_gb, rel=1e-4) == expected_routed_gb

    # 6. Disjoint additive columns sum without double counting
    expected_disjoint_sum = (
        res_text.backbone_weight_traffic_gb
        + res_text.output_head_traffic_gb
        + res_text.routed_weight_traffic_gb
        + res_text.embedding_lookup_traffic_gb
        + res_text.vision_weight_traffic_gb
    )
    assert pytest.approx(res_text.card_weight_traffic_gb, rel=1e-5) == expected_disjoint_sum
    assert pytest.approx(res_text.streamed_shared_traffic_gb, rel=1e-5) == (
        res_text.backbone_weight_traffic_gb + res_text.output_head_traffic_gb
    )

    # 7. In multimodal mode without explicit vision execution, vision tower weights are NOT charged on decode
    res_mm = evaluate_moe_decode(
        total_experts=256,
        resident_experts=256,
        top_k=8,
        shared_params_b=2.8,
        total_params_b=35.0,
        bits_per_param=4.5,
        batch_size=b,
        kv_cache_per_sequence_gb=0.232,
        card_sustained_bw_tb_s=1.4336,
        streamed_shared_params_b=1.94,
        shared_backbone_params_b=1.43,
        embedding_params_b=0.51,
        output_head_params_b=0.51,
        vision_params_b=0.35,
        hidden_size=2048,
        mode="multimodal",
        execute_vision_encoder=False,
    )
    # Regression test: switching to multimodal does not by itself stream vision weights during decode
    assert res_mm.vision_weight_traffic_gb == 0.0
    assert pytest.approx(res_mm.card_weight_traffic_gb, rel=1e-5) == res_text.card_weight_traffic_gb

    # 8. Only when explicitly instructed (execute_vision_encoder=True) does the step charge vision weights
    res_mm_exec = evaluate_moe_decode(
        total_experts=256,
        resident_experts=256,
        top_k=8,
        shared_params_b=2.8,
        total_params_b=35.0,
        bits_per_param=4.5,
        batch_size=b,
        kv_cache_per_sequence_gb=0.232,
        card_sustained_bw_tb_s=1.4336,
        streamed_shared_params_b=1.94,
        shared_backbone_params_b=1.43,
        embedding_params_b=0.51,
        output_head_params_b=0.51,
        vision_params_b=0.35,
        hidden_size=2048,
        mode="multimodal",
        execute_vision_encoder=True,
    )
    assert pytest.approx(res_mm_exec.vision_weight_traffic_gb, rel=1e-4) == (0.35 * 4.5 / 8.0)
    assert res_mm_exec.card_weight_traffic_gb > res_text.card_weight_traffic_gb


def test_storage_constancy_and_parameter_conservation():
    """Validates parameter conservation invariant and storage constancy."""
    from accelerator_sim.capacity import calculate_stored_weights_gb, evaluate_capacity
    from accelerator_sim.models import WorkloadConfig

    # 1. Total storage invariant: Total parameters = 35.0B regardless of inference mode
    stored_weights_gb = calculate_stored_weights_gb(params_b=35.0, bits_per_param=4.5)
    assert pytest.approx(stored_weights_gb, rel=1e-5) == 19.6875

    # Memory footprint in capacity evaluation remains identical
    cap_eval = evaluate_capacity(
        physical_memory_gb=48.0,
        params_b=35.0,
        kv_cache_gb=0.929,
        bits_per_param=4.5,
        runtime_reserve_gb=4.0,
    )
    assert pytest.approx(cap_eval.weights_gb, rel=1e-5) == 19.6875

    # 2. WorkloadConfig parameter conservation validation:
    # Valid decomposition
    valid_cfg = WorkloadConfig(
        key="test_qwen",
        name="Test Qwen",
        total_parameters_b=35.0,
        active_parameters_b=2.94,
        num_layers=40,
        hidden_size=2048,
        num_attention_heads=16,
        num_kv_heads=2,
        head_dim=256,
        is_moe=True,
        num_experts=256,
        experts_per_token=8,
        shared_params_b=2.8,
        routed_expert_params_b=32.2,
        shared_backbone_params_b=1.43,
        embedding_params_b=0.51,
        output_head_params_b=0.51,
        vision_params_b=0.35,
        streamed_shared_params_b=1.94,
    )
    assert valid_cfg.effective_streamed_shared_params_b == 1.94

    # 3. Invalid detailed split: detailed categories do not sum to shared_params_b
    with pytest.raises(ValueError, match="Inconsistent detailed parameter accounting"):
        WorkloadConfig(
            key="test_bad_split",
            name="Test Bad Split",
            total_parameters_b=35.0,
            active_parameters_b=2.94,
            num_layers=40,
            hidden_size=2048,
            num_attention_heads=16,
            num_kv_heads=2,
            head_dim=256,
            is_moe=True,
            num_experts=256,
            experts_per_token=8,
            shared_params_b=2.8,
            routed_expert_params_b=32.2,
            shared_backbone_params_b=1.43,
            embedding_params_b=0.51,
            output_head_params_b=0.51,
            vision_params_b=1.50,  # deliberately incorrect (sum = 3.95 != 2.8)
        )

    # 4. Inconsistent active parameter count: active_parameters_b does not match decomposition
    with pytest.raises(ValueError, match="Inconsistent active parameter count"):
        WorkloadConfig(
            key="test_bad_active",
            name="Test Bad Active",
            total_parameters_b=35.0,
            active_parameters_b=5.0,  # deliberately wrong active count
            num_layers=40,
            hidden_size=2048,
            num_attention_heads=16,
            num_kv_heads=2,
            head_dim=256,
            is_moe=True,
            num_experts=256,
            experts_per_token=8,
            shared_params_b=2.8,
            routed_expert_params_b=32.2,
            shared_backbone_params_b=1.43,
            embedding_params_b=0.51,
            output_head_params_b=0.51,
            vision_params_b=0.35,
        )


def test_moe_embedding_precision_and_disjoint_traffic():
    """Validates configurable embedding precision and exact additive disjoint traffic."""
    b = 8
    hidden_dim = 4096

    # Test with 32-bit embedding precision
    res_fp32 = evaluate_moe_decode(
        total_experts=64,
        resident_experts=64,
        top_k=4,
        shared_params_b=4.0,
        total_params_b=32.0,
        bits_per_param=4.0,
        batch_size=b,
        kv_cache_per_sequence_gb=0.5,
        card_sustained_bw_tb_s=1.0,
        shared_backbone_params_b=2.0,
        embedding_params_b=1.0,
        output_head_params_b=1.0,
        vision_params_b=0.0,
        hidden_size=hidden_dim,
        embedding_precision_bits=32.0,
    )
    expected_fp32_lookup_gb = (b * hidden_dim * (32.0 / 8.0)) / 1e9
    assert pytest.approx(res_fp32.embedding_lookup_traffic_gb, rel=1e-5) == expected_fp32_lookup_gb
    assert res_fp32.embedding_precision_bits == 32.0

    # Test with 8-bit embedding precision
    res_int8 = evaluate_moe_decode(
        total_experts=64,
        resident_experts=64,
        top_k=4,
        shared_params_b=4.0,
        total_params_b=32.0,
        bits_per_param=4.0,
        batch_size=b,
        kv_cache_per_sequence_gb=0.5,
        card_sustained_bw_tb_s=1.0,
        shared_backbone_params_b=2.0,
        embedding_params_b=1.0,
        output_head_params_b=1.0,
        vision_params_b=0.0,
        hidden_size=hidden_dim,
        embedding_precision_bits=8.0,
    )
    expected_int8_lookup_gb = (b * hidden_dim * (8.0 / 8.0)) / 1e9
    assert pytest.approx(res_int8.embedding_lookup_traffic_gb, rel=1e-5) == expected_int8_lookup_gb
    assert res_int8.embedding_lookup_traffic_gb == expected_fp32_lookup_gb / 4.0

    # Additive reconciliation
    additive_sum = (
        res_int8.backbone_weight_traffic_gb
        + res_int8.output_head_traffic_gb
        + res_int8.routed_weight_traffic_gb
        + res_int8.embedding_lookup_traffic_gb
        + res_int8.vision_weight_traffic_gb
    )
    assert pytest.approx(res_int8.card_weight_traffic_gb, rel=1e-6) == additive_sum

    # Invalid embedding precision
    with pytest.raises(ValueError, match="embedding_precision_bits"):
        evaluate_moe_decode(
            total_experts=64,
            resident_experts=64,
            top_k=4,
            shared_params_b=4.0,
            total_params_b=32.0,
            bits_per_param=4.0,
            batch_size=b,
            kv_cache_per_sequence_gb=0.5,
            card_sustained_bw_tb_s=1.0,
            embedding_precision_bits=-16.0,
        )




