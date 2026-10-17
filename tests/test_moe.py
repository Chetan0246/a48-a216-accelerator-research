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

    # 1. Uniform explicit distribution matches analytical closed form
    uniform_probs = [1.0 / float(e)] * e
    u_analytical = calculate_expected_expert_union(e, k, batch_size=b)
    u_uniform_dist = calculate_expected_expert_union(e, k, batch_size=b, expert_probs=uniform_probs)
    assert pytest.approx(u_uniform_dist, rel=1e-5) == u_analytical

    # 2. Skewed distribution: tokens concentrate on top 4 experts (60% of selections)
    # Correct skew semantics: concentration reduces total unique experts read
    skewed_probs = [0.15] * 4 + [(1.0 - 0.60) / 60.0] * 60
    assert pytest.approx(sum(skewed_probs), rel=1e-6) == 1.0
    u_skewed = calculate_expected_expert_union(e, k, batch_size=b, expert_probs=skewed_probs)
    assert u_skewed < u_analytical

    # 3. Validation rejections for malformed distributions
    with pytest.raises(ValueError, match="length"):
        calculate_expected_expert_union(e, k, b, expert_probs=[0.5, 0.5])
    with pytest.raises(ValueError, match="negative"):
        calculate_expected_expert_union(e, k, b, expert_probs=[-0.1] + [1.1 / 63] * 63)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        calculate_expected_expert_union(e, k, b, expert_probs=[0.01] * 64)


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


