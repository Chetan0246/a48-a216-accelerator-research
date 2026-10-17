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


def test_moe_host_offloading_and_limits():
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
    )
    assert res_no_host.expected_nonresident_union > 0.0
    assert res_no_host.host_weight_traffic_gb > 0.0
    assert res_no_host.combined_aggregate_tps == 0.0
    assert res_no_host.limiting_factor == "host offload unspecified"

    # When host link is slow (e.g. 0.032 TB/s = 32 GB/s PCIe), it limits throughput
    res_slow_host = evaluate_moe_decode(
        total_experts=64,
        resident_experts=32,
        top_k=2,
        shared_params_b=8.0,
        total_params_b=40.0,
        bits_per_param=4.5,
        batch_size=8,
        kv_cache_per_sequence_gb=1.610612736,
        card_sustained_bw_tb_s=1.4336,
        host_bw_tb_s=0.032,
        host_latency_ms=1.0,
    )
    assert res_slow_host.host_aggregate_tps < res_slow_host.card_aggregate_tps
    assert res_slow_host.combined_aggregate_tps == res_slow_host.host_aggregate_tps
    assert res_slow_host.limiting_factor == "host offload link"


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
