import pytest
from accelerator_sim.capacity import (
    calculate_stored_weights_gb,
    calculate_usable_memory_gb,
    evaluate_capacity,
)
from accelerator_sim.models import SystemConfig, WorkloadConfig


def test_reference_stored_weights():
    # 32B dense model at 4.5 bits per parameter = 18.00 GB
    weights_32b = calculate_stored_weights_gb(params_b=32.0, bits_per_param=4.5)
    assert pytest.approx(weights_32b, rel=1e-4) == 18.00

    # 70B model at 4.5 bits per parameter = 39.375 GB
    weights_70b = calculate_stored_weights_gb(params_b=70.0, bits_per_param=4.5)
    assert pytest.approx(weights_70b, rel=1e-4) == 39.375


def test_reference_usable_memory_and_reserves():
    # A48 baseline: 48 GB physical, 4 GB reserve -> 44 GB usable
    assert calculate_usable_memory_gb(physical_memory_gb=48.0, runtime_reserve_gb=4.0) == 44.0

    # Sensitivity scenarios: 2 GB reserve -> 46 GB usable; 8 GB reserve -> 40 GB usable
    assert calculate_usable_memory_gb(physical_memory_gb=48.0, runtime_reserve_gb=2.0) == 46.0
    assert calculate_usable_memory_gb(physical_memory_gb=48.0, runtime_reserve_gb=8.0) == 40.0

    # A216 baseline: 216 GB physical, 4 GB reserve -> 212 GB usable
    assert calculate_usable_memory_gb(physical_memory_gb=216.0, runtime_reserve_gb=4.0) == 212.0


def test_evaluate_capacity_headroom_and_fit():
    # 32B model (18.0 GB) + 1.610612736 GB KV cache on A48 (48 GB physical, 4 GB reserve)
    cap = evaluate_capacity(
        physical_memory_gb=48.0,
        params_b=32.0,
        kv_cache_gb=1.610612736,
        bits_per_param=4.5,
        runtime_reserve_gb=4.0,
    )
    assert cap.weights_gb == 18.0
    assert cap.usable_memory_gb == 44.0
    expected_footprint = 18.0 + 1.610612736
    assert pytest.approx(cap.model_footprint_gb, rel=1e-5) == expected_footprint
    expected_headroom = 44.0 - expected_footprint
    assert pytest.approx(cap.headroom_gb, rel=1e-5) == expected_headroom
    assert cap.fits_in_memory is True


def test_capacity_oom_behavior():
    # 100B model @ 4.5 bits = 56.25 GB weights -> exceeds A48 usable memory (44 GB)
    cap_a48 = evaluate_capacity(
        physical_memory_gb=48.0,
        params_b=100.0,
        kv_cache_gb=0.0,
        bits_per_param=4.5,
        runtime_reserve_gb=4.0,
    )
    assert cap_a48.fits_in_memory is False
    assert cap_a48.headroom_gb < 0.0

    # Fits on A216 (usable 212 GB)
    cap_a216 = evaluate_capacity(
        physical_memory_gb=216.0,
        params_b=100.0,
        kv_cache_gb=0.0,
        bits_per_param=4.5,
        runtime_reserve_gb=4.0,
    )
    assert cap_a216.fits_in_memory is True
    assert cap_a216.headroom_gb > 0.0


def test_capacity_invalid_and_impossible_inputs():
    # Non-positive weights
    with pytest.raises(ValueError):
        calculate_stored_weights_gb(params_b=-1.0, bits_per_param=4.5)
    with pytest.raises(ValueError):
        calculate_stored_weights_gb(params_b=32.0, bits_per_param=0.0)

    # Negative physical memory or reserve
    with pytest.raises(ValueError):
        calculate_usable_memory_gb(physical_memory_gb=0.0, runtime_reserve_gb=4.0)
    with pytest.raises(ValueError):
        calculate_usable_memory_gb(physical_memory_gb=48.0, runtime_reserve_gb=-1.0)

    # Impossible: reserve equal to or exceeding physical memory
    with pytest.raises(ValueError, match="strictly less than"):
        calculate_usable_memory_gb(physical_memory_gb=48.0, runtime_reserve_gb=48.0)
    with pytest.raises(ValueError, match="strictly less than"):
        calculate_usable_memory_gb(physical_memory_gb=48.0, runtime_reserve_gb=50.0)

    # Evaluate capacity with impossible reserve
    with pytest.raises(ValueError, match="strictly less than"):
        evaluate_capacity(physical_memory_gb=48.0, params_b=32.0, kv_cache_gb=1.0, runtime_reserve_gb=48.0)
    with pytest.raises(ValueError, match="strictly less than"):
        evaluate_capacity(physical_memory_gb=48.0, params_b=32.0, kv_cache_gb=1.0, runtime_reserve_gb=60.0)

    # Negative KV cache
    with pytest.raises(ValueError):
        evaluate_capacity(physical_memory_gb=48.0, params_b=32.0, kv_cache_gb=-5.0)


def test_malformed_system_config_rejection():
    # Reject system where reserve >= physical memory
    with pytest.raises(ValueError, match="strictly less than"):
        SystemConfig(
            key="invalid_a48",
            name="Invalid A48",
            memory_capacity_gb=48.0,
            peak_bandwidth_tb_s=1.792,
            memory_bandwidth_gbs=1792.0,
            runtime_reserve_gb=50.0,
        )

    # Reject non-positive memory capacity
    with pytest.raises(ValueError, match="must be positive"):
        SystemConfig(
            key="zero_mem",
            name="Zero Mem",
            memory_capacity_gb=0.0,
            peak_bandwidth_tb_s=1.792,
            memory_bandwidth_gbs=1792.0,
        )


def test_malformed_workload_config_rejection():
    # Reject active params > total params
    with pytest.raises(ValueError, match="cannot exceed"):
        WorkloadConfig(
            key="bad_wl",
            name="Bad Workload",
            total_parameters_b=8.0,
            active_parameters_b=12.0,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )

    # Reject invalid top-k > experts
    with pytest.raises(ValueError, match="must satisfy"):
        WorkloadConfig(
            key="bad_moe",
            name="Bad MoE",
            total_parameters_b=40.0,
            active_parameters_b=10.0,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            is_moe=True,
            num_experts=8,
            experts_per_token=16,
        )
