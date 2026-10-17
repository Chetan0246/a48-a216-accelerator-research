import pytest
from accelerator_sim.capacity import (
    calculate_stored_weights_gb,
    calculate_usable_memory_gb,
    evaluate_capacity,
)


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


def test_capacity_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_stored_weights_gb(params_b=-1.0, bits_per_param=4.5)
    with pytest.raises(ValueError):
        calculate_usable_memory_gb(physical_memory_gb=48.0, runtime_reserve_gb=-1.0)
    with pytest.raises(ValueError):
        evaluate_capacity(physical_memory_gb=48.0, params_b=32.0, kv_cache_gb=-5.0)
