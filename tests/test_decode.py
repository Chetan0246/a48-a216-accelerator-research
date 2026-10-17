import pytest

from accelerator_sim.decode import calculate_decode_bandwidth_ceiling


def test_reference_a48_decode_ceilings():
    # A48: peak 1.792 TB/s, 80% efficiency -> 1433.6 GB/s sustained
    # 32B model @ 4.5 bpp -> 18.00 GB weight traffic
    # 8K context standard KV -> 1.610612736 GB per sequence

    # Batch 1
    res_b1 = calculate_decode_bandwidth_ceiling(
        peak_bandwidth_tb_s=1.792,
        effective_bandwidth_fraction=0.8,
        weight_traffic_gb=18.0,
        kv_cache_per_sequence_gb=1.610612736,
        batch_size=1,
    )
    assert pytest.approx(res_b1.sustained_bandwidth_gb_s, rel=1e-5) == 1433.6
    assert pytest.approx(res_b1.total_step_traffic_gb, rel=1e-5) == 18.0 + 1.610612736
    # 1433.6 / 19.610612736 = 73.10327...
    assert pytest.approx(res_b1.aggregate_tps, rel=1e-4) == 73.1033
    assert pytest.approx(res_b1.per_sequence_tps, rel=1e-4) == 73.1033

    # Batch 8
    res_b8 = calculate_decode_bandwidth_ceiling(
        peak_bandwidth_tb_s=1.792,
        effective_bandwidth_fraction=0.8,
        weight_traffic_gb=18.0,
        kv_cache_per_sequence_gb=1.610612736,
        batch_size=8,
    )
    expected_step_traffic = 18.0 + 8 * 1.610612736
    assert pytest.approx(res_b8.total_step_traffic_gb, rel=1e-5) == expected_step_traffic
    # (1433.6 * 8) / 30.884901888 = 371.3400...
    assert pytest.approx(res_b8.aggregate_tps, rel=1e-4) == 371.3400
    assert pytest.approx(res_b8.per_sequence_tps, rel=1e-4) == 371.3400 / 8.0


def test_reference_a216_decode_ceiling():
    # A216: peak 16.8 TB/s, 80% efficiency -> 13,440 GB/s sustained
    res_a216 = calculate_decode_bandwidth_ceiling(
        peak_bandwidth_tb_s=16.8,
        effective_bandwidth_fraction=0.8,
        weight_traffic_gb=18.0,
        kv_cache_per_sequence_gb=1.610612736,
        batch_size=8,
    )
    assert pytest.approx(res_a216.sustained_bandwidth_gb_s, rel=1e-5) == 13440.0
    # (13440.0 * 8) / (18.0 + 8 * 1.610612736) = 107520 / 30.884901888 = 3481.31...
    expected_tps = (13440.0 * 8) / (18.0 + 8 * 1.610612736)
    assert pytest.approx(res_a216.aggregate_tps, rel=1e-4) == expected_tps


def test_decode_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_decode_bandwidth_ceiling(
            peak_bandwidth_tb_s=-1.0,
            effective_bandwidth_fraction=0.8,
            weight_traffic_gb=18.0,
            kv_cache_per_sequence_gb=1.6,
            batch_size=1,
        )
    with pytest.raises(ValueError):
        calculate_decode_bandwidth_ceiling(
            peak_bandwidth_tb_s=1.792,
            effective_bandwidth_fraction=1.5,
            weight_traffic_gb=18.0,
            kv_cache_per_sequence_gb=1.6,
            batch_size=1,
        )
    with pytest.raises(ValueError):
        calculate_decode_bandwidth_ceiling(
            peak_bandwidth_tb_s=1.792,
            effective_bandwidth_fraction=0.8,
            weight_traffic_gb=-1.0,
            kv_cache_per_sequence_gb=1.6,
            batch_size=1,
        )
    with pytest.raises(ValueError, match="finite number"):
        calculate_decode_bandwidth_ceiling(
            peak_bandwidth_tb_s=float("nan"),
            effective_bandwidth_fraction=0.8,
            weight_traffic_gb=18.0,
            kv_cache_per_sequence_gb=1.6,
            batch_size=1,
        )
    with pytest.raises(ValueError, match="finite number"):
        calculate_decode_bandwidth_ceiling(
            peak_bandwidth_tb_s=1.792,
            effective_bandwidth_fraction=float("inf"),
            weight_traffic_gb=18.0,
            kv_cache_per_sequence_gb=1.6,
            batch_size=1,
        )


def test_decode_with_explicit_cache_traffic():
    """calculate_decode_bandwidth_ceiling correctly uses step_cache_traffic_gb when provided."""
    # Weight traffic = 10.0 GB, explicit step cache traffic = 2.5 GB, batch 2
    res = calculate_decode_bandwidth_ceiling(
        peak_bandwidth_tb_s=1.0,
        effective_bandwidth_fraction=1.0,
        weight_traffic_gb=10.0,
        kv_cache_per_sequence_gb=0.0,
        batch_size=2,
        step_cache_traffic_gb=2.5,
    )
    # Sustained BW = 1000 GB/s. Total step traffic = 10.0 + 2.5 = 12.5 GB.
    assert pytest.approx(res.total_step_traffic_gb, rel=1e-5) == 12.5
    assert pytest.approx(res.card_cache_traffic_gb, rel=1e-5) == 2.5
    # Aggregate TPS = (1000 * 2) / 12.5 = 160.0
    assert pytest.approx(res.aggregate_tps, rel=1e-5) == 160.0
    assert pytest.approx(res.per_sequence_tps, rel=1e-5) == 80.0


