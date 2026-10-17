import os

import pytest

from accelerator_sim.calibration import (
    BenchmarkManifest,
    CalibrationModel,
)
from accelerator_sim.capacity import calculate_stored_weights_gb
from accelerator_sim.decode import calculate_decode_bandwidth_ceiling
from accelerator_sim.kv_cache import KVCacheShape, calculate_kv_cache_per_sequence_gb


def test_load_benchmark_manifest_unverified_labeling():
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "benchmarks",
        "benchmark_manifest.csv",
    )
    manifest = BenchmarkManifest.load_csv(manifest_path)
    assert len(manifest.records) >= 2

    first = manifest.records[0]
    assert first.benchmark_id == "pub_h100_001"
    assert "H100" in first.hardware_name
    assert first.measured_tpot_ms == 9.8
    assert first.source != ""
    assert first.source_url.startswith("https://")
    assert "HBM3" in first.hardware_details
    assert "tensor_parallel_size" in first.execution_settings
    # Verify provenance labeling is unverified
    assert first.verification_status == "unverified"
    assert not first.is_verified

    unverified = manifest.get_unverified_records()
    assert len(unverified) == len(manifest.records)
    verified = manifest.get_verified_records()
    assert len(verified) == 0


def test_empty_calibration_dataset_raises_value_error():
    calib = CalibrationModel()
    with pytest.raises(ValueError, match="Empty dataset cannot yield 0% error"):
        calib.compute_mape()

    # Adding a record for 'tpot' still raises if filtering for 'ttft'
    calib.add_comparison(
        benchmark_id="bm_001",
        metric="tpot",
        measured=10.0,
        simulated=12.0,
        verification_status="verified",
    )
    with pytest.raises(ValueError, match="metric='ttft'"):
        calib.compute_mape(metric="ttft")

    # If only unverified records exist, verified_only filter raises ValueError
    calib_unver = CalibrationModel()
    calib_unver.add_comparison(
        benchmark_id="bm_unver",
        metric="tpot",
        measured=10.0,
        simulated=12.0,
        verification_status="unverified",
    )
    with pytest.raises(ValueError, match="verified only"):
        calib_unver.compute_mape(metric="tpot", verified_only=True)


def test_gpu_baseline_algebraic_sanity_check():
    """Algebraic sanity check: compare analytical roofline ceiling against provisional H100 literature baseline.

    NOTE: This is an algebraic sanity check against unverified published literature records,
    NOT an empirical validation of predictive accuracy for proposed accelerators.
    """
    # Llama-3-8B FP16
    weights_gb = calculate_stored_weights_gb(params_b=8.03, bits_per_param=16.0)
    assert pytest.approx(weights_gb, rel=1e-4) == 16.06

    # 32 layers, 8 kv heads, 128 head dim, FP16, context 2048
    kv_shape = KVCacheShape(num_layers=32, num_kv_heads=8, head_dim=128, precision_bits=16.0)
    kv_seq_gb = calculate_kv_cache_per_sequence_gb(kv_shape, context_length=2048)
    expected_kv = (2 * 32 * 8 * 128 * 2 * 2048) / 1e9
    assert pytest.approx(kv_seq_gb, rel=1e-5) == expected_kv

    # H100 SXM5 theoretical bandwidth: 3.35 TB/s (3350 GB/s)
    h100_peak_tb_s = 3.350
    # 1. Theoretical ceiling at 100% bandwidth efficiency (Batch 1)
    ideal_b1 = calculate_decode_bandwidth_ceiling(
        peak_bandwidth_tb_s=h100_peak_tb_s,
        effective_bandwidth_fraction=1.0,
        weight_traffic_gb=weights_gb,
        kv_cache_per_sequence_gb=kv_seq_gb,
        batch_size=1,
    )
    # Ideal step time = total traffic / bandwidth = (16.06 + 0.2684) / 3350 = ~4.874 ms
    ideal_step_ms_b1 = (ideal_b1.total_step_traffic_gb / ideal_b1.sustained_bandwidth_gb_s) * 1000.0
    assert pytest.approx(ideal_step_ms_b1, rel=1e-3) == 4.874

    # Measured TPOT from manifest: 9.8 ms (unverified literature record)
    measured_tpot_b1 = 9.8
    empirical_eff_b1 = ideal_step_ms_b1 / measured_tpot_b1
    # Observed sustained efficiency on provisional literature baseline is ~49.7%
    assert pytest.approx(empirical_eff_b1, rel=1e-2) == 0.497

    # 2. When derated with this empirical factor (eta = 0.4973), predicted TPOT matches provisional value
    calibrated_b1 = calculate_decode_bandwidth_ceiling(
        peak_bandwidth_tb_s=h100_peak_tb_s,
        effective_bandwidth_fraction=empirical_eff_b1,
        weight_traffic_gb=weights_gb,
        kv_cache_per_sequence_gb=kv_seq_gb,
        batch_size=1,
    )
    calibrated_tpot_ms = (1.0 / calibrated_b1.aggregate_tps) * 1000.0
    assert pytest.approx(calibrated_tpot_ms, rel=1e-3) == measured_tpot_b1


def test_calibration_model_unverified_warning_and_mape():
    calib = CalibrationModel()

    # Adding unverified record issues a UserWarning
    with pytest.warns(UserWarning, match="unverified benchmark"):
        calib.add_comparison(
            benchmark_id="bm_unver",
            metric="tpot",
            measured=10.0,
            simulated=11.0,
            verification_status="unverified",
        )

    # Adding verified record does not issue a warning
    calib.add_comparison(
        benchmark_id="bm_ver",
        metric="tpot",
        measured=20.0,
        simulated=18.0,
        verification_status="verified",
    )

    # Overall MAPE: average of 10% and 10% = 10%
    assert pytest.approx(calib.compute_mape(metric="tpot"), rel=1e-4) == 10.0

    # Verified-only MAPE: only bm_ver (10%)
    assert pytest.approx(calib.compute_mape(metric="tpot", verified_only=True), rel=1e-4) == 10.0

    # Non-finite inputs are rejected
    with pytest.raises(ValueError, match="must be a finite number"):
        calib.add_comparison("bad", "tpot", float("nan"), 10.0)
    with pytest.raises(ValueError, match="must be a finite number"):
        calib.add_comparison("bad", "tpot", 10.0, float("inf"))

