import os
import pytest
from accelerator_sim.calibration import BenchmarkManifest, BenchmarkRecord, CalibrationModel


def test_load_benchmark_manifest():
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


def test_calibration_model_mape():
    calib = CalibrationModel()
    calib.add_comparison(benchmark_id="bm_1", metric="tpot", measured=10.0, simulated=11.0)
    calib.add_comparison(benchmark_id="bm_2", metric="tpot", measured=20.0, simulated=18.0)

    # Errors: 10% and 10% -> MAPE = 10.0%
    assert pytest.approx(calib.compute_mape(metric="tpot"), rel=1e-4) == 10.0
