import pytest
from accelerator_sim.prefill import (
    calculate_prefill_tflop,
    calculate_required_prefill_throughput,
    estimate_prefill,
)


def test_reference_prefill_calculations():
    # An 8K-token prompt on a 32B dense model requires approx 524.3 TFLOP (exact: 524.288)
    tflop = calculate_prefill_tflop(params_b=32.0, prompt_tokens=8192, batch_size=1)
    assert pytest.approx(tflop, rel=1e-4) == 524.288

    # A 3-second prefill compute budget requires approx 174.8 TFLOP/s (exact: 174.76266...)
    req_throughput = calculate_required_prefill_throughput(prefill_tflop=tflop, target_ttft_seconds=3.0)
    assert pytest.approx(req_throughput, rel=1e-4) == 174.7627


def test_prefill_batch_scaling():
    # Batch size 4 scales FLOPs linearly
    tflop_b4 = calculate_prefill_tflop(params_b=32.0, prompt_tokens=8192, batch_size=4)
    assert pytest.approx(tflop_b4, rel=1e-4) == 4 * 524.288


def test_estimate_prefill_with_compute_roofline():
    est = estimate_prefill(
        params_b=32.0,
        prompt_tokens=8192,
        target_ttft_seconds=3.0,
        batch_size=1,
        peak_tflops=300.0,
        compute_utilization=0.5,
    )
    assert pytest.approx(est.dense_equivalent_tflop, rel=1e-4) == 524.288
    assert pytest.approx(est.required_effective_tflops, rel=1e-4) == 174.7627
    # 524.288 / (300 * 0.5) = 524.288 / 150 = 3.49525...
    assert pytest.approx(est.idealized_ttft_seconds, rel=1e-4) == 524.288 / 150.0


def test_prefill_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_prefill_tflop(params_b=0.0, prompt_tokens=8192)
    with pytest.raises(ValueError):
        calculate_required_prefill_throughput(prefill_tflop=500.0, target_ttft_seconds=0.0)
    with pytest.raises(ValueError):
        estimate_prefill(params_b=32.0, prompt_tokens=8192, peak_tflops=300.0, compute_utilization=1.5)
