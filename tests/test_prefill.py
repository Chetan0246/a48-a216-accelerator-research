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
    with pytest.raises(ValueError, match="finite number"):
        calculate_prefill_tflop(params_b=float("nan"), prompt_tokens=8192)
    with pytest.raises(ValueError, match="finite number"):
        calculate_required_prefill_throughput(prefill_tflop=float("inf"), target_ttft_seconds=3.0)
    with pytest.raises(ValueError, match="finite number"):
        estimate_prefill(params_b=32.0, prompt_tokens=8192, target_ttft_seconds=float("nan"))


def test_calculate_prefill_compute_breakdown():
    """Granular prefill FLOP accounting and active compute parameter reconciliation."""
    from accelerator_sim.prefill import calculate_prefill_compute_breakdown

    # Qwen-like parameters:
    # 1.43B text backbone, 32.2B routed across 256 experts (top-8 -> 1.00625B active)
    # 0.51B output LM head, 0.51B embedding, 0.35B vision tower
    # Active compute: 1.43 + 1.00625 + 0.51 = 2.94625B (~2.94B ~ 3.0B A3B designation)
    s = 8192
    b = 1

    breakdown_text = calculate_prefill_compute_breakdown(
        prompt_tokens=s,
        batch_size=b,
        shared_backbone_params_b=1.43,
        routed_expert_params_b=32.2,
        num_experts=256,
        experts_per_token=8,
        output_head_params_b=0.51,
        embedding_params_b=0.51,
        vision_params_b=0.35,
        mode="text_only",
    )

    # 1. Active compute parameters reconciliation
    assert pytest.approx(breakdown_text.active_compute_params_b, rel=1e-4) == 2.94625
    assert pytest.approx(breakdown_text.active_compute_params_b, abs=0.06) == 3.0

    # 2. Embedding lookup has 0 GEMM FLOPs
    assert breakdown_text.embedding_lookup_tflop == 0.0

    # 3. Vision tower has 0 FLOPs in text-only mode
    assert breakdown_text.vision_tflop == 0.0

    # 4. Dense-equivalent work matches sum of components
    expected_backbone = (2.0 * 1.43 * s * b) / 1000.0
    expected_routed = (2.0 * (8 * (32.2 / 256.0)) * s * b) / 1000.0
    expected_head = (2.0 * 0.51 * s * b) / 1000.0
    expected_total = expected_backbone + expected_routed + expected_head

    assert pytest.approx(breakdown_text.shared_backbone_tflop, rel=1e-5) == expected_backbone
    assert pytest.approx(breakdown_text.routed_expert_tflop, rel=1e-5) == expected_routed
    assert pytest.approx(breakdown_text.output_head_tflop, rel=1e-5) == expected_head
    assert pytest.approx(breakdown_text.total_tflop, rel=1e-5) == expected_total

    # 5. Multimodal prefill charges vision tower GEMMs
    vision_tokens = 576  # typical 24x24 vision patches
    breakdown_mm = calculate_prefill_compute_breakdown(
        prompt_tokens=s,
        batch_size=b,
        shared_backbone_params_b=1.43,
        routed_expert_params_b=32.2,
        num_experts=256,
        experts_per_token=8,
        output_head_params_b=0.51,
        embedding_params_b=0.51,
        vision_params_b=0.35,
        mode="multimodal",
        vision_tokens_per_seq=vision_tokens,
    )
    expected_vision_tflop = (2.0 * 0.35 * vision_tokens * b) / 1000.0
    assert pytest.approx(breakdown_mm.vision_tflop, rel=1e-5) == expected_vision_tflop
    assert pytest.approx(breakdown_mm.total_tflop, rel=1e-5) == expected_total + expected_vision_tflop

    # 6. Invalid mode
    with pytest.raises(ValueError, match="mode"):
        calculate_prefill_compute_breakdown(
            prompt_tokens=s,
            batch_size=b,
            shared_backbone_params_b=1.43,
            routed_expert_params_b=32.2,
            num_experts=256,
            experts_per_token=8,
            mode="invalid",
        )

