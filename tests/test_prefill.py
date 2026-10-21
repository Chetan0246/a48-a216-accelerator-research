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

    # 4. Dense-equivalent work matches sum of components (default logit_positions=1 for TTFT next-token generation)
    expected_backbone = (2.0 * 1.43 * s * b) / 1000.0
    expected_routed = (2.0 * (8 * (32.2 / 256.0)) * s * b) / 1000.0
    expected_head_next_token = (2.0 * 0.51 * 1 * b) / 1000.0
    expected_total_next_token = expected_backbone + expected_routed + expected_head_next_token

    assert pytest.approx(breakdown_text.shared_backbone_tflop, rel=1e-5) == expected_backbone
    assert pytest.approx(breakdown_text.routed_expert_tflop, rel=1e-5) == expected_routed
    assert pytest.approx(breakdown_text.output_head_tflop, rel=1e-5) == expected_head_next_token
    assert pytest.approx(breakdown_text.total_tflop, rel=1e-5) == expected_total_next_token
    assert breakdown_text.logit_positions == 1

    # Full sequence logit evaluation (logit_positions=s)
    breakdown_full = calculate_prefill_compute_breakdown(
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
        logit_positions=s,
    )
    expected_head_full = (2.0 * 0.51 * s * b) / 1000.0
    assert pytest.approx(breakdown_full.output_head_tflop, rel=1e-5) == expected_head_full
    assert breakdown_full.shared_backbone_tflop == breakdown_text.shared_backbone_tflop
    assert breakdown_full.routed_expert_tflop == breakdown_text.routed_expert_tflop
    assert pytest.approx(breakdown_full.total_tflop, rel=1e-5) == (expected_backbone + expected_routed + expected_head_full)

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
        logit_positions=1,
    )
    expected_vision_tflop = (2.0 * 0.35 * vision_tokens * b) / 1000.0
    assert pytest.approx(breakdown_mm.vision_tflop, rel=1e-5) == expected_vision_tflop
    assert pytest.approx(breakdown_mm.total_tflop, rel=1e-5) == expected_total_next_token + expected_vision_tflop

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

    # 7. Invalid logit_positions boundaries
    with pytest.raises(ValueError, match="logit_positions"):
        calculate_prefill_compute_breakdown(
            prompt_tokens=s,
            batch_size=b,
            shared_backbone_params_b=1.43,
            routed_expert_params_b=32.2,
            num_experts=256,
            experts_per_token=8,
            logit_positions=0,
        )
    with pytest.raises(ValueError, match="logit_positions"):
        calculate_prefill_compute_breakdown(
            prompt_tokens=s,
            batch_size=b,
            shared_backbone_params_b=1.43,
            routed_expert_params_b=32.2,
            num_experts=256,
            experts_per_token=8,
            logit_positions=s + 1,
        )


def test_calculate_vision_encoding_stage():
    """Validates the one-time vision encoding stage compute and weight traffic."""
    from accelerator_sim.capacity import calculate_stored_weights_gb
    from accelerator_sim.prefill import calculate_vision_encoding_stage

    # Qwen 3.6 35B vision tower: 0.35B parameters, 576 tokens/seq, batch=2
    res = calculate_vision_encoding_stage(
        vision_params_b=0.35,
        image_tokens_per_seq=576,
        bits_per_param=4.5,
        batch_size=2,
    )
    expected_traffic = calculate_stored_weights_gb(0.35, 4.5)
    expected_tflop = (2.0 * 0.35 * 576 * 2) / 1000.0
    assert pytest.approx(res.vision_weight_traffic_gb, rel=1e-5) == expected_traffic
    assert pytest.approx(res.vision_compute_tflop, rel=1e-5) == expected_tflop
    assert res.total_image_tokens == 576 * 2

    # Zero vision params returns 0 traffic and 0 compute
    res_zero = calculate_vision_encoding_stage(0.0, 576, 4.5, 2)
    assert res_zero.vision_weight_traffic_gb == 0.0
    assert res_zero.vision_compute_tflop == 0.0

    # Invalid inputs
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_vision_encoding_stage(-1.0, 576, 4.5, 2)
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_vision_encoding_stage(0.35, -10, 4.5, 2)
    with pytest.raises(ValueError, match="positive"):
        calculate_vision_encoding_stage(0.35, 576, 4.5, 0)
    with pytest.raises(ValueError, match="positive"):
        calculate_vision_encoding_stage(0.35, 576, 0.0, 2)

