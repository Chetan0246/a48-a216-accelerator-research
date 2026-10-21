import pytest

from accelerator_sim.cli import main


def test_cli_dense_reference_run(capsys):
    test_args = [
        "--system", "A48",
        "--model", "dense",
        "--params-b", "32.0",
        "--bits-per-param", "4.5",
        "--context", "8192",
        "--batch", "1",
        "--prompt-tokens", "8192",
        "--ttft-seconds", "3.0",
        "--runtime-reserve-gb", "4.0",
    ]
    main(test_args)
    captured = capsys.readouterr().out
    assert "ACCELERATOR SIMULATOR: A48 Base" in captured
    assert "18.000 GB" in captured
    assert "1.611 GB" in captured or "1.610 GB" in captured
    assert "524.288 TFLOP" in captured
    assert "174.8 effective TFLOP/s" in captured
    assert "73.10 tokens/s" in captured


def test_cli_workload_preset_run(capsys):
    test_args = [
        "--system", "A48",
        "--workload", "dense-32b",
        "--batch", "1",
    ]
    main(test_args)
    captured = capsys.readouterr().out
    assert "Dense 32.0B" in captured
    assert "18.000 GB" in captured


def test_cli_moe_run(capsys):
    test_args = [
        "--system", "A48",
        "--model", "moe",
        "--params-b", "40.0",
        "--shared-params-b", "8.0",
        "--experts", "64",
        "--top-k", "2",
        "--batch", "4",
    ]
    main(test_args)
    captured = capsys.readouterr().out
    assert "MIXTURE-OF-EXPERTS TOPOLOGY & UNION" in captured
    # Batch 4 union for 64 exp, top-2 should be approx 7.6328
    assert "7.6328" in captured


def test_cli_incomplete_workload_rejection():
    test_args = [
        "--system", "A48",
        "--workload", "deepseek-v2-lite",
    ]
    with pytest.raises(ValueError, match="incomplete.*mla"):
        main(test_args)


def test_cli_qwen_preset_run(capsys):
    test_args = [
        "--system", "A48",
        "--workload", "qwen-3.6-35b-a3b",
        "--batch", "4",
    ]
    main(test_args)
    captured = capsys.readouterr().out
    assert "ACCELERATOR SIMULATOR: A48 Base" in captured
    assert "19.688 GB" in captured
    assert "Gated Attention KV (10 layers, 16-bit):   0.168 GB/seq" in captured
    assert "Gated DeltaNet State (30 layers, 32-bit): 0.063 GB/seq" in captured
    assert "Causal Conv1D State (30 layers, 16-bit):  0.001 GB/seq" in captured
    assert "Total Persistent State:                     0.232 GB/seq (0.929 GB batch)" in captured
    assert "20.616 GB" in captured
    assert "MIXTURE-OF-EXPERTS TOPOLOGY & UNION (MoE 35.0B)" in captured
    assert "256 total, top-8 active/token" in captured
    assert "30.5310" in captured
    assert "DECODE THROUGHPUT CEILING (MoE - Hybrid Cache Traffic)" in captured
    assert "Card Weight Traffic Breakdown:" in captured
    assert "Streamed Shared Weights: 1.091 GB/step (1.94B params: backbone + LM head)" in captured
    assert "Output LM Head:   0.287 GB/step" in captured
    assert "Active Routed Experts:   2.160 GB/step" in captured
    assert "Vision Tower Weights:    0.000 GB/step (excluded in text-only mode)" in captured
    assert "1.186 GB/step" in captured
    assert "1292.21 tokens/s" in captured


def test_cli_phase_aware_and_prefill_breakdown(capsys):
    """Tests CLI execution with explicit logit-positions, vision-encoder execution, and disjoint columns."""
    # 1. Next-token TTFT prediction (logit_positions=1)
    test_args_ttft = [
        "--system", "A48",
        "--workload", "qwen-3.6-35b-a3b",
        "--batch", "4",
        "--prompt-tokens", "8192",
        "--logit-positions", "1",
    ]
    main(test_args_ttft)
    captured_ttft = capsys.readouterr().out
    assert "PREFILL COMPUTE ESTIMATE (Prompt: 8,192 tokens/seq, Batch: 4, Logit Positions: 1)" in captured_ttft
    assert "Backbone Work:      93.716 TFLOP" in captured_ttft
    assert "Routed Active Work: 65.946 TFLOP" in captured_ttft
    assert "Output LM Head:     0.0041 TFLOP" in captured_ttft
    assert "Embedding Lookup:   0.0 TFLOP" in captured_ttft
    assert "159.666 TFLOP" in captured_ttft
    assert "Disjoint Additive Components:" in captured_ttft
    assert "Backbone:              0.804 GB/step" in captured_ttft
    assert "Total Card Weights:    3.251 GB/step" in captured_ttft

    # 2. Execution with explicit vision encoder enabled
    test_args_vision = [
        "--system", "A48",
        "--workload", "qwen-3.6-35b-a3b",
        "--batch", "4",
        "--execute-vision-encoder",
    ]
    main(test_args_vision)
    captured_vis = capsys.readouterr().out
    assert "Vision Tower Weights:    0.197 GB/step (executed this step)" in captured_vis


def test_cli_qwen_multimodal_and_mtp_presets(capsys):
    """Tests CLI execution for explicit multimodal and MTP speculative checkpoint presets."""
    # 1. Multimodal preset
    main(["--system", "A48", "--workload", "qwen-3.6-35b-a3b-multimodal", "--batch", "4"])
    captured_mm = capsys.readouterr().out
    assert "19.688 GB" in captured_mm
    assert "One-Time Vision Stage: 1.613 TFLOP (576 image tokens/seq, 0.197 GB stage weight traffic)" in captured_mm
    assert "Vision Tower:       1.613 TFLOP" in captured_mm
    assert "161.279 TFLOP" in captured_mm
    assert "Vision Tower Weights:    0.000 GB/step (excluded in text-only mode)" in captured_mm

    # 2. MTP speculative checkpoint preset
    main(["--system", "A48", "--workload", "qwen-3.6-35b-a3b-mtp", "--batch", "4"])
    captured_mtp = capsys.readouterr().out
    assert "20.160 GB (4.50 bits/param; incl. 0.84B MTP auxiliary weights)" in captured_mtp
    assert "Speculative MTP Note: Speculative draft-verification throughput modeling is deferred" in captured_mtp




