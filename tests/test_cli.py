import io
import sys
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
