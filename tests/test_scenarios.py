import os
import sys

import pandas as pd
import pytest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from scripts.run_scenarios import main as run_scenarios_main


@pytest.fixture(scope="module")
def generated_scenario_csvs():
    """Runs scenario generation once and provides paths to generated CSVs."""
    run_scenarios_main()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    return {
        "capacity": os.path.join(results_dir, "capacity", "scenario_capacity.csv"),
        "prefill": os.path.join(results_dir, "prefill", "scenario_prefill.csv"),
        "decode": os.path.join(results_dir, "decode", "scenario_decode.csv"),
    }


def test_scenario_capacity_csv_correctness(generated_scenario_csvs):
    """Validates scenario_capacity.csv records across deployment choices and storage constancy."""
    df_cap = pd.read_csv(generated_scenario_csvs["capacity"])

    # 1. Base non-speculative Qwen: 35.0B params = 19.688 GB, 0 MTP params
    qwen_base = df_cap[
        (df_cap["system"] == "A48")
        & (df_cap["workload"] == "qwen-3.6-35b-a3b")
        & (df_cap["batch_size"] == 4)
    ]
    assert len(qwen_base) == 1
    row_base = qwen_base.iloc[0]
    assert row_base["deployment_mode"] == "base_non_speculative"
    assert row_base["inference_mode"] == "text_only"
    assert pytest.approx(float(row_base["weights_gb"]), rel=1e-4) == 19.688
    assert float(row_base["mtp_params_b"]) == 0.0
    assert str(row_base["fits_in_memory"]).lower() in ("true", "1")

    # 2. Multimodal Qwen: Storage constancy holds identical 19.688 GB
    qwen_mm = df_cap[
        (df_cap["system"] == "A48")
        & (df_cap["workload"] == "qwen-3.6-35b-a3b-multimodal")
        & (df_cap["batch_size"] == 4)
    ]
    assert len(qwen_mm) == 1
    row_mm = qwen_mm.iloc[0]
    assert row_mm["deployment_mode"] == "base_non_speculative"
    assert row_mm["inference_mode"] == "multimodal"
    assert pytest.approx(float(row_mm["weights_gb"]), rel=1e-4) == 19.688
    assert float(row_mm["mtp_params_b"]) == 0.0

    # 3. MTP speculative checkpoint: Accounts for 0.84B MTP weights = 20.160 GB
    qwen_mtp = df_cap[
        (df_cap["system"] == "A48")
        & (df_cap["workload"] == "qwen-3.6-35b-a3b-mtp")
        & (df_cap["batch_size"] == 4)
    ]
    assert len(qwen_mtp) == 1
    row_mtp = qwen_mtp.iloc[0]
    assert row_mtp["deployment_mode"] == "speculative_mtp"
    assert row_mtp["inference_mode"] == "text_only"
    assert pytest.approx(float(row_mtp["weights_gb"]), rel=1e-4) == 20.160
    assert pytest.approx(float(row_mtp["mtp_params_b"]), rel=1e-4) == 0.84
    assert str(row_mtp["fits_in_memory"]).lower() in ("true", "1")


def test_scenario_prefill_csv_vision_stage_accounting(generated_scenario_csvs):
    """Validates prefill CSV records for one-time vision encoding work and multimodal FLOPs."""
    df_prefill = pd.read_csv(generated_scenario_csvs["prefill"])

    # 1. Text-only Qwen: vision stage traffic and vision compute are strictly 0.0
    qwen_text = df_prefill[
        (df_prefill["system"] == "A48")
        & (df_prefill["workload"] == "qwen-3.6-35b-a3b")
        & (df_prefill["batch_size"] == 4)
    ].iloc[0]
    assert float(qwen_text["vision_stage_weight_traffic_gb"]) == 0.0
    assert float(qwen_text["vision_stage_compute_tflop"]) == 0.0
    assert float(qwen_text["vision_tflop"]) == 0.0
    assert pytest.approx(float(qwen_text["prefill_tflop"]), rel=1e-3) == 159.666
    assert qwen_text["prefill_status"] == "detailed_breakdown_first_order_estimate"

    # 2. Multimodal Qwen: Charges one-time vision encoding stage and multimodal prefill FLOPs
    qwen_mm_b1 = df_prefill[
        (df_prefill["system"] == "A48")
        & (df_prefill["workload"] == "qwen-3.6-35b-a3b-multimodal")
        & (df_prefill["batch_size"] == 1)
    ].iloc[0]
    assert pytest.approx(float(qwen_mm_b1["vision_stage_weight_traffic_gb"]), rel=1e-3) == 0.197
    assert pytest.approx(float(qwen_mm_b1["vision_stage_compute_tflop"]), rel=1e-3) == 0.403
    assert pytest.approx(float(qwen_mm_b1["vision_tflop"]), rel=1e-3) == 0.403
    assert pytest.approx(float(qwen_mm_b1["prefill_tflop"]), rel=1e-3) == 40.320
    assert qwen_mm_b1["prefill_status"] == "detailed_multimodal_breakdown"

    qwen_mm_b4 = df_prefill[
        (df_prefill["system"] == "A48")
        & (df_prefill["workload"] == "qwen-3.6-35b-a3b-multimodal")
        & (df_prefill["batch_size"] == 4)
    ].iloc[0]
    assert pytest.approx(float(qwen_mm_b4["vision_stage_weight_traffic_gb"]), rel=1e-3) == 0.197
    assert pytest.approx(float(qwen_mm_b4["vision_stage_compute_tflop"]), rel=1e-3) == 1.613
    assert pytest.approx(float(qwen_mm_b4["vision_tflop"]), rel=1e-3) == 1.613
    # Total prefill TFLOP for batch 4: 159.666 text GEMMs + 1.613 vision GEMMs = 161.279 TFLOP
    assert pytest.approx(float(qwen_mm_b4["prefill_tflop"]), rel=1e-3) == 161.279


def test_scenario_decode_csv_multimodal_and_mtp(generated_scenario_csvs):
    """Validates decode CSV records: multimodal decode excludes vision weights; MTP is deferred."""
    df_decode = pd.read_csv(generated_scenario_csvs["decode"])

    # 1. Multimodal decode: vision tower weights are NOT streamed on output tokens
    qwen_mm_b4 = df_decode[
        (df_decode["system"] == "A48")
        & (df_decode["workload"] == "qwen-3.6-35b-a3b-multimodal")
        & (df_decode["batch_size"] == 4)
    ].iloc[0]
    assert float(qwen_mm_b4["vision_weight_traffic_gb"]) == 0.0
    assert pytest.approx(float(qwen_mm_b4["card_weight_traffic_gb"]), rel=1e-3) == 3.251
    assert qwen_mm_b4["decode_status"] == "supported_hybrid_modeled"

    # Verify disjoint additive components sum exactly to card_weight_traffic_gb
    additive_sum = (
        float(qwen_mm_b4["backbone_weight_traffic_gb"])
        + float(qwen_mm_b4["output_head_traffic_gb"])
        + float(qwen_mm_b4["routed_weight_traffic_gb"])
        + float(qwen_mm_b4["embedding_lookup_traffic_gb"])
        + float(qwen_mm_b4["vision_weight_traffic_gb"])
    )
    assert pytest.approx(float(qwen_mm_b4["card_weight_traffic_gb"]), rel=1e-5) == additive_sum

    # 2. MTP speculative checkpoint: decode throughput is deferred pending empirical calibration
    qwen_mtp_b4 = df_decode[
        (df_decode["system"] == "A48")
        & (df_decode["workload"] == "qwen-3.6-35b-a3b-mtp")
        & (df_decode["batch_size"] == 4)
    ].iloc[0]
    assert "deferred_speculative_mtp" in str(qwen_mtp_b4["decode_status"])
    assert str(qwen_mtp_b4["aggregate_tps"]) == "deferred"
    assert pytest.approx(float(qwen_mtp_b4["stored_weights_gb"]), rel=1e-3) == 20.160
