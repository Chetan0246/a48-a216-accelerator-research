#!/usr/bin/env python3
"""Run evaluation scenarios across systems and workloads and save results."""

import json
import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from accelerator_sim.capacity import (
    calculate_stored_weights_gb,
    calculate_usable_memory_gb,
    evaluate_capacity,
)
from accelerator_sim.kv_cache import (
    KVCacheShape,
    calculate_kv_cache_per_sequence_gb,
    calculate_kv_cache_total_gb,
)
from accelerator_sim.prefill import (
    calculate_prefill_tflop,
    calculate_required_prefill_throughput,
)
from accelerator_sim.decode import calculate_decode_bandwidth_ceiling
from accelerator_sim.moe import evaluate_moe_decode


def main():
    configs_dir = os.path.join(PROJECT_ROOT, "configs")
    with open(os.path.join(configs_dir, "systems.json"), "r") as f:
        systems = json.load(f)
    with open(os.path.join(configs_dir, "workloads.json"), "r") as f:
        workloads = json.load(f)

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(os.path.join(results_dir, "capacity"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "prefill"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "decode"), exist_ok=True)

    capacity_records = []
    prefill_records = []
    decode_records = []

    batch_sizes = [1, 4, 8, 16]

    for sys_name, sys_cfg in systems.items():
        physical_mem_gb = float(sys_cfg["memory_capacity_gb"])
        peak_bw_tb_s = float(sys_cfg["peak_bandwidth_tb_s"])
        eff_bw_fraction = float(sys_cfg.get("effective_bandwidth_fraction", 0.8))
        runtime_reserve_gb = float(sys_cfg.get("runtime_reserve_gb", 4.0))

        for wl_name, wl_cfg in workloads.items():
            params_b = float(wl_cfg["total_parameters_b"])
            bits_per_param = float(wl_cfg.get("bits_per_param", 4.5))
            layers = int(wl_cfg["num_layers"])
            kv_heads = int(wl_cfg["num_kv_heads"])
            head_dim = int(wl_cfg["head_dim"])
            is_moe = bool(wl_cfg.get("is_moe", False))
            experts = int(wl_cfg.get("num_experts", 1))
            top_k = int(wl_cfg.get("experts_per_token", 1))

            context_len = int(wl_cfg.get("default_context_len", 8192))
            kv_shape = KVCacheShape(
                num_layers=layers,
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                precision_bits=16.0,
            )
            kv_per_seq_gb = calculate_kv_cache_per_sequence_gb(kv_shape, context_len)

            for b in batch_sizes:
                kv_total_gb = kv_per_seq_gb * b

                # Capacity
                cap_res = evaluate_capacity(
                    physical_memory_gb=physical_mem_gb,
                    params_b=params_b,
                    kv_cache_gb=kv_total_gb,
                    bits_per_param=bits_per_param,
                    runtime_reserve_gb=runtime_reserve_gb,
                )
                capacity_records.append({
                    "system": sys_name,
                    "workload": wl_name,
                    "batch_size": b,
                    "weights_gb": round(cap_res.weights_gb, 3),
                    "kv_cache_gb": round(cap_res.kv_cache_gb, 3),
                    "model_footprint_gb": round(cap_res.model_footprint_gb, 3),
                    "usable_memory_gb": round(cap_res.usable_memory_gb, 3),
                    "headroom_gb": round(cap_res.headroom_gb, 3),
                    "fits_in_memory": cap_res.fits_in_memory,
                })

                # Prefill
                active_params_b = float(wl_cfg.get("active_parameters_b", params_b))
                p_tflop = calculate_prefill_tflop(active_params_b, context_len, b)
                req_tflops = calculate_required_prefill_throughput(p_tflop, target_ttft_seconds=3.0)
                prefill_records.append({
                    "system": sys_name,
                    "workload": wl_name,
                    "batch_size": b,
                    "context_len": context_len,
                    "prefill_tflop": round(p_tflop, 3),
                    "required_throughput_3s_tflops": round(req_tflops, 1),
                })

                # Decode
                if is_moe:
                    moe_res = evaluate_moe_decode(
                        total_experts=experts,
                        resident_experts=experts,
                        top_k=top_k,
                        shared_params_b=0.0,
                        total_params_b=params_b,
                        bits_per_param=bits_per_param,
                        batch_size=b,
                        kv_cache_per_sequence_gb=kv_per_seq_gb,
                        card_sustained_bw_tb_s=peak_bw_tb_s * eff_bw_fraction,
                    )
                    decode_records.append({
                        "system": sys_name,
                        "workload": wl_name,
                        "batch_size": b,
                        "expected_active_experts": round(moe_res.expected_total_union, 4),
                        "card_weight_traffic_gb": round(moe_res.card_weight_traffic_gb, 3),
                        "aggregate_tps": round(moe_res.combined_aggregate_tps, 2),
                        "per_sequence_tps": round(moe_res.per_sequence_tps, 2),
                    })
                else:
                    dec_res = calculate_decode_bandwidth_ceiling(
                        peak_bandwidth_tb_s=peak_bw_tb_s,
                        effective_bandwidth_fraction=eff_bw_fraction,
                        weight_traffic_gb=cap_res.weights_gb,
                        kv_cache_per_sequence_gb=kv_per_seq_gb,
                        batch_size=b,
                    )
                    decode_records.append({
                        "system": sys_name,
                        "workload": wl_name,
                        "batch_size": b,
                        "expected_active_experts": 1.0,
                        "card_weight_traffic_gb": round(dec_res.weight_traffic_gb, 3),
                        "aggregate_tps": round(dec_res.aggregate_tps, 2),
                        "per_sequence_tps": round(dec_res.per_sequence_tps, 2),
                    })

    df_cap = pd.DataFrame(capacity_records)
    df_cap.to_csv(os.path.join(results_dir, "capacity", "scenario_capacity.csv"), index=False)

    df_prefill = pd.DataFrame(prefill_records)
    df_prefill.to_csv(os.path.join(results_dir, "prefill", "scenario_prefill.csv"), index=False)

    df_decode = pd.DataFrame(decode_records)
    df_decode.to_csv(os.path.join(results_dir, "decode", "scenario_decode.csv"), index=False)

    print(f"Scenarios successfully executed. Results saved to {results_dir}")
    print("\nSample Capacity Results:")
    print(df_cap.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
