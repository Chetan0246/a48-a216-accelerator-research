#!/usr/bin/env python3
"""Run evaluation scenarios across systems and workloads and save results."""

import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from accelerator_sim.capacity import (
    evaluate_capacity,
)
from accelerator_sim.decode import calculate_decode_bandwidth_ceiling
from accelerator_sim.kv_cache import (
    KVCacheShape,
    calculate_kv_cache_per_sequence_gb,
)
from accelerator_sim.models import (
    load_systems_config,
    load_workloads_config,
)
from accelerator_sim.moe import evaluate_moe_decode
from accelerator_sim.prefill import (
    calculate_prefill_tflop,
    calculate_required_prefill_throughput,
)


def main():
    configs_dir = os.path.join(PROJECT_ROOT, "configs")
    systems = load_systems_config(os.path.join(configs_dir, "systems.json"))
    workloads = load_workloads_config(os.path.join(configs_dir, "workloads.json"))

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(os.path.join(results_dir, "capacity"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "prefill"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "decode"), exist_ok=True)

    capacity_records = []
    prefill_records = []
    decode_records = []

    batch_sizes = [1, 4, 8, 16]

    for sys_name, sys_cfg in systems.items():
        physical_mem_gb = sys_cfg.memory_capacity_gb
        peak_bw_tb_s = sys_cfg.peak_bandwidth_tb_s
        eff_bw_fraction = sys_cfg.effective_bandwidth_fraction
        runtime_reserve_gb = sys_cfg.runtime_reserve_gb

        for wl_name, wl in workloads.items():
            attention_type = wl.attention_type.lower()

            if not wl.is_complete or attention_type in ("mla", "latent"):
                print(
                    f"Notice: Workload '{wl_name}' is marked incomplete ({wl.status}) "
                    f"with attention_type='{attention_type}'. Excluding from standard K/V cache and decode "
                    "calculations until dedicated architecture modeling is implemented."
                )
                continue

            params_b = wl.total_parameters_b
            bits_per_param = wl.bits_per_param
            layers = wl.num_layers
            kv_heads = wl.num_kv_heads
            head_dim = wl.head_dim
            is_moe = wl.is_moe
            experts = wl.num_experts
            top_k = wl.experts_per_token
            shared_params_b = wl.shared_params_b if wl.shared_params_b is not None else 0.0

            context_len = wl.default_context_len
            kv_shape = KVCacheShape(
                num_layers=layers,
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                precision_bits=16.0,
                attention_type=attention_type,
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
                active_params_b = wl.active_parameters_b
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
                        shared_params_b=shared_params_b,
                        total_params_b=params_b,
                        bits_per_param=bits_per_param,
                        batch_size=b,
                        kv_cache_per_sequence_gb=kv_per_seq_gb,
                        card_sustained_bw_tb_s=peak_bw_tb_s * eff_bw_fraction,
                        offload_mode="serialized",
                    )
                    decode_records.append({
                        "system": sys_name,
                        "workload": wl_name,
                        "batch_size": b,
                        "expected_active_experts": round(moe_res.expected_total_union, 4),
                        "card_weight_traffic_gb": round(moe_res.card_weight_traffic_gb, 3),
                        "aggregate_tps": round(moe_res.combined_aggregate_tps, 2),
                        "serialized_aggregate_tps": round(moe_res.serialized_aggregate_tps, 2),
                        "overlapped_aggregate_tps": round(moe_res.overlapped_aggregate_tps, 2),
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
                        "serialized_aggregate_tps": round(dec_res.aggregate_tps, 2),
                        "overlapped_aggregate_tps": round(dec_res.aggregate_tps, 2),
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
