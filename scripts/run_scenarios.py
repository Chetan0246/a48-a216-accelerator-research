#!/usr/bin/env python3
"""Run evaluation scenarios across systems and workloads and save results."""

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
from accelerator_sim.decode import calculate_decode_bandwidth_ceiling
from accelerator_sim.kv_cache import (
    HybridCacheConfig,
    KVCacheShape,
    calculate_hybrid_cache_breakdown,
    calculate_kv_cache_per_sequence_gb,
)
from accelerator_sim.models import (
    load_systems_config,
    load_workloads_config,
)
from accelerator_sim.moe import evaluate_moe_decode
from accelerator_sim.prefill import (
    calculate_prefill_compute_breakdown,
    calculate_prefill_tflop,
    calculate_required_prefill_throughput,
    calculate_vision_encoding_stage,
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

    notified_workloads = set()

    for sys_name, sys_cfg in systems.items():
        physical_mem_gb = sys_cfg.memory_capacity_gb
        peak_bw_tb_s = sys_cfg.peak_bandwidth_tb_s
        eff_bw_fraction = sys_cfg.effective_bandwidth_fraction
        runtime_reserve_gb = sys_cfg.runtime_reserve_gb
        usable_mem_gb = round(calculate_usable_memory_gb(physical_mem_gb, runtime_reserve_gb), 3)

        for wl_name, wl in workloads.items():
            attention_type = wl.attention_type.lower()
            params_b = wl.total_parameters_b
            bits_per_param = wl.bits_per_param
            weights_gb = round(calculate_stored_weights_gb(params_b, bits_per_param), 3)
            weights_fit = weights_gb <= usable_mem_gb
            context_len = wl.default_context_len

            is_hybrid = "hybrid" in attention_type or attention_type in ("linear_attention", "gated_deltanet")
            is_standard = attention_type in ("mha", "gqa", "mqa")
            is_cache_supported = wl.is_complete and (is_standard or is_hybrid)

            if not is_cache_supported:
                if wl_name not in notified_workloads:
                    print(
                        f"Notice: Workload '{wl_name}' has unsupported cache architecture ('{attention_type}'). "
                        "Deferring cache-capacity and decode throughput projections; parameter-storage calculations are retained."
                    )
                    notified_workloads.add(wl_name)

                for b in batch_sizes:
                    # Capacity record with deferred cache
                    capacity_records.append({
                        "system": sys_name,
                        "workload": wl_name,
                        "deployment_mode": wl.deployment_mode,
                        "inference_mode": wl.inference_mode,
                        "batch_size": b,
                        "weights_gb": weights_gb,
                        "mtp_params_b": round(wl.mtp_params_b, 3),
                        "kv_cache_gb": "deferred",
                        "gated_attn_kv_gb": "deferred",
                        "recurrent_state_gb": "deferred",
                        "conv_state_gb": "deferred",
                        "model_footprint_gb": "deferred",
                        "usable_memory_gb": usable_mem_gb,
                        "headroom_gb": "deferred",
                        "fits_in_memory": "deferred (weights fit)" if weights_fit else "DOES NOT FIT (weights exceed memory)",
                        "cache_status": f"deferred ({wl.status})",
                    })

                    # Decode record with deferred throughput
                    decode_records.append({
                        "system": sys_name,
                        "workload": wl_name,
                        "deployment_mode": wl.deployment_mode,
                        "inference_mode": wl.inference_mode,
                        "batch_size": b,
                        "stored_weights_gb": weights_gb,
                        "expected_active_experts": "deferred",
                        "backbone_weight_traffic_gb": "deferred",
                        "output_head_traffic_gb": "deferred",
                        "routed_weight_traffic_gb": "deferred",
                        "embedding_lookup_traffic_gb": "deferred",
                        "vision_weight_traffic_gb": "deferred",
                        "card_weight_traffic_gb": "deferred",
                        "streamed_shared_traffic_gb": "deferred",
                        "card_cache_traffic_gb": "deferred",
                        "total_step_traffic_gb": "deferred",
                        "aggregate_tps": "deferred",
                        "serialized_aggregate_tps": "deferred",
                        "overlapped_aggregate_tps": "deferred",
                        "per_sequence_tps": "deferred",
                        "decode_status": f"deferred ({wl.status})",
                    })

                    # Prefill estimate is first-order based on active parameters
                    active_params_b = wl.active_parameters_b
                    p_tflop = calculate_prefill_tflop(active_params_b, context_len, b)
                    req_tflops = calculate_required_prefill_throughput(p_tflop, target_ttft_seconds=3.0)
                    prefill_records.append({
                        "system": sys_name,
                        "workload": wl_name,
                        "deployment_mode": wl.deployment_mode,
                        "inference_mode": wl.inference_mode,
                        "batch_size": b,
                        "context_len": context_len,
                        "logit_positions": context_len,
                        "active_compute_params_b": round(active_params_b, 2),
                        "shared_backbone_tflop": "deferred",
                        "routed_expert_tflop": "deferred",
                        "output_head_tflop": "deferred",
                        "embedding_lookup_tflop": "deferred",
                        "vision_tflop": "deferred",
                        "vision_stage_weight_traffic_gb": 0.0,
                        "vision_stage_compute_tflop": 0.0,
                        "prefill_tflop": round(p_tflop, 3),
                        "required_throughput_3s_tflops": round(req_tflops, 1),
                        "prefill_status": f"deferred ({wl.status})",
                    })
                continue

            if is_hybrid:
                ha = wl.hybrid_attention or {}
                hybrid_cfg = HybridCacheConfig(
                    total_layers=wl.num_layers,
                    gated_attention_layers=ha.get("gated_attention_layers", 10),
                    gated_deltanet_layers=ha.get("linear_attention_layers", 30),
                    attn_kv_heads=wl.num_kv_heads,
                    attn_head_dim=wl.head_dim,
                    attn_precision_bits=16.0,
                    linear_key_heads=ha.get("linear_key_heads", 16),
                    linear_key_head_dim=ha.get("linear_key_head_dim", 128),
                    linear_value_heads=ha.get("linear_value_heads", 32),
                    linear_value_head_dim=ha.get("linear_value_head_dim", 128),
                    linear_state_precision_bits=ha.get("linear_state_precision_bits", 32.0),
                    conv_kernel_size=ha.get("conv_kernel_size", 4),
                    conv_channels=ha.get("conv_channels", 8192),
                    conv_precision_bits=ha.get("conv_precision_bits", 16.0),
                )
                for b in batch_sizes:
                    h_breakdown = calculate_hybrid_cache_breakdown(hybrid_cfg, context_len, b)
                    cap_res = evaluate_capacity(
                        physical_memory_gb=physical_mem_gb,
                        params_b=params_b,
                        kv_cache_gb=h_breakdown.total_state_batch_gb,
                        bits_per_param=bits_per_param,
                        runtime_reserve_gb=runtime_reserve_gb,
                    )
                    capacity_records.append({
                        "system": sys_name,
                        "workload": wl_name,
                        "deployment_mode": wl.deployment_mode,
                        "inference_mode": wl.inference_mode,
                        "batch_size": b,
                        "weights_gb": round(cap_res.weights_gb, 3),
                        "mtp_params_b": round(wl.mtp_params_b, 3),
                        "kv_cache_gb": round(cap_res.kv_cache_gb, 3),
                        "gated_attn_kv_gb": round(h_breakdown.attn_kv_batch_gb, 3),
                        "recurrent_state_gb": round(h_breakdown.recurrent_state_batch_gb, 3),
                        "conv_state_gb": round(h_breakdown.conv_state_batch_gb, 3),
                        "model_footprint_gb": round(cap_res.model_footprint_gb, 3),
                        "usable_memory_gb": round(cap_res.usable_memory_gb, 3),
                        "headroom_gb": round(cap_res.headroom_gb, 3),
                        "fits_in_memory": cap_res.fits_in_memory,
                        "cache_status": "supported_hybrid_modeled",
                    })

                    # One-time vision encoding stage work
                    if wl.inference_mode == "multimodal" and wl.vision_params_b and wl.vision_params_b > 0:
                        img_tokens = wl.image_tokens_per_seq or 576
                        vis_stage = calculate_vision_encoding_stage(
                            vision_params_b=wl.vision_params_b,
                            image_tokens_per_seq=img_tokens,
                            bits_per_param=bits_per_param,
                            batch_size=b,
                        )
                        vis_stage_traffic_gb = round(vis_stage.vision_weight_traffic_gb, 3)
                        vis_stage_compute_tflop = round(vis_stage.vision_compute_tflop, 3)
                    else:
                        vis_stage_traffic_gb = 0.0
                        vis_stage_compute_tflop = 0.0

                    # Prefill (Detailed breakdown for Qwen)
                    if wl.shared_backbone_params_b is not None:
                        img_tokens = (wl.image_tokens_per_seq or 576) if wl.inference_mode == "multimodal" else 0
                        breakdown = calculate_prefill_compute_breakdown(
                            prompt_tokens=context_len,
                            batch_size=b,
                            shared_backbone_params_b=wl.shared_backbone_params_b,
                            routed_expert_params_b=wl.routed_expert_params_b,
                            num_experts=wl.num_experts,
                            experts_per_token=wl.experts_per_token,
                            output_head_params_b=wl.output_head_params_b or 0.0,
                            embedding_params_b=wl.embedding_params_b or 0.0,
                            vision_params_b=wl.vision_params_b or 0.0,
                            mode=wl.inference_mode,
                            vision_tokens_per_seq=img_tokens,
                            logit_positions=1,  # TTFT first-token generation
                        )
                        p_tflop = breakdown.total_tflop
                        req_tflops = calculate_required_prefill_throughput(p_tflop, target_ttft_seconds=3.0)
                        prefill_records.append({
                            "system": sys_name,
                            "workload": wl_name,
                            "deployment_mode": wl.deployment_mode,
                            "inference_mode": wl.inference_mode,
                            "batch_size": b,
                            "context_len": context_len,
                            "logit_positions": 1,
                            "active_compute_params_b": round(wl.active_parameters_b, 2),
                            "shared_backbone_tflop": round(breakdown.shared_backbone_tflop, 3),
                            "routed_expert_tflop": round(breakdown.routed_expert_tflop, 3),
                            "output_head_tflop": round(breakdown.output_head_tflop, 3),
                            "embedding_lookup_tflop": 0.0,
                            "vision_tflop": round(breakdown.vision_tflop, 3),
                            "vision_stage_weight_traffic_gb": vis_stage_traffic_gb,
                            "vision_stage_compute_tflop": vis_stage_compute_tflop,
                            "prefill_tflop": round(p_tflop, 3),
                            "required_throughput_3s_tflops": round(req_tflops, 1),
                            "prefill_status": (
                                "detailed_multimodal_breakdown"
                                if wl.inference_mode == "multimodal"
                                else "detailed_breakdown_first_order_estimate"
                            ),
                        })
                    else:
                        active_params_b = wl.active_parameters_b
                        p_tflop = calculate_prefill_tflop(active_params_b, context_len, b)
                        req_tflops = calculate_required_prefill_throughput(p_tflop, target_ttft_seconds=3.0)
                        prefill_records.append({
                            "system": sys_name,
                            "workload": wl_name,
                            "deployment_mode": wl.deployment_mode,
                            "inference_mode": wl.inference_mode,
                            "batch_size": b,
                            "context_len": context_len,
                            "logit_positions": context_len,
                            "active_compute_params_b": round(active_params_b, 2),
                            "shared_backbone_tflop": "n/a",
                            "routed_expert_tflop": "n/a",
                            "output_head_tflop": "n/a",
                            "embedding_lookup_tflop": 0.0,
                            "vision_tflop": 0.0,
                            "vision_stage_weight_traffic_gb": 0.0,
                            "vision_stage_compute_tflop": 0.0,
                            "prefill_tflop": round(p_tflop, 3),
                            "required_throughput_3s_tflops": round(req_tflops, 1),
                            "prefill_status": "generic_first_order_estimate",
                        })

                    # Decode (Qwen is MoE)
                    if wl.deployment_mode == "speculative_mtp":
                        decode_records.append({
                            "system": sys_name,
                            "workload": wl_name,
                            "deployment_mode": wl.deployment_mode,
                            "inference_mode": wl.inference_mode,
                            "batch_size": b,
                            "stored_weights_gb": round(cap_res.weights_gb, 3),
                            "expected_active_experts": "deferred",
                            "backbone_weight_traffic_gb": "deferred",
                            "output_head_traffic_gb": "deferred",
                            "routed_weight_traffic_gb": "deferred",
                            "embedding_lookup_traffic_gb": "deferred",
                            "vision_weight_traffic_gb": "deferred",
                            "card_weight_traffic_gb": "deferred",
                            "streamed_shared_traffic_gb": "deferred",
                            "card_cache_traffic_gb": "deferred",
                            "total_step_traffic_gb": "deferred",
                            "aggregate_tps": "deferred",
                            "serialized_aggregate_tps": "deferred",
                            "overlapped_aggregate_tps": "deferred",
                            "per_sequence_tps": "deferred",
                            "decode_status": "deferred_speculative_mtp (requires empirical acceptance rate calibration)",
                        })
                    else:
                        moe_res = evaluate_moe_decode(
                            total_experts=wl.num_experts,
                            resident_experts=wl.num_experts,
                            top_k=wl.experts_per_token,
                            shared_params_b=wl.shared_params_b if wl.shared_params_b is not None else 0.0,
                            total_params_b=params_b,
                            bits_per_param=bits_per_param,
                            batch_size=b,
                            kv_cache_per_sequence_gb=h_breakdown.total_state_gb_per_seq,
                            card_sustained_bw_tb_s=peak_bw_tb_s * eff_bw_fraction,
                            offload_mode="serialized",
                            step_cache_traffic_gb=h_breakdown.total_step_cache_traffic_gb,
                            streamed_shared_params_b=wl.effective_streamed_shared_params_b,
                            shared_backbone_params_b=wl.shared_backbone_params_b,
                            embedding_params_b=wl.embedding_params_b,
                            output_head_params_b=wl.output_head_params_b,
                            vision_params_b=wl.vision_params_b,
                            hidden_size=wl.hidden_size,
                            mode=wl.inference_mode,
                            execute_vision_encoder=False,
                            embedding_precision_bits=wl.embedding_precision_bits,
                        )
                        decode_records.append({
                            "system": sys_name,
                            "workload": wl_name,
                            "deployment_mode": wl.deployment_mode,
                            "inference_mode": wl.inference_mode,
                            "batch_size": b,
                            "stored_weights_gb": round(cap_res.weights_gb, 3),
                            "expected_active_experts": round(moe_res.expected_total_union, 4),
                            "backbone_weight_traffic_gb": round(moe_res.backbone_weight_traffic_gb, 3),
                            "output_head_traffic_gb": round(moe_res.output_head_traffic_gb, 3),
                            "routed_weight_traffic_gb": round(moe_res.routed_weight_traffic_gb, 3),
                            "embedding_lookup_traffic_gb": round(moe_res.embedding_lookup_traffic_gb, 6),
                            "vision_weight_traffic_gb": round(moe_res.vision_weight_traffic_gb, 3),
                            "card_weight_traffic_gb": round(moe_res.card_weight_traffic_gb, 3),
                            "streamed_shared_traffic_gb": round(moe_res.streamed_shared_traffic_gb, 3),
                            "card_cache_traffic_gb": round(moe_res.card_cache_traffic_gb, 3),
                            "total_step_traffic_gb": round(moe_res.card_weight_traffic_gb + moe_res.card_cache_traffic_gb, 3),
                            "aggregate_tps": round(moe_res.combined_aggregate_tps, 2),
                            "serialized_aggregate_tps": round(moe_res.serialized_aggregate_tps, 2),
                            "overlapped_aggregate_tps": round(moe_res.overlapped_aggregate_tps, 2),
                            "per_sequence_tps": round(moe_res.per_sequence_tps, 2),
                            "decode_status": "supported_hybrid_modeled",
                        })
                continue

            layers = wl.num_layers
            kv_heads = wl.num_kv_heads
            head_dim = wl.head_dim
            is_moe = wl.is_moe
            experts = wl.num_experts
            top_k = wl.experts_per_token
            shared_params_b = wl.shared_params_b if wl.shared_params_b is not None else 0.0

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
                    "deployment_mode": wl.deployment_mode,
                    "inference_mode": wl.inference_mode,
                    "batch_size": b,
                    "weights_gb": round(cap_res.weights_gb, 3),
                    "mtp_params_b": round(wl.mtp_params_b, 3),
                    "kv_cache_gb": round(cap_res.kv_cache_gb, 3),
                    "gated_attn_kv_gb": round(cap_res.kv_cache_gb, 3),
                    "recurrent_state_gb": 0.0,
                    "conv_state_gb": 0.0,
                    "model_footprint_gb": round(cap_res.model_footprint_gb, 3),
                    "usable_memory_gb": round(cap_res.usable_memory_gb, 3),
                    "headroom_gb": round(cap_res.headroom_gb, 3),
                    "fits_in_memory": cap_res.fits_in_memory,
                    "cache_status": "supported",
                })

                # Prefill
                if wl.shared_backbone_params_b is not None and is_moe:
                    breakdown = calculate_prefill_compute_breakdown(
                        prompt_tokens=context_len,
                        batch_size=b,
                        shared_backbone_params_b=wl.shared_backbone_params_b,
                        routed_expert_params_b=wl.routed_expert_params_b,
                        num_experts=wl.num_experts,
                        experts_per_token=wl.experts_per_token,
                        output_head_params_b=wl.output_head_params_b or 0.0,
                        embedding_params_b=wl.embedding_params_b or 0.0,
                        vision_params_b=wl.vision_params_b or 0.0,
                        mode="text_only",
                        logit_positions=1,
                    )
                    p_tflop = breakdown.total_tflop
                    req_tflops = calculate_required_prefill_throughput(p_tflop, target_ttft_seconds=3.0)
                    prefill_records.append({
                        "system": sys_name,
                        "workload": wl_name,
                        "deployment_mode": wl.deployment_mode,
                        "inference_mode": wl.inference_mode,
                        "batch_size": b,
                        "context_len": context_len,
                        "logit_positions": 1,
                        "active_compute_params_b": round(wl.active_parameters_b, 2),
                        "shared_backbone_tflop": round(breakdown.shared_backbone_tflop, 3),
                        "routed_expert_tflop": round(breakdown.routed_expert_tflop, 3),
                        "output_head_tflop": round(breakdown.output_head_tflop, 3),
                        "embedding_lookup_tflop": 0.0,
                        "vision_tflop": 0.0,
                        "vision_stage_weight_traffic_gb": 0.0,
                        "vision_stage_compute_tflop": 0.0,
                        "prefill_tflop": round(p_tflop, 3),
                        "required_throughput_3s_tflops": round(req_tflops, 1),
                        "prefill_status": "detailed_breakdown_first_order_estimate",
                    })
                else:
                    active_params_b = wl.active_parameters_b
                    p_tflop = calculate_prefill_tflop(active_params_b, context_len, b)
                    req_tflops = calculate_required_prefill_throughput(p_tflop, target_ttft_seconds=3.0)
                    prefill_records.append({
                        "system": sys_name,
                        "workload": wl_name,
                        "deployment_mode": wl.deployment_mode,
                        "inference_mode": wl.inference_mode,
                        "batch_size": b,
                        "context_len": context_len,
                        "logit_positions": context_len,
                        "active_compute_params_b": round(active_params_b, 2),
                        "shared_backbone_tflop": "n/a",
                        "routed_expert_tflop": "n/a",
                        "output_head_tflop": "n/a",
                        "embedding_lookup_tflop": 0.0,
                        "vision_tflop": 0.0,
                        "vision_stage_weight_traffic_gb": 0.0,
                        "vision_stage_compute_tflop": 0.0,
                        "prefill_tflop": round(p_tflop, 3),
                        "required_throughput_3s_tflops": round(req_tflops, 1),
                        "prefill_status": "generic_first_order_estimate",
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
                        streamed_shared_params_b=wl.effective_streamed_shared_params_b,
                        shared_backbone_params_b=wl.shared_backbone_params_b,
                        embedding_params_b=wl.embedding_params_b,
                        output_head_params_b=wl.output_head_params_b,
                        vision_params_b=wl.vision_params_b,
                        hidden_size=wl.hidden_size,
                        mode=wl.inference_mode,
                        execute_vision_encoder=False,
                        embedding_precision_bits=wl.embedding_precision_bits,
                    )
                    decode_records.append({
                        "system": sys_name,
                        "workload": wl_name,
                        "deployment_mode": wl.deployment_mode,
                        "inference_mode": wl.inference_mode,
                        "batch_size": b,
                        "stored_weights_gb": round(cap_res.weights_gb, 3),
                        "expected_active_experts": round(moe_res.expected_total_union, 4),
                        "backbone_weight_traffic_gb": round(moe_res.backbone_weight_traffic_gb, 3),
                        "output_head_traffic_gb": round(moe_res.output_head_traffic_gb, 3),
                        "routed_weight_traffic_gb": round(moe_res.routed_weight_traffic_gb, 3),
                        "embedding_lookup_traffic_gb": round(moe_res.embedding_lookup_traffic_gb, 6),
                        "vision_weight_traffic_gb": round(moe_res.vision_weight_traffic_gb, 3),
                        "card_weight_traffic_gb": round(moe_res.card_weight_traffic_gb, 3),
                        "streamed_shared_traffic_gb": round(moe_res.streamed_shared_traffic_gb, 3),
                        "card_cache_traffic_gb": round(moe_res.card_cache_traffic_gb, 3),
                        "total_step_traffic_gb": round(moe_res.card_weight_traffic_gb + moe_res.card_cache_traffic_gb, 3),
                        "aggregate_tps": round(moe_res.combined_aggregate_tps, 2),
                        "serialized_aggregate_tps": round(moe_res.serialized_aggregate_tps, 2),
                        "overlapped_aggregate_tps": round(moe_res.overlapped_aggregate_tps, 2),
                        "per_sequence_tps": round(moe_res.per_sequence_tps, 2),
                        "decode_status": "supported",
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
                        "deployment_mode": wl.deployment_mode,
                        "inference_mode": wl.inference_mode,
                        "batch_size": b,
                        "stored_weights_gb": round(cap_res.weights_gb, 3),
                        "expected_active_experts": 1.0,
                        "backbone_weight_traffic_gb": round(dec_res.weight_traffic_gb, 3),
                        "output_head_traffic_gb": 0.0,
                        "routed_weight_traffic_gb": 0.0,
                        "embedding_lookup_traffic_gb": 0.0,
                        "vision_weight_traffic_gb": 0.0,
                        "card_weight_traffic_gb": round(dec_res.weight_traffic_gb, 3),
                        "streamed_shared_traffic_gb": round(dec_res.weight_traffic_gb, 3),
                        "card_cache_traffic_gb": round(dec_res.card_cache_traffic_gb, 3),
                        "total_step_traffic_gb": round(dec_res.total_step_traffic_gb, 3),
                        "aggregate_tps": round(dec_res.aggregate_tps, 2),
                        "serialized_aggregate_tps": round(dec_res.aggregate_tps, 2),
                        "overlapped_aggregate_tps": round(dec_res.aggregate_tps, 2),
                        "per_sequence_tps": round(dec_res.per_sequence_tps, 2),
                        "decode_status": "supported",
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
