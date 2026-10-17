"""Command Line Interface for the Accelerator Simulator."""

import argparse
import json
import os
import sys
from typing import Any

try:
    from .capacity import evaluate_capacity
    from .decode import calculate_decode_bandwidth_ceiling
    from .kv_cache import (
        KVCacheShape,
        calculate_kv_cache_per_sequence_gb,
    )
    from .moe import evaluate_moe_decode
    from .prefill import (
        calculate_prefill_tflop,
        calculate_required_prefill_throughput,
    )
except (ImportError, ValueError):
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    from accelerator_sim.capacity import evaluate_capacity
    from accelerator_sim.decode import calculate_decode_bandwidth_ceiling
    from accelerator_sim.kv_cache import (
        KVCacheShape,
        calculate_kv_cache_per_sequence_gb,
    )
    from accelerator_sim.moe import evaluate_moe_decode
    from accelerator_sim.prefill import (
        calculate_prefill_tflop,
        calculate_required_prefill_throughput,
    )


def load_json_file(file_path: str) -> dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_config_dir() -> str:
    cwd = os.getcwd()
    candidate = os.path.join(cwd, "configs")
    if os.path.isdir(candidate):
        return candidate
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidate = os.path.join(base, "configs")
    if os.path.isdir(candidate):
        return candidate
    return "configs"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A48/A216 Accelerator Research Simulator (Workstreams A & B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--system", type=str, default="A48", help="System key in systems.json (A48, A216)")
    p.add_argument("--workload", type=str, default=None, help="Preset workload key from workloads.json")
    p.add_argument("--model", choices=("dense", "moe"), default="dense", help="Model architecture type")
    p.add_argument("--params-b", type=float, default=32.0, help="Total parameters in billions")
    p.add_argument("--bits-per-param", type=float, default=4.5, help="Effective weight precision in bits")
    p.add_argument("--shared-params-b", type=float, default=0.0, help="MoE shared parameters in billions")
    p.add_argument("--experts", type=int, default=64, help="Total routed experts in MoE")
    p.add_argument("--top-k", type=int, default=2, help="Experts selected per token")
    p.add_argument("--resident-experts", type=int, default=None, help="Experts resident on accelerator")
    p.add_argument("--host-expert-bandwidth-tb-s", type=float, default=0.0, help="Host-offload bandwidth in TB/s")
    p.add_argument("--host-expert-latency-ms", type=float, default=0.0, help="Host-offload per-step latency in ms")
    p.add_argument("--layers", type=int, default=48, help="Transformer layer count")
    p.add_argument("--kv-heads", type=int, default=8, help="KV heads count")
    p.add_argument("--head-dim", type=int, default=128, help="Dimension per attention head")
    p.add_argument("--context", type=int, default=8192, help="Context length in tokens")
    p.add_argument("--batch", type=int, default=1, help="Inference batch size")
    p.add_argument("--kv-bits", type=float, default=16.0, help="KV cache precision in bits (16.0 for FP16)")
    p.add_argument("--kv-overhead-percent", type=float, default=0.0, help="KV cache memory overhead percent")
    p.add_argument("--runtime-reserve-gb", type=float, default=4.0, help="Reserved memory in decimal GB")
    p.add_argument("--effective-bandwidth-fraction", type=float, default=0.8, help="Sustained bandwidth efficiency")
    p.add_argument("--prompt-tokens", type=int, default=8192, help="Prompt token count per sequence")
    p.add_argument("--ttft-seconds", type=float, default=3.0, help="Target prefill time budget in seconds")
    p.add_argument("--peak-fp16-tflops", type=float, default=0.0, help="Peak FP16 TFLOP/s (0 disables ceiling)")
    p.add_argument("--compute-utilization", type=float, default=0.5, help="Effective prefill compute utilization")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg_dir = find_config_dir()
    systems_data = load_json_file(os.path.join(cfg_dir, "systems.json"))
    workloads_data = load_json_file(os.path.join(cfg_dir, "workloads.json"))

    # System lookup
    sys_key = args.system.upper()
    matching_keys = [k for k in systems_data if k.upper() == sys_key]
    if not matching_keys:
        raise ValueError(f"System '{args.system}' not found. Available: {list(systems_data.keys())}")
    sys_cfg = systems_data[matching_keys[0]]

    # Workload preset override if provided
    params_b = args.params_b
    model_type = args.model
    layers = args.layers
    kv_heads = args.kv_heads
    head_dim = args.head_dim
    bits_per_param = args.bits_per_param
    shared_params_b = args.shared_params_b
    experts = args.experts
    top_k = args.top_k

    attention_type = "gqa"

    if args.workload:
        if args.workload not in workloads_data:
            raise ValueError(f"Workload '{args.workload}' not found. Available: {list(workloads_data.keys())}")
        wl = workloads_data[args.workload]
        if not wl.get("is_complete", True) or wl.get("attention_type") in ("mla", "latent"):
            raise ValueError(
                f"Workload '{args.workload}' is marked incomplete ({wl.get('status', 'incomplete')}) "
                f"with attention_type='{wl.get('attention_type')}'. Evaluation is deferred until dedicated "
                "architecture modeling is implemented."
            )
        params_b = wl.get("total_parameters_b", params_b)
        model_type = "moe" if wl.get("is_moe", False) else "dense"
        layers = wl.get("num_layers", layers)
        kv_heads = wl.get("num_kv_heads", kv_heads)
        head_dim = wl.get("head_dim", head_dim)
        bits_per_param = wl.get("bits_per_param", bits_per_param)
        shared_params_b = wl.get("shared_params_b", shared_params_b)
        experts = wl.get("num_experts", experts)
        top_k = wl.get("experts_per_token", top_k)
        attention_type = wl.get("attention_type", "gqa")

    # Hardware specs
    physical_mem_gb = float(sys_cfg["memory_capacity_gb"])
    peak_bw_tb_s = float(sys_cfg["peak_bandwidth_tb_s"])
    eff_bw_frac = args.effective_bandwidth_fraction
    sustained_bw_tb_s = peak_bw_tb_s * eff_bw_frac
    runtime_reserve = args.runtime_reserve_gb

    # KV Cache Sizing
    kv_shape = KVCacheShape(
        num_layers=layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        precision_bits=args.kv_bits,
        overhead_percent=args.kv_overhead_percent,
        attention_type=attention_type,
    )
    kv_per_seq_gb = calculate_kv_cache_per_sequence_gb(kv_shape, args.context)
    kv_total_gb = kv_per_seq_gb * args.batch

    print("=" * 70)
    print(f" ACCELERATOR SIMULATOR: {sys_cfg.get('name', args.system)}")
    print(f" Topology: {sys_cfg.get('topology', 'N/A')}")
    print(f" Memory: {physical_mem_gb:.1f} GB | Peak BW: {peak_bw_tb_s:.3f} TB/s | Sustained: {sustained_bw_tb_s:.3f} TB/s ({eff_bw_frac:.0%})")
    print("=" * 70)

    # Capacity Evaluation
    if model_type == "dense":
        cap_eval = evaluate_capacity(
            physical_memory_gb=physical_mem_gb,
            params_b=params_b,
            kv_cache_gb=kv_total_gb,
            bits_per_param=bits_per_param,
            runtime_reserve_gb=runtime_reserve,
        )
        print(f"\n[1] MEMORY CAPACITY EVALUATION (Dense {params_b:.1f}B)")
        print(f"  - Stored Weights:       {cap_eval.weights_gb:.3f} GB ({bits_per_param:.2f} bits/param)")
        print(f"  - KV Cache per sequence:{kv_per_seq_gb:.3f} GB (Context: {args.context}, {args.kv_bits:g}-bit)")
        print(f"  - Total Batch KV:       {kv_total_gb:.3f} GB (Batch: {args.batch})")
        print(f"  - Model Footprint:      {cap_eval.model_footprint_gb:.3f} GB")
        print(f"  - Runtime Reserve:      {cap_eval.runtime_reserve_gb:.2f} GB")
        print(f"  - Physical Memory:      {cap_eval.physical_memory_gb:.1f} GB")
        print(f"  - Usable Memory:        {cap_eval.usable_memory_gb:.3f} GB")
        print(f"  - Usable Headroom:      {cap_eval.headroom_gb:.3f} GB")
        print(f"  - Capacity Status:      {'FITS IN ACCELERATOR MEMORY' if cap_eval.fits_in_memory else 'DOES NOT FIT (OOM)'}")

        # Decode Ceiling
        dec_ceiling = calculate_decode_bandwidth_ceiling(
            peak_bandwidth_tb_s=peak_bw_tb_s,
            effective_bandwidth_fraction=eff_bw_frac,
            weight_traffic_gb=cap_eval.weights_gb,
            kv_cache_per_sequence_gb=kv_per_seq_gb,
            batch_size=args.batch,
        )
        print("\n[2] DECODE BANDWIDTH CEILING (Dense)")
        print(f"  - Sustained Bandwidth:  {dec_ceiling.sustained_bandwidth_gb_s:.1f} GB/s")
        print(f"  - Weight Traffic/step:  {dec_ceiling.weight_traffic_gb:.3f} GB (shared once per batch)")
        print(f"  - KV Traffic/step:      {dec_ceiling.kv_traffic_batch_gb:.3f} GB (all sequences)")
        print(f"  - Total Step Traffic:   {dec_ceiling.total_step_traffic_gb:.3f} GB")
        print(f"  - Aggregate Throughput: {dec_ceiling.aggregate_tps:.2f} tokens/s")
        print(f"  - Per-Sequence Latency: {dec_ceiling.per_sequence_tps:.2f} tokens/s/sequence")

        # Prefill Evaluation
        prefill_tflop = calculate_prefill_tflop(params_b, args.prompt_tokens, args.batch)
        req_tflops = calculate_required_prefill_throughput(prefill_tflop, args.ttft_seconds)
        print(f"\n[3] PREFILL COMPUTE ESTIMATE (Prompt: {args.prompt_tokens:,} tokens/seq, Batch: {args.batch})")
        print(f"  - Dense-equivalent Work:{prefill_tflop:.3f} TFLOP (2 * P * S * B)")
        print(f"  - Target TTFT Budget:   {args.ttft_seconds:.2f} s")
        print(f"  - Required Throughput:  {req_tflops:.1f} effective TFLOP/s")

    else:
        # MoE Architecture
        resident = experts if args.resident_experts is None else args.resident_experts
        moe_res = evaluate_moe_decode(
            total_experts=experts,
            resident_experts=resident,
            top_k=top_k,
            shared_params_b=shared_params_b,
            total_params_b=params_b,
            bits_per_param=bits_per_param,
            batch_size=args.batch,
            kv_cache_per_sequence_gb=kv_per_seq_gb,
            card_sustained_bw_tb_s=sustained_bw_tb_s,
            host_bw_tb_s=args.host_expert_bandwidth_tb_s,
            host_latency_ms=args.host_expert_latency_ms,
        )
        print(f"\n[1] MIXTURE-OF-EXPERTS TOPOLOGY & UNION (MoE {params_b:.1f}B)")
        print(f"  - Routed Experts:       {experts} total, top-{top_k} active/token")
        print(f"  - Resident Experts:     {resident} / {experts} on-chip")
        print(f"  - Expected Batch Union: {moe_res.expected_total_union:.4f} unique experts")
        print(f"  - Expected Resident:    {moe_res.expected_resident_union:.4f} unique experts")
        print(f"  - Expected Nonresident: {moe_res.expected_nonresident_union:.4f} unique experts")

        print("\n[2] DECODE THROUGHPUT CEILING (MoE)")
        print(f"  - Card Weight Traffic:  {moe_res.card_weight_traffic_gb:.3f} GB/step")
        print(f"  - Host Weight Traffic:  {moe_res.host_weight_traffic_gb:.3f} GB/step")
        print(f"  - Card Aggregate TPS:   {moe_res.card_aggregate_tps:.2f} tokens/s")
        if resident < experts and args.host_expert_bandwidth_tb_s == 0.0:
            print("  - Host Aggregate TPS:   NOT COMPUTED (supply --host-expert-bandwidth-tb-s)")
        else:
            print(f"  - Host Aggregate TPS:   {moe_res.host_aggregate_tps:.2f} tokens/s")
        print(f"  - Combined Decode TPS:  {moe_res.combined_aggregate_tps:.2f} tokens/s ({moe_res.per_sequence_tps:.2f} tok/s/seq)")
        print(f"  - Primary Limiter:      {moe_res.limiting_factor}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
