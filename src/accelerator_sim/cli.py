"""Command Line Interface for the Accelerator Simulator."""

import argparse
import os
import sys

try:
    from .capacity import evaluate_capacity
    from .decode import calculate_decode_bandwidth_ceiling
    from .kv_cache import (
        HybridCacheConfig,
        KVCacheShape,
        calculate_hybrid_cache_breakdown,
        calculate_kv_cache_per_sequence_gb,
    )
    from .models import (
        WorkloadConfig,
        load_systems_config,
        load_workloads_config,
    )
    from .moe import evaluate_moe_decode
    from .prefill import (
        calculate_prefill_compute_breakdown,
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
        HybridCacheConfig,
        KVCacheShape,
        calculate_hybrid_cache_breakdown,
        calculate_kv_cache_per_sequence_gb,
    )
    from accelerator_sim.models import (
        WorkloadConfig,
        load_systems_config,
        load_workloads_config,
    )
    from accelerator_sim.moe import evaluate_moe_decode
    from accelerator_sim.prefill import (
        calculate_prefill_compute_breakdown,
        calculate_prefill_tflop,
        calculate_required_prefill_throughput,
    )


def find_config_dir() -> str:
    cwd = os.getcwd()
    candidate = os.path.join(cwd, "configs")
    if os.path.isdir(candidate):
        return candidate
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    p.add_argument("--routed-params-b", type=float, default=None, help="MoE routed parameters in billions")
    p.add_argument("--experts", type=int, default=64, help="Total routed experts in MoE")
    p.add_argument("--top-k", type=int, default=2, help="Experts selected per token")
    p.add_argument("--resident-experts", type=int, default=None, help="Experts resident on accelerator")
    p.add_argument("--host-expert-bandwidth-tb-s", type=float, default=0.0, help="Host-offload bandwidth in TB/s")
    p.add_argument("--host-expert-latency-ms", type=float, default=0.0, help="Host-offload per-step latency in ms")
    p.add_argument(
        "--offload-mode",
        choices=("serialized", "overlapped"),
        default="serialized",
        help="Host-offload execution bound: serialized (conservative) or overlapped (optimistic ceiling)",
    )
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
    p.add_argument(
        "--logit-positions",
        type=int,
        default=1,
        help="Number of prompt positions to compute logits for in prefill (default: 1 for next-token TTFT prediction matching Qwen logits_to_keep=1)",
    )
    p.add_argument(
        "--embedding-precision-bits",
        type=float,
        default=None,
        help="Embedding lookup precision in bits (default: from workload config or 16.0)",
    )
    p.add_argument(
        "--execute-vision-encoder",
        action="store_true",
        default=False,
        help="Include vision encoder tower execution during this step",
    )
    p.add_argument(
        "--mode",
        choices=("text_only", "multimodal"),
        default="text_only",
        help="Inference mode (text_only or multimodal)",
    )
    p.add_argument("--ttft-seconds", type=float, default=3.0, help="Target prefill time budget in seconds")
    p.add_argument("--peak-fp16-tflops", type=float, default=0.0, help="Peak FP16 TFLOP/s (0 disables ceiling)")
    p.add_argument("--compute-utilization", type=float, default=0.5, help="Effective prefill compute utilization")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg_dir = find_config_dir()
    systems_data = load_systems_config(os.path.join(cfg_dir, "systems.json"))
    workloads_data = load_workloads_config(os.path.join(cfg_dir, "workloads.json"))

    # System lookup
    sys_key = args.system.upper()
    matching_keys = [k for k in systems_data if k.upper() == sys_key]
    if not matching_keys:
        raise ValueError(f"System '{args.system}' not found. Available: {list(systems_data.keys())}")
    sys_cfg = systems_data[matching_keys[0]]

    # Workload preset override if provided
    wl = None
    params_b = args.params_b
    model_type = args.model
    layers = args.layers
    kv_heads = args.kv_heads
    head_dim = args.head_dim
    bits_per_param = args.bits_per_param
    shared_params_b = args.shared_params_b
    routed_params_b = args.routed_params_b
    experts = args.experts
    top_k = args.top_k
    attention_type = "gqa"

    if args.workload:
        if args.workload not in workloads_data:
            raise ValueError(f"Workload '{args.workload}' not found. Available: {list(workloads_data.keys())}")
        wl = workloads_data[args.workload]
        if wl.attention_type in ("mla", "latent") or wl.status == "incomplete_mla_deferred":
            raise ValueError(
                f"Workload '{args.workload}' is marked incomplete ({wl.status}) "
                f"with attention_type='{wl.attention_type}'. Evaluation is deferred until dedicated "
                "architecture modeling is implemented."
            )
        params_b = wl.total_parameters_b
        model_type = "moe" if wl.is_moe else "dense"
        layers = wl.num_layers
        kv_heads = wl.num_kv_heads
        head_dim = wl.head_dim
        bits_per_param = wl.bits_per_param
        shared_params_b = wl.shared_params_b if wl.shared_params_b is not None else 0.0
        routed_params_b = wl.routed_expert_params_b
        experts = wl.num_experts
        top_k = wl.experts_per_token
        attention_type = wl.attention_type
    elif model_type == "moe":
        # Validate manual MoE configuration through WorkloadConfig
        routed = routed_params_b if routed_params_b is not None else (params_b - shared_params_b)
        active_est = shared_params_b + top_k * (routed / float(experts))
        WorkloadConfig(
            key="cli_manual_moe",
            name="CLI Manual MoE",
            total_parameters_b=params_b,
            active_parameters_b=active_est,
            num_layers=layers,
            hidden_size=4096,
            num_attention_heads=kv_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            is_moe=True,
            num_experts=experts,
            experts_per_token=top_k,
            shared_params_b=shared_params_b,
            routed_expert_params_b=routed,
            is_complete=True,
        )

    # Hardware specs
    physical_mem_gb = sys_cfg.memory_capacity_gb
    peak_bw_tb_s = sys_cfg.peak_bandwidth_tb_s
    eff_bw_frac = args.effective_bandwidth_fraction
    sustained_bw_tb_s = peak_bw_tb_s * eff_bw_frac
    runtime_reserve = args.runtime_reserve_gb

    # Check if workload has hybrid attention whose cache is deferred
    is_hybrid = "hybrid" in attention_type.lower() or attention_type.lower() in ("linear_attention", "gated_deltanet")

    if is_hybrid:
        ha = wl.hybrid_attention if (args.workload and wl is not None and wl.hybrid_attention) else {}
        hybrid_cfg = HybridCacheConfig(
            total_layers=layers,
            gated_attention_layers=ha.get("gated_attention_layers", 10),
            gated_deltanet_layers=ha.get("linear_attention_layers", 30),
            attn_kv_heads=kv_heads,
            attn_head_dim=head_dim,
            attn_precision_bits=args.kv_bits,
            attn_overhead_percent=args.kv_overhead_percent,
            linear_key_heads=ha.get("linear_key_heads", 16),
            linear_key_head_dim=ha.get("linear_key_head_dim", 128),
            linear_value_heads=ha.get("linear_value_heads", 32),
            linear_value_head_dim=ha.get("linear_value_head_dim", 128),
            linear_state_precision_bits=ha.get("linear_state_precision_bits", 32.0),
            conv_kernel_size=ha.get("conv_kernel_size", 4),
            conv_channels=ha.get("conv_channels", 8192),
            conv_precision_bits=ha.get("conv_precision_bits", 16.0),
        )
        h_breakdown = calculate_hybrid_cache_breakdown(hybrid_cfg, args.context, args.batch)

        cap_eval = evaluate_capacity(
            physical_memory_gb=physical_mem_gb,
            params_b=params_b,
            kv_cache_gb=h_breakdown.total_state_batch_gb,
            bits_per_param=bits_per_param,
            runtime_reserve_gb=runtime_reserve,
        )

        print("=" * 70)
        print(f" ACCELERATOR SIMULATOR: {sys_cfg.name}")
        print(f" Topology: {sys_cfg.topology or 'N/A'}")
        print(f" Memory: {physical_mem_gb:.1f} GB | Peak BW: {peak_bw_tb_s:.3f} TB/s | Sustained: {sustained_bw_tb_s:.3f} TB/s ({eff_bw_frac:.0%})")
        print("=" * 70)

        print(f"\n[1] MEMORY CAPACITY EVALUATION (MoE {params_b:.1f}B - Hybrid Attention Breakdown)")
        print(f"  - Stored Weights:       {cap_eval.weights_gb:.3f} GB ({bits_per_param:.2f} bits/param)")
        print(f"  - Persistent State Breakdown (Batch: {args.batch}, Context: {args.context:,}):")
        print(f"    * Gated Attention KV ({hybrid_cfg.gated_attention_layers} layers, {hybrid_cfg.attn_precision_bits:g}-bit):   {h_breakdown.attn_kv_gb_per_seq:.3f} GB/seq ({h_breakdown.attn_kv_batch_gb:.3f} GB batch)")
        print(f"    * Gated DeltaNet State ({hybrid_cfg.gated_deltanet_layers} layers, {hybrid_cfg.linear_state_precision_bits:g}-bit): {h_breakdown.recurrent_state_gb_per_seq:.3f} GB/seq ({h_breakdown.recurrent_state_batch_gb:.3f} GB batch)")
        print(f"    * Causal Conv1D State ({hybrid_cfg.gated_deltanet_layers} layers, {hybrid_cfg.conv_precision_bits:g}-bit):  {h_breakdown.conv_state_gb_per_seq:.3f} GB/seq ({h_breakdown.conv_state_batch_gb:.3f} GB batch)")
        print(f"    * Total Persistent State:                     {h_breakdown.total_state_gb_per_seq:.3f} GB/seq ({h_breakdown.total_state_batch_gb:.3f} GB batch)")
        print(f"  - Model Footprint:      {cap_eval.model_footprint_gb:.3f} GB (Weights + Persistent State)")
        print(f"  - Runtime Reserve:      {cap_eval.runtime_reserve_gb:.2f} GB")
        print(f"  - Physical Memory:      {cap_eval.physical_memory_gb:.1f} GB")
        print(f"  - Usable Memory:        {cap_eval.usable_memory_gb:.3f} GB")
        print(f"  - Usable Headroom:      {cap_eval.headroom_gb:.3f} GB")
        print(f"  - Capacity Status:      {'FITS IN ACCELERATOR MEMORY' if cap_eval.fits_in_memory else 'DOES NOT FIT (OOM)'}")

        resident = experts if args.resident_experts is None else args.resident_experts
        streamed_shared = wl.effective_streamed_shared_params_b if (args.workload and wl is not None) else shared_params_b
        embedding_prec = (
            args.embedding_precision_bits
            if args.embedding_precision_bits is not None
            else (wl.embedding_precision_bits if (args.workload and wl is not None and wl.embedding_precision_bits is not None) else 16.0)
        )
        moe_res = evaluate_moe_decode(
            total_experts=experts,
            resident_experts=resident,
            top_k=top_k,
            shared_params_b=shared_params_b,
            total_params_b=params_b,
            bits_per_param=bits_per_param,
            batch_size=args.batch,
            kv_cache_per_sequence_gb=h_breakdown.total_state_gb_per_seq,
            card_sustained_bw_tb_s=sustained_bw_tb_s,
            host_bw_tb_s=args.host_expert_bandwidth_tb_s,
            host_latency_ms=args.host_expert_latency_ms,
            offload_mode=args.offload_mode,
            step_cache_traffic_gb=h_breakdown.total_step_cache_traffic_gb,
            streamed_shared_params_b=streamed_shared,
            shared_backbone_params_b=wl.shared_backbone_params_b if (args.workload and wl is not None) else None,
            embedding_params_b=wl.embedding_params_b if (args.workload and wl is not None) else None,
            output_head_params_b=wl.output_head_params_b if (args.workload and wl is not None) else None,
            vision_params_b=wl.vision_params_b if (args.workload and wl is not None) else None,
            hidden_size=wl.hidden_size if (args.workload and wl is not None) else None,
            mode=args.mode,
            execute_vision_encoder=args.execute_vision_encoder,
            embedding_precision_bits=embedding_prec,
        )

        print(f"\n[2] MIXTURE-OF-EXPERTS TOPOLOGY & UNION (MoE {params_b:.1f}B)")
        print(f"  - Routed Experts:       {experts} total, top-{top_k} active/token")
        print(f"  - Resident Experts:     {resident} / {experts} on-chip")
        print(f"  - Expected Batch Union: {moe_res.expected_total_union:.4f} unique experts")
        print(f"  - Expected Resident:    {moe_res.expected_resident_union:.4f} unique experts")
        print(f"  - Expected Nonresident: {moe_res.expected_nonresident_union:.4f} unique experts")

        print("\n[3] DECODE THROUGHPUT CEILING (MoE - Hybrid Cache Traffic)")
        print(f"  - Card Weight Traffic:  {moe_res.card_weight_traffic_gb:.3f} GB/step ({moe_res.card_step_time_ms:.2f} ms)")
        print("  - Card Weight Traffic Breakdown:")
        print(f"    * Streamed Shared Weights: {moe_res.streamed_shared_traffic_gb:.3f} GB/step ({moe_res.streamed_shared_params_b:.2f}B params: backbone + LM head)")
        if moe_res.output_head_traffic_gb > 0:
            print(f"      (incl. Output LM Head:   {moe_res.output_head_traffic_gb:.3f} GB/step)")
        print(f"    * Active Routed Experts:   {moe_res.routed_weight_traffic_gb:.3f} GB/step ({moe_res.expected_resident_union:.4f} unique resident experts)")
        print(f"    * Embedding Lookup:        {moe_res.embedding_lookup_traffic_gb:.6f} GB/step (sparse per-token index gather at {moe_res.embedding_precision_bits:g}-bit)")
        if moe_res.vision_weight_traffic_gb > 0:
            print(f"    * Vision Tower Weights:    {moe_res.vision_weight_traffic_gb:.3f} GB/step (executed this step)")
        else:
            print(f"    * Vision Tower Weights:    {moe_res.vision_weight_traffic_gb:.3f} GB/step (excluded in text-only mode)")
        print("    * Disjoint Additive Components:")
        print(f"        Backbone:              {moe_res.backbone_weight_traffic_gb:.3f} GB/step")
        print(f"        Output LM Head:        {moe_res.output_head_traffic_gb:.3f} GB/step")
        print(f"        Routed Weights:        {moe_res.routed_weight_traffic_gb:.3f} GB/step")
        print(f"        Embedding Lookup:      {moe_res.embedding_lookup_traffic_gb:.6f} GB/step")
        print(f"        Vision Weights:        {moe_res.vision_weight_traffic_gb:.3f} GB/step")
        print(f"        Total Card Weights:    {moe_res.card_weight_traffic_gb:.3f} GB/step")
        print("  - Cache Step Traffic Breakdown:")
        print(f"    * Attention KV Read:   {h_breakdown.step_attn_kv_read_gb:.3f} GB/step (O(S) context history)")
        print(f"    * Attention KV Write:  {h_breakdown.step_attn_kv_write_gb:.4f} GB/step (new token KV)")
        print(f"    * Recurrent State:     {h_breakdown.step_recurrent_state_traffic_gb:.3f} GB/step (read S_{{t-1}} + write S_t)")
        print(f"    * Conv1D State:        {h_breakdown.step_conv_state_traffic_gb:.4f} GB/step (read + write shifted buffer)")
        print(f"    * Total Card Cache:    {moe_res.card_cache_traffic_gb:.3f} GB/step")
        if resident < experts:
            print(f"  - Host Weight Traffic:  {moe_res.host_weight_traffic_gb:.3f} GB/step ({moe_res.host_step_time_ms:.2f} ms)")
        print(f"  - Card Aggregate TPS:   {moe_res.card_aggregate_tps:.2f} tokens/s")
        if resident < experts and args.host_expert_bandwidth_tb_s == 0.0:
            print("  - Host Aggregate TPS:   NOT COMPUTED (supply --host-expert-bandwidth-tb-s)")
        elif resident < experts:
            print(f"  - Host Aggregate TPS:   {moe_res.host_aggregate_tps:.2f} tokens/s")
        print(f"  - Serialized Bound TPS: {moe_res.serialized_aggregate_tps:.2f} tokens/s")
        print(f"  - Overlapped Bound TPS: {moe_res.overlapped_aggregate_tps:.2f} tokens/s")
        print(f"  - Selected Bound ({moe_res.offload_mode}): {moe_res.combined_aggregate_tps:.2f} tokens/s ({moe_res.per_sequence_tps:.2f} tok/s/seq)")
        print(f"  - Primary Limiter:      {moe_res.limiting_factor}")

        if wl is not None and wl.shared_backbone_params_b is not None:
            prefill_breakdown = calculate_prefill_compute_breakdown(
                prompt_tokens=args.prompt_tokens,
                batch_size=args.batch,
                shared_backbone_params_b=wl.shared_backbone_params_b,
                routed_expert_params_b=wl.routed_expert_params_b,
                num_experts=experts,
                experts_per_token=top_k,
                output_head_params_b=wl.output_head_params_b or 0.0,
                embedding_params_b=wl.embedding_params_b or 0.0,
                vision_params_b=wl.vision_params_b or 0.0,
                mode=args.mode,
                logit_positions=args.logit_positions,
            )
            prefill_tflop = prefill_breakdown.total_tflop
            req_tflops = calculate_required_prefill_throughput(prefill_tflop, args.ttft_seconds)
            print(f"\n[4] PREFILL COMPUTE ESTIMATE (Prompt: {args.prompt_tokens:,} tokens/seq, Batch: {args.batch}, Logit Positions: {args.logit_positions})")
            print(f"  - Active Compute Params:{prefill_breakdown.active_compute_params_b:.2f}B (reconciled text forward pass)")
            print(f"  - Dense-equivalent Work:{prefill_tflop:.3f} TFLOP (Detailed breakdown across categories)")
            print(f"    * Backbone Work:      {prefill_breakdown.shared_backbone_tflop:.3f} TFLOP (2 * P_backbone * S * B)")
            print(f"    * Routed Active Work: {prefill_breakdown.routed_expert_tflop:.3f} TFLOP (2 * P_routed_active * S * B)")
            print(f"    * Output LM Head:     {prefill_breakdown.output_head_tflop:.4f} TFLOP (2 * P_head * {args.logit_positions} * B; logits_to_keep={args.logit_positions})")
            print(f"    * Embedding Lookup:   {prefill_breakdown.embedding_lookup_tflop:.1f} TFLOP (sparse gather, 0 GEMM FLOPs)")
            if prefill_breakdown.vision_tflop > 0:
                print(f"    * Vision Tower:       {prefill_breakdown.vision_tflop:.3f} TFLOP")
            print(f"  - Target TTFT Budget:   {args.ttft_seconds:.2f} s")
            print(f"  - Required Throughput:  {req_tflops:.1f} effective TFLOP/s")
        else:
            effective_routed = routed_params_b if routed_params_b is not None else (params_b - shared_params_b)
            active_params = wl.active_parameters_b if (args.workload and wl is not None) else (shared_params_b + top_k * (effective_routed / float(experts)))
            prefill_tflop = calculate_prefill_tflop(active_params, args.prompt_tokens, args.batch)
            req_tflops = calculate_required_prefill_throughput(prefill_tflop, args.ttft_seconds)
            print(f"\n[4] PREFILL COMPUTE ESTIMATE (Prompt: {args.prompt_tokens:,} tokens/seq, Batch: {args.batch})")
            print(f"  - Active Compute Params:{active_params:.2f}B (reconciled text forward pass)")
            print(f"  - Dense-equivalent Work:{prefill_tflop:.3f} TFLOP (2 * P_active * S * B)")
            print(f"  - Target TTFT Budget:   {args.ttft_seconds:.2f} s")
            print(f"  - Required Throughput:  {req_tflops:.1f} effective TFLOP/s")

        print("\n" + "=" * 70)
        return

    # Standard KV Cache Sizing for supported attention types
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
    print(f" ACCELERATOR SIMULATOR: {sys_cfg.name}")
    print(f" Topology: {sys_cfg.topology or 'N/A'}")
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
        streamed_shared = wl.effective_streamed_shared_params_b if (args.workload and wl is not None) else shared_params_b
        embedding_prec = (
            args.embedding_precision_bits
            if args.embedding_precision_bits is not None
            else (wl.embedding_precision_bits if (args.workload and wl is not None and wl.embedding_precision_bits is not None) else 16.0)
        )
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
            offload_mode=args.offload_mode,
            streamed_shared_params_b=streamed_shared,
            shared_backbone_params_b=wl.shared_backbone_params_b if (args.workload and wl is not None) else None,
            embedding_params_b=wl.embedding_params_b if (args.workload and wl is not None) else None,
            output_head_params_b=wl.output_head_params_b if (args.workload and wl is not None) else None,
            vision_params_b=wl.vision_params_b if (args.workload and wl is not None) else None,
            hidden_size=wl.hidden_size if (args.workload and wl is not None) else None,
            mode=args.mode,
            execute_vision_encoder=args.execute_vision_encoder,
            embedding_precision_bits=embedding_prec,
        )
        print(f"\n[1] MIXTURE-OF-EXPERTS TOPOLOGY & UNION (MoE {params_b:.1f}B)")
        print(f"  - Routed Experts:       {experts} total, top-{top_k} active/token")
        print(f"  - Resident Experts:     {resident} / {experts} on-chip")
        print(f"  - Expected Batch Union: {moe_res.expected_total_union:.4f} unique experts")
        print(f"  - Expected Resident:    {moe_res.expected_resident_union:.4f} unique experts")
        print(f"  - Expected Nonresident: {moe_res.expected_nonresident_union:.4f} unique experts")

        print("\n[2] DECODE THROUGHPUT CEILING (MoE)")
        print(f"  - Card Weight Traffic:  {moe_res.card_weight_traffic_gb:.3f} GB/step ({moe_res.card_step_time_ms:.2f} ms)")
        print(f"  - Host Weight Traffic:  {moe_res.host_weight_traffic_gb:.3f} GB/step ({moe_res.host_step_time_ms:.2f} ms)")
        print(f"  - Card Aggregate TPS:   {moe_res.card_aggregate_tps:.2f} tokens/s")
        if resident < experts and args.host_expert_bandwidth_tb_s == 0.0:
            print("  - Host Aggregate TPS:   NOT COMPUTED (supply --host-expert-bandwidth-tb-s)")
        else:
            print(f"  - Host Aggregate TPS:   {moe_res.host_aggregate_tps:.2f} tokens/s")
        print(f"  - Serialized Bound TPS: {moe_res.serialized_aggregate_tps:.2f} tokens/s")
        print(f"  - Overlapped Bound TPS: {moe_res.overlapped_aggregate_tps:.2f} tokens/s")
        print(f"  - Selected Bound ({moe_res.offload_mode}): {moe_res.combined_aggregate_tps:.2f} tokens/s ({moe_res.per_sequence_tps:.2f} tok/s/seq)")
        print(f"  - Primary Limiter:      {moe_res.limiting_factor}")

        # Prefill Evaluation
        if wl is not None and wl.shared_backbone_params_b is not None:
            prefill_breakdown = calculate_prefill_compute_breakdown(
                prompt_tokens=args.prompt_tokens,
                batch_size=args.batch,
                shared_backbone_params_b=wl.shared_backbone_params_b,
                routed_expert_params_b=wl.routed_expert_params_b,
                num_experts=experts,
                experts_per_token=top_k,
                output_head_params_b=wl.output_head_params_b or 0.0,
                embedding_params_b=wl.embedding_params_b or 0.0,
                vision_params_b=wl.vision_params_b or 0.0,
                mode=args.mode,
                logit_positions=args.logit_positions,
            )
            prefill_tflop = prefill_breakdown.total_tflop
            req_tflops = calculate_required_prefill_throughput(prefill_tflop, args.ttft_seconds)
            print(f"\n[3] PREFILL COMPUTE ESTIMATE (Prompt: {args.prompt_tokens:,} tokens/seq, Batch: {args.batch}, Logit Positions: {args.logit_positions})")
            print(f"  - Active Compute Params:{prefill_breakdown.active_compute_params_b:.2f}B (reconciled text forward pass)")
            print(f"  - Dense-equivalent Work:{prefill_tflop:.3f} TFLOP (Detailed breakdown across categories)")
            print(f"  - Target TTFT Budget:   {args.ttft_seconds:.2f} s")
            print(f"  - Required Throughput:  {req_tflops:.1f} effective TFLOP/s")
        else:
            effective_routed = routed_params_b if routed_params_b is not None else (params_b - shared_params_b)
            active_params = wl.active_parameters_b if (args.workload and wl is not None) else (shared_params_b + top_k * (effective_routed / float(experts)))
            prefill_tflop = calculate_prefill_tflop(active_params, args.prompt_tokens, args.batch)
            req_tflops = calculate_required_prefill_throughput(prefill_tflop, args.ttft_seconds)
            print(f"\n[3] PREFILL COMPUTE ESTIMATE (Prompt: {args.prompt_tokens:,} tokens/seq, Batch: {args.batch})")
            print(f"  - Active Compute Params:{active_params:.2f}B")
            print(f"  - Dense-equivalent Work:{prefill_tflop:.3f} TFLOP (2 * P_active * S * B)")
            print(f"  - Target TTFT Budget:   {args.ttft_seconds:.2f} s")
            print(f"  - Required Throughput:  {req_tflops:.1f} effective TFLOP/s")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
