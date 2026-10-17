#!/usr/bin/env python3
"""A48/A216 workload + first-order roofline simulator (Workstreams A/B only).

This is an analytical model, not a cycle-accurate simulator or a performance claim.
Capacity is reported in decimal GB; KV calculations are exact for the supplied
standard-K/V shape. MoE expert-union is an expectation under uniform independent
routing, not a trace-based estimate. Supply measured/assumed compute peaks with
--peak-fp16-tflops / --peak-fp8-tflops / --peak-fp4-tflops to activate a prefill
compute roofline. No physical-design or cost model is included.
"""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class System:
    key: str
    name: str
    memory_gb: float
    peak_bandwidth_tb_s: float
    sram_mib: float
    topology: str

SYSTEMS = {
    # A48 assumes 512-bit GDDR7 at 28 Gb/s: 512/8 * 28 Gb/s = 1.792 TB/s.
    "a48": System("a48", "A48 Base", 48.0, 1.792, 160.0,
                  "single compute die; 48 GB GDDR7; 512-bit @ 28 Gb/s (target assumption)"),
    # Conservative stack-bandwidth case; 19.8 TB/s is an optimistic 6 x 3.3 TB/s case.
    "a216": System("a216", "A216 Advanced", 216.0, 16.8, 240.0,
                   "3 compute chiplets; 6 x 36 GB HBM4 stacks; 2 stacks/chiplet (logical target)"),
}

def kv_gb(layers: int, kv_heads: int, head_dim: int, context: int, batch: int, bits: float) -> float:
    """Standard K+V cache, decimal GB."""
    return (2 * layers * kv_heads * head_dim * context * batch * bits / 8) / 1e9

def stored_gb(params_b: float, bits_per_param: float) -> float:
    # params_b is billions of parameters; bits_per_param includes quant scales/metadata.
    return params_b * bits_per_param / 8

def expected_expert_union(expert_count: int, top_k: int, batch: int) -> float:
    """Expected unique experts in a batch with uniform, independent top-k choices.

    For each token, top-k experts are assumed distinct. For any expert, the
    probability it is not selected by one token is 1-k/E; across batch it is
    (1-k/E)^batch. Sum the inclusion probabilities over E experts.
    """
    if expert_count <= 0 or top_k <= 0 or top_k > expert_count or batch <= 0:
        raise ValueError("require experts > 0, 1 <= top_k <= experts, batch > 0")
    return expert_count * (1.0 - (1.0 - top_k / expert_count) ** batch)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--system', choices=SYSTEMS, default='a48')
    p.add_argument('--model', choices=('dense','moe'), default='dense')
    p.add_argument('--params-b', type=float, default=40.0, help='total parameters, billions')
    p.add_argument('--bits-per-param', type=float, default=4.5, help='effective weight bits including scales/metadata')
    p.add_argument('--shared-params-b', type=float, default=0.0, help='MoE shared parameters, billions; if omitted for MoE, use --params-b as dense')
    p.add_argument('--experts', type=int, default=64, help='number of routed experts')
    p.add_argument('--top-k', type=int, default=2, help='experts selected per token')
    p.add_argument('--resident-experts', type=int, default=None, help='experts resident on accelerator; defaults to all experts')
    p.add_argument('--host-expert-bandwidth-tb-s', type=float, default=0.0, help='effective host-offload bandwidth; 0 disables offload performance modeling')
    p.add_argument('--host-expert-latency-ms', type=float, default=0.0, help='per decode-step host-offload latency penalty')
    p.add_argument('--layers', type=int, default=48)
    p.add_argument('--kv-heads', type=int, default=8)
    p.add_argument('--head-dim', type=int, default=128)
    p.add_argument('--context', type=int, default=4096)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--kv-bits', type=float, default=16.0)
    p.add_argument('--kv-overhead-percent', type=float, default=0.0)
    p.add_argument('--runtime-reserve-gb', type=float, default=2.0)
    p.add_argument('--effective-bandwidth-fraction', type=float, default=0.8)
    p.add_argument('--peak-fp16-tflops', type=float, default=0.0, help='assumed peak FP16/BF16 TFLOP/s; 0 disables compute ceiling')
    p.add_argument('--peak-fp8-tflops', type=float, default=0.0, help='assumed peak FP8 TFLOP/s; 0 disables compute ceiling')
    p.add_argument('--peak-fp4-tflops', type=float, default=0.0, help='assumed peak FP4 TFLOP/s; 0 disables compute ceiling')
    p.add_argument('--compute-utilization', type=float, default=0.5, help='effective fraction of selected peak compute for prefill')
    p.add_argument('--prompt-tokens', type=int, default=8192)
    p.add_argument('--ttft-seconds', type=float, default=3.0, help='target prefill time budget; excludes tokenizer and sampling')
    p.add_argument('--sweep', action='store_true', help='print context and bandwidth sweep for current workload')
    return p.parse_args()

def main():
    a = parse_args()
    if min(a.params_b, a.bits_per_param, a.layers, a.kv_heads, a.head_dim, a.context, a.batch, a.kv_bits, a.prompt_tokens, a.ttft_seconds) <= 0:
        raise SystemExit('parameters, dimensions, context, batch, precisions, prompt tokens and TTFT must be positive')
    if a.runtime_reserve_gb < 0 or a.kv_overhead_percent < 0 or a.host_expert_bandwidth_tb_s < 0 or a.host_expert_latency_ms < 0:
        raise SystemExit('reserve, overhead, host bandwidth and host latency cannot be negative')
    if not 0 < a.effective_bandwidth_fraction <= 1 or not 0 < a.compute_utilization <= 1:
        raise SystemExit('bandwidth fraction and compute utilization must be in (0,1]')
    if a.model == 'moe':
        if a.shared_params_b < 0 or a.shared_params_b >= a.params_b:
            raise SystemExit('for MoE, shared-params-b must be >=0 and < params-b')
        if not (1 <= a.top_k <= a.experts):
            raise SystemExit('top-k must be between 1 and experts')
        resident = a.experts if a.resident_experts is None else a.resident_experts
        if not (0 <= resident <= a.experts):
            raise SystemExit('resident-experts must be between 0 and experts')
    else:
        resident = 0
    s = SYSTEMS[a.system]
    # Dense or MoE parameter split.
    if a.model == 'dense':
        shared_b, routed_total_b, expert_b = a.params_b, 0.0, 0.0
        expected_union = 0.0
        active_b = a.params_b
        # All dense weights are treated as resident accelerator weights.
        device_weight_gb = stored_gb(a.params_b, a.bits_per_param)
        host_weight_gb = 0.0
        weight_traffic_gb = device_weight_gb
    else:
        shared_b = a.shared_params_b
        routed_total_b = a.params_b - shared_b
        expert_b = routed_total_b / a.experts
        expected_union = expected_expert_union(a.experts, a.top_k, a.batch)
        active_b = shared_b + a.top_k * expert_b
        resident_weight_b = shared_b + resident * expert_b
        device_weight_gb = stored_gb(resident_weight_b, a.bits_per_param)
        host_weight_gb = stored_gb((a.experts-resident) * expert_b, a.bits_per_param)
        # Shared weights are read once per batch step; each distinct resident expert once.
        # Expected unique experts in a subset of resident experts. Under uniform
        # routing, each particular expert is selected with probability 1-(1-k/E)^B.
        per_expert_inclusion = 1.0 - (1.0 - a.top_k / a.experts) ** a.batch
        resident_union = resident * per_expert_inclusion
        weight_traffic_gb = stored_gb(shared_b + resident_union * expert_b, a.bits_per_param)
        # If any expected expert traffic is offloaded, model host link separately below.
    kv_one = kv_gb(a.layers, a.kv_heads, a.head_dim, a.context, 1, a.kv_bits) * (1 + a.kv_overhead_percent/100)
    kv_total = kv_one * a.batch
    runtime = a.runtime_reserve_gb
    footprint = device_weight_gb + kv_total + runtime
    remaining = s.memory_gb - footprint
    sustained_tb_s = s.peak_bandwidth_tb_s * a.effective_bandwidth_fraction
    # On-card traffic roofline. Batch tokens share common/shared weights and union experts.
    card_step_gb = weight_traffic_gb + kv_total
    card_aggregate_tps = (sustained_tb_s * 1000 * a.batch / card_step_gb) if card_step_gb > 0 else math.inf
    # Host expert offload: expected unique nonresident experts are estimated by proportional symmetry.
    if a.model == 'moe' and resident < a.experts:
        nonresident_union = (a.experts-resident) * (1.0 - (1.0 - a.top_k / a.experts) ** a.batch)
        host_step_gb = stored_gb(nonresident_union * expert_b, a.bits_per_param)
        if a.host_expert_bandwidth_tb_s > 0 and host_step_gb > 0:
            host_tps = a.batch / (host_step_gb / (a.host_expert_bandwidth_tb_s*1000) + a.host_expert_latency_ms/1000)
        else:
            host_tps = 0.0
    else:
        nonresident_union, host_step_gb, host_tps = 0.0, 0.0, math.inf
    if host_tps == 0.0:
        aggregate_tps = 0.0 if (a.model == 'moe' and resident < a.experts) else card_aggregate_tps
        limiting = 'host offload unspecified' if (a.model == 'moe' and resident < a.experts) else 'on-card bandwidth'
    else:
        aggregate_tps = min(card_aggregate_tps, host_tps)
        limiting = 'host offload' if host_tps < card_aggregate_tps else 'on-card bandwidth'
    per_seq_tps = aggregate_tps / a.batch

    print(f'\n{s.name} — Workstreams A/B analytical estimate; not a performance guarantee')
    print(f'Topology: {s.topology}')
    print(f'Memory: {s.memory_gb:.1f} GB | peak BW assumption: {s.peak_bandwidth_tb_s:.3f} TB/s | sustained assumption: {sustained_tb_s:.3f} TB/s ({a.effective_bandwidth_fraction:.0%})')
    print(f'Workload: {a.model.upper()} | total params {a.params_b:.1f}B | effective weight format {a.bits_per_param:.2f} bits/param | batch {a.batch}')
    print(f'Weights on accelerator: {device_weight_gb:.3f} GB | weights on host tier: {host_weight_gb:.3f} GB')
    if a.model == 'moe':
        print(f'MoE: {a.experts} experts, top-{a.top_k}, {expert_b:.3f}B params/expert, {shared_b:.3f}B shared params')
        print(f'Expected distinct experts in batch: {expected_union:.2f}/{a.experts}; expected nonresident distinct experts: {nonresident_union:.2f}')
        print(f'Approx active params/token: {active_b:.3f}B (not the same as batch weight traffic)')
    print(f'KV per sequence: {kv_one:.3f} GB | batch KV: {kv_total:.3f} GB ({a.kv_bits:g}-bit KV)')
    print(f'Runtime reserve: {runtime:.2f} GB | accelerator footprint: {footprint:.3f} GB | headroom: {remaining:.3f} GB')
    print(f'Capacity: {"FITS modeled accelerator memory" if remaining >= 0 else "DOES NOT FIT modeled accelerator memory"}')
    print(f'On-card weight traffic per decode step (batch union): {weight_traffic_gb:.3f} GB')
    if a.model == 'moe' and resident < a.experts:
        print(f'Host offload traffic/step estimate: {host_step_gb:.3f} GB | host BW input: {a.host_expert_bandwidth_tb_s:.3f} TB/s | host latency input: {a.host_expert_latency_ms:.3f} ms')
    print(f'Bandwidth-only aggregate decode ceiling: {card_aggregate_tps:.1f} tokens/s across batch')
    if aggregate_tps == 0:
        print('Combined decode estimate: NOT COMPUTED — host-offload bandwidth/latency must be supplied for nonresident experts')
    else:
        print(f'Combined bandwidth estimate: {aggregate_tps:.1f} aggregate tokens/s | {per_seq_tps:.1f} tokens/s/sequence | apparent limiter: {limiting}')

    # Prefill: conventional dense-equivalent FLOPs = 2 * parameter count * prompt tokens.
    # For MoE, use active params/token as a first-order approximation, excluding attention-specific extras.
    prefill_params_b = a.params_b if a.model == 'dense' else active_b
    prefill_tflop = 2 * prefill_params_b * 1e9 * a.prompt_tokens / 1e12
    required_tflops = prefill_tflop / a.ttft_seconds
    print(f'\nPrefill estimate: {a.prompt_tokens:,} prompt tokens; dense-equivalent FLOPs ≈ {prefill_tflop:.2f} TFLOP')
    print(f'Compute required to finish that work in {a.ttft_seconds:.2f}s: {required_tflops:.1f} effective TFLOP/s')
    peak_by_precision = [('FP16/BF16',a.peak_fp16_tflops),('FP8',a.peak_fp8_tflops),('FP4',a.peak_fp4_tflops)]
    any_peak = False
    for label, peak in peak_by_precision:
        if peak > 0:
            any_peak = True
            effective = peak * a.compute_utilization
            secs = prefill_tflop / effective
            print(f'{label} prefill roofline: {peak:.1f} peak TFLOP/s × {a.compute_utilization:.0%} = {effective:.1f} effective TFLOP/s; idealized time {secs:.2f}s')
    if not any_peak:
        print('Compute roofline: not calculated; supply a precision-specific peak with --peak-fp*-tflops.')

    if a.sweep:
        print('\nContext sweep (same model/batch/precision; bandwidth-only, no host offload)')
        print('context,kv_bits,kv_GB_per_seq,footprint_GB,headroom_GB,card_aggregate_tok_s')
        for ctx in (4096,8192,32768,131072):
            k = kv_gb(a.layers,a.kv_heads,a.head_dim,ctx,1,a.kv_bits)*(1+a.kv_overhead_percent/100)
            f = device_weight_gb + k*a.batch + runtime
            t = (sustained_tb_s*1000*a.batch)/(weight_traffic_gb+k*a.batch)
            print(f'{ctx},{a.kv_bits:g},{k:.3f},{f:.3f},{s.memory_gb-f:.3f},{t:.2f}')

    print('\nModel limitations / interpretation:')
    print(' * KV formula assumes standard K/V: 2 × layers × KV heads × head dimension × context × batch.')
    print(' * MoE union assumes uniform independent routing; real routing skew and expert caching require trace-driven measurements.')
    print(' * Host offload is a simple bandwidth/latency bound; it omits PCIe protocol overhead, prefetch overlap, pinning and contention.')
    print(' * Decode roofline assumes weights are read once per batch union and each sequence reads its KV cache once per step.')
    print(' * Prefill FLOPs use 2 × parameter count × prompt tokens; attention, logits, batching, padding and kernel inefficiency are omitted.')
    print(' * Thermal/power is not a separate min() ceiling: sustained clocks and bandwidth must be derated using measured power/thermal curves.')
    print(' * MLA, sliding-window, SSM/hybrid attention, speculative decoding and multi-card scaling are workload extensions, not modeled in this version.')

if __name__ == '__main__':
    main()
