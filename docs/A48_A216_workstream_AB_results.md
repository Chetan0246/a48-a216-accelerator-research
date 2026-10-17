# A48 / A216 — Workstreams A and B only

**Status:** analytical simulator v2; initial benchmark calibration candidates collected; not a hardware performance prediction.

## 1. Scope discipline

This phase does **not** begin floorplanning, package design, process-node selection, power-delivery design, or BOM validation. Tile counts and SRAM capacities remain hypotheses. They should be revisited only after the workload and prefill/decode model says what compute and memory balance is required.

## 2. Anchor calculations

Assumptions for the standard-KV examples below: 48 layers, 8 KV heads, head dimension 128, 4.5 effective weight bits/parameter including scales/metadata, 80% of peak bandwidth, A48 memory bandwidth 1.792 TB/s (512-bit × 28 Gb/s), no runtime reserve for reproducing the user's anchor figures.

| Case | Weight footprint | KV footprint | Total modeled footprint | Bandwidth-only decode ceiling |
|---|---:|---:|---:|---:|
| A48, 40B, 4K context, FP16 KV | 22.500 GB | 0.805 GB | 23.305 GB | 61.5 tok/s |
| A48, 40B, 128K context, FP16 KV | 22.500 GB | 25.770 GB | 48.270 GB | 29.7 tok/s; does not fit before runtime reserve |
| A48, 40B, 128K context, FP8 KV | 22.500 GB | 12.885 GB | 35.385 GB | 40.5 tok/s |
| A216, dense 250B, 4K context, FP16 KV | 140.625 GB | 0.805 GB | 141.430 GB | 94.5 tok/s |

These are analytical rooflines, not predicted application speeds. Real runtime reserve makes the 128K FP16 A48 case less feasible, not more. The A216 dense-250B example demonstrates the central product thesis problem: very high bandwidth still yields only about 95 tokens/s in the idealized batch-1 weight-streaming limit, so A216 needs a stronger MoE / multi-user / throughput case to justify its price.

### Prefill compute anchor

For a dense 32B model and an 8,192-token prompt, the first-order FLOP estimate is:

`2 × 32e9 parameters × 8192 tokens = 524.288 TFLOP`

A 3-second prefill budget therefore requires about **174.8 effective TFLOP/s**. At 50% compute utilization, the corresponding peak target is about **349.5 TFLOP/s** for the relevant precision. This is the kind of workload-derived constraint that should inform compute-unit count; a label like “32 matrix engines” is not a meaningful performance specification without dimensions, clock, precision, and utilization.

## 3. Initial benchmark calibration candidates

These are useful anchors, but they are **not yet a controlled calibration set**. The model, quantization, context, speculative decoding, software version, prompt length, and host setup differ. Do not fit one universal efficiency factor to them.

| Hardware / source | Reported workload and result | Use in calibration | Caveat |
|---|---|---|---|
| RTX 5090 / llama.cpp benchmark roundup | Qwen3 32B Q4, 4K context: about 61.4 generated tok/s; Qwen3 8B Q4: about 200 tok/s | Dense decode sanity check; 32B result is near the A48 40B/4K analytical anchor despite different model size and quantization | Secondary compilation of benchmark results; exact run settings need tracing to primary run |
| RTX 5090 / llama.cpp issue with run details | Qwen3.5 27B Q4_K_M: 81.82 tok/s in one run; prompt processing also reported | Candidate for estimating sustained effective bandwidth when exact model-file size and context are matched | Issue includes multiple OS/backend runs and MTP variants; isolate the exact run before fitting |
| RTX PRO 6000 / vLLM Blackwell guide | 27B dense INT4: 75 tok/s with CUDA graphs + FlashAttention; 100 tok/s with FlashInfer + MTP + FP8 KV | Shows software and speculative-decoding effects can be large; compare plain and optimized rows separately | Guide is not an independent standardized benchmark suite; don't compare its optimized row directly to plain llama.cpp |
| MI355X / SemiAnalysis InferenceX | Llama 3.3 70B: 2,878 aggregate tok/s/GPU at 50 tok/s per user, vLLM FP4, 8K input / 1K output | High-bandwidth HBM inference reference, especially for throughput-at-interactivity | Multi-user aggregate throughput, not single-stream decode; different quantization and serving target |

### Calibration protocol

1. Select a dense model with a known exact quantized file size and one fixed context (start with Qwen3.5 27B Q4_K_M at 4K or 8K).
2. Record GPU SKU, memory bandwidth, driver, OS, backend/version, exact model hash, quantization, KV precision, prompt tokens, output tokens, batch/concurrency, speculative decoding on/off, warm-up and repeat count.
3. First compare the **weight-only** bound against measured single-stream decode. Estimate effective bandwidth as `measured_tok_s × model_weight_bytes / peak_bandwidth_bytes_s` only when KV traffic is small and the weights are streamed each step.
4. Add KV traffic/context next. Then add batch/concurrency and speculative decoding as separate cases.
5. Calibrate A216 against an HBM accelerator using a workload with public, exact configuration. Multi-user throughput must not be compared with single-user decode as if they were the same metric.

## 4. Competitive reference table (prices are indicative, not normalized)

| Product | Memory | Peak bandwidth | Price reference | Why it matters |
|---|---:|---:|---:|---|
| NVIDIA RTX 5090 | 32GB GDDR7 | 1.792 TB/s | US list price was $1,999; current retail listings can be much higher (one MSI listing around $5,000) | A48's proposed 48GB at similar bandwidth only wins if its price and supported model capacity are compelling |
| NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 96GB GDDR7 ECC | 1.792 TB/s | NVIDIA Marketplace was reported at $16,000 in Aug 2026; availability/retailer prices vary | Professional capacity/ECC competitor; its price is a warning against assuming workstation cards are cheap |
| AMD Ryzen AI Max+ 395 / Strix Halo 128GB systems | Up to 128GB unified LPDDR5X | 256 GB/s theoretical | Around $3,999 for the Ryzen AI Halo developer system in the US; OEM systems vary | Competes on model capacity and integrated system convenience, not decode bandwidth |
| NVIDIA DGX Spark | 128GB unified LPDDR5X | 273 GB/s | $4,699 US NVIDIA Marketplace listing | Software-integrated local AI system with much more memory than a consumer GPU, but lower memory bandwidth |
| Apple Mac Studio M5 Ultra | 96GB base unified memory; configurable to 256GB/512GB | 1.2 TB/s | Starts at ₹629,900 in India; 512GB option price may differ | Strong integrated workstation competitor; compare full-system value, software support, and model capacity |
| AMD Instinct MI355X | 288GB HBM3E | Up to 8 TB/s | No single generally applicable retail price; typically quote / system dependent | HBM accelerator reference for bandwidth and aggregate serving throughput |
| Proposed A48 Base | 48GB GDDR7 | 1.792 TB/s target | $1,499–$1,999 hypothesis, unvalidated | Must prove price advantage and capacity advantage versus 5090, while accounting for software maturity |
| Proposed A216 Advanced | 216GB HBM4 | 16.8 TB/s conservative target; 19.8 TB/s optimistic target | $9,999–$14,999 hypothesis, very high risk | Must win on MoE/throughput or other workloads; dense batch-1 decode alone is not a sufficient value case |

Sources for these references include the [RTX 5090 specification page](https://www.nvidia.com/en-in/geforce/graphics-cards/50-series/rtx-5090/), [RTX PRO 6000 product page](https://www.nvidia.com/en-in/products/workstations/professional-desktop-gpus/rtx-pro-6000/), [AMD Ryzen AI Max+ 395 page](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html), [DGX Spark specifications](https://www.nvidia.com/en-in/products/workstations/dgx-spark/), [Apple Mac Studio announcement](https://www.apple.com/in/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/), and [MI355X documentation](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi355x.html). Price references are date- and region-dependent.

## 5. What simulator v2 covers

- Capacity accounting with effective weight bits/parameter, KV precision, batch, and runtime reserve.
- Standard K/V cache formula in decimal GB.
- Dense weight-streaming decode roofline.
- MoE expected distinct-expert union across a batch under uniform independent top-k routing.
- Device-resident experts versus host-offloaded experts, with user-supplied host bandwidth and latency assumptions.
- Prefill FLOP estimate and required effective TFLOP/s for a target time-to-first-token budget.
- Optional precision-specific compute peaks and compute-utilization assumption.
- Context and bandwidth sweep.

Run examples:

```bash
# A48 40B, 4K, 16-bit KV: reproduce short-context anchor
python accelerator_roofline.py --system a48 --params-b 40 --bits-per-param 4.5 --context 4096 --runtime-reserve-gb 0

# A48 40B, 128K, FP8 KV
python accelerator_roofline.py --system a48 --params-b 40 --bits-per-param 4.5 --context 131072 --kv-bits 8 --runtime-reserve-gb 0

# 32B dense, 8K prefill, 3-second budget, 350 TFLOP/s peak at 50% utilization
python accelerator_roofline.py --system a48 --params-b 32 --bits-per-param 4.5 --prompt-tokens 8192 --ttft-seconds 3 --peak-fp8-tflops 350

# MoE: 120B total, 2B shared, 64 experts, top-4, batch 8, 16 experts resident
python accelerator_roofline.py --system a48 --model moe --params-b 120 --shared-params-b 2 --bits-per-param 4.5 --experts 64 --top-k 4 --batch 8 --resident-experts 16 --host-expert-bandwidth-tb-s 0.025 --host-expert-latency-ms 0.05
```

## 6. Explicitly not yet modeled

- MLA, sliding-window, and SSM/hybrid KV/state formulas: need architecture-specific shape parameters rather than a fake generic multiplier.
- Speculative decoding: needs draft-model traffic, acceptance length, verification cost, and separate prefill/decode measurements.
- Multi-card scaling: needs interconnect bandwidth/latency, parallelization strategy, and measured scaling efficiency.
- Thermal/power derating, process node, die area, HBM4 package feasibility, packaging allocation, and BOM: deliberately deferred until workload/model results justify moving into later workstreams.

The simulator does not use thermal as an independent `min()` throughput ceiling. Power/thermal effects should derate sustained clock and bandwidth through measured or explicitly assumed curves.
