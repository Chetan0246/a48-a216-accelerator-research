# Calibration Analysis: Algebraic Sanity Check against Provisional Literature Baseline

## 1. Objective
Establish an algebraic sanity check comparing analytical simulator roofline equations against provisional published literature figures (NVIDIA H100 SXM5 80GB running Meta-Llama-3-8B in FP16), documenting:
- Theoretical roofline ceilings vs. published empirical runtime.
- Observed effective hardware utilization factor ($\eta_{\text{eff}}$) implied by literature figures.
- Explicit boundaries: why an algebraic sanity check against provisional literature numbers **does not** constitute empirical model validation or prove predictive accuracy for proposed unbuilt accelerators (A48 Base and A216 Advanced).

---

## 2. Provisional Baseline Configuration (Unverified Literature Figures)

> [!WARNING]
> **Provisional / Unverified Status**:
> The H100 baseline records are literature-sourced reference figures marked `unverified` in `benchmarks/benchmark_manifest.csv`. They serve strictly as an algebraic sanity check for formula correctness, not as controlled, independently audited empirical validation.

| Parameter | Value / Source |
| :--- | :--- |
| **Hardware** | NVIDIA H100 SXM5 80GB HBM3 |
| **Theoretical Peak Bandwidth** | 3,350 GB/s (3.35 TB/s) |
| **Model** | Meta-Llama-3-8B (8.03B parameters, FP16 = 16.06 GB weights) |
| **Attention Architecture** | 32 layers, 8 KV heads (GQA), 128 head dim |
| **Context Length ($S$)** | 2,048 tokens |
| **Output Length** | 256 tokens |
| **Runtime Software** | vLLM v0.6.0 with FlashAttention-2 (literature reference) |
| **Measurement Conditions** | Literature reported (unverified environment details) |
| **Status in Manifest** | `unverified` (provisional literature reference) |

---

## 3. Comparison: Analytical Ceiling vs. Measured Performance

### Decode Phase (TPOT)

- **KV cache per sequence** ($S = 2048$):
  $$\text{KV}_{\text{seq}} = \frac{2 \times 32 \times 8 \times 128 \times 2 \times 2048}{10^9} = 0.2684\text{ GB}$$

- **Batch 1**:
  - Weight traffic: $16.06\text{ GB}$
  - KV traffic: $0.2684\text{ GB}$
  - Total step traffic: $16.3284\text{ GB}$
  - Theoretical minimum step time at 100% BW ($3,350\text{ GB/s}$):
    $$t_{\text{ideal}} = \frac{16.3284\text{ GB}}{3350\text{ GB/s}} = 4.874\text{ ms} \implies 205.2\text{ tok/s}$$
  - Measured TPOT on H100: **$9.8\text{ ms}$** ($102.0\text{ tok/s}$)
  - **Empirical Memory Bandwidth Utilization**:
    $$\eta_{\text{eff}} = \frac{4.874\text{ ms}}{9.8\text{ ms}} = \mathbf{49.7\%}$$

- **Batch 8**:
  - Weight traffic: $16.06\text{ GB}$
  - KV traffic ($8 \times 0.2684$): $2.1475\text{ GB}$
  - Total step traffic: $18.2075\text{ GB}$
  - Theoretical minimum step time at 100% BW:
    $$t_{\text{ideal}} = \frac{18.2075\text{ GB}}{3350\text{ GB/s}} = 5.435\text{ ms} \implies 1471.9\text{ aggregate tok/s}$$
  - Measured TPOT on H100: **$11.5\text{ ms}$** ($695.6\text{ aggregate tok/s}$)
  - **Empirical Memory Bandwidth Utilization**:
    $$\eta_{\text{eff}} = \frac{5.435\text{ ms}}{11.5\text{ ms}} = \mathbf{47.3\%}$$

---

## 4. Key Calibration Findings

1. **Efficiency Gap Attribution**:
   The $\approx 47\% - 50\%$ sustained memory efficiency observed on small-batch single-device GPU inference is attributable to:
   - Kernel launch overheads and kernel transition latencies.
   - Non-GEMV memory traffic (RMSNorm, RoPE, activation buffers, Softmax).
   - DRAM page misses and non-ideal burst utilization in unbatched memory access patterns.

2. **Impact on A48 / A216 Modeling Assumptions**:
   - The default baseline in `configs/systems.json` assumes $\eta = 0.80$ sustained efficiency, which represents an **optimistic high-utilization serving target** (assuming heavily batched or fused kernels).
   - If calibrated against typical un-fused / single-stream GPU measurements, an effective derating factor of $\eta \approx 0.50$ should be evaluated in sensitivity sweeps.

3. **Predictive Accuracy Boundary**:
   - This comparison calibrates our **understanding of real-world efficiency gaps on GPU architectures**.
   - It **does not prove** the performance of A48 (GDDR7) or A216 (HBM4 chiplets), which feature different memory bus widths, memory controllers, and interconnect fabrics.
   - All A48 and A216 numbers remain **analytical feasibility estimates under stated assumptions**, strictly labeled as unvalidated projections per `AGENTS.md`.
