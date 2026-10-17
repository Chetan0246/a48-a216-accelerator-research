# Model Assumptions and Analytical Formulations

This document formalizes the mathematical equations, hardware simplifications, and analytical assumptions incorporated in the simulation models for Workstreams A and B.

---

## 1. Capacity & Memory Reserve
- **Physical Memory**: Total installed device capacity in decimal GB ($10^9$ bytes).
- **Runtime Reserve**: Configurable reserve for driver, OS buffers, and framework context.
  - Requirement: $\text{runtime\_reserve\_gb} < \text{physical\_memory\_gb}$.
  - Any configuration where $\text{reserve} \ge \text{physical}$ is strictly invalid and rejected.
- **Usable Memory**:
  $$\text{usable\_memory\_gb} = \text{physical\_memory\_gb} - \text{runtime\_reserve\_gb}$$
- **Model Footprint & Headroom**:
  $$\text{model\_footprint\_gb} = \text{weights\_gb} + \text{kv\_cache\_total\_gb}$$
  $$\text{headroom\_gb} = \text{usable\_memory\_gb} - \text{model\_footprint\_gb}$$
  Fit condition: $\text{headroom\_gb} \ge 0$.

---

## 2. KV Cache Sizing and Attention Mechanisms
- **Standard K/V Cache Formula**:
  $$\text{bytes\_per\_token} = 2 \times L \times H_{\text{kv}} \times D \times \left(\frac{P_{\text{bits}}}{8}\right) \times \left(1 + \frac{\text{overhead\_percent}}{100}\right)$$
  where $L$ is layer count, $H_{\text{kv}}$ is KV heads count, $D$ is head dimension, and $P_{\text{bits}}$ is precision (16.0 for FP16/BF16, 8.0 for FP8).
- **Overhead Baseline vs Paging Overhead**:
  - Pure analytical baseline: $\text{overhead\_percent} = 0.0\%$ default.
  - PagedAttention allocation: Configurable $\text{overhead\_percent} \approx 5.0\%$ to model block allocation table overhead and internal fragmentation.
- **Attention Type Constraints**:
  - The standard formula is mathematically valid for MHA, GQA, and MQA.
  - Multi-Head Latent Attention (MLA, e.g., DeepSeek-V2/V3) compresses key-value representations into low-rank latent vectors and decouples RoPE keys. Applying standard GQA/MHA formulas to MLA workloads is invalid and strictly deferred via `UnsupportedAttentionTypeError`.

---

## 3. Prefill Phase Compute (TTFT)
- First-order dense-equivalent FLOPs for prompt processing:
  $$F_{\text{TFLOP}} = \frac{2 \times P_{\text{B}} \times S \times B}{1000}$$
  where $P_{\text{B}}$ is parameter count in billions, $S$ is prompt tokens per sequence, and $B$ is batch size.
- Required effective compute throughput for a target time-to-first-token (TTFT) budget $T_{\text{budget}}$:
  $$R_{\text{TFLOP/s}} = \frac{F_{\text{TFLOP}}}{T_{\text{budget}}}$$
- *Omitted in first-order model*: Quadratic attention FLOPs ($4LS^2D$), logits projection overhead, and padding tokens.

---

## 4. Decode Phase Memory Bandwidth Ceilings (TPOT)
- **Weight Streaming**: Shared model weights are read once per batch step across all concurrent sequences:
  $$W_{\text{traffic}} = \frac{P_{\text{B}} \times \text{bits\_per\_param}}{8}$$
- **KV Cache Retrieval**: Each sequence in the batch reads its accumulated context:
  $$\text{KV}_{\text{batch}} = B \times \text{KV}_{\text{per\_sequence}}$$
- **Aggregate Decode Throughput Ceiling**:
  $$T_{\text{aggregate}} = \frac{B \cdot \text{BW}_{\text{sustained}}}{W_{\text{traffic}} + B \cdot \text{KV}_{\text{per\_sequence}}}$$
  where $\text{BW}_{\text{sustained}} = \text{BW}_{\text{peak}} \times 1000 \times \eta_{\text{bw}}$ in GB/s.
- **Per-Sequence Throughput**: $T_{\text{per\_seq}} = T_{\text{aggregate}} / B$.

---

## 5. Mixture-of-Experts (MoE) Modeling Boundaries

> [!CAUTION]
> **Analytical First-Order Bound Notice**:
> The MoE offload model provides an **analytical first-order throughput ceiling**, NOT a validated end-to-end performance predictor. Real systems exhibit routing skew, expert caching/pinning, PCIe DMA framing, and asynchronous compute/transfer pipelining.

### Parameter Accounting:
- Total parameters are explicitly split into shared and routed components:
  $$P_{\text{total}} = P_{\text{shared}} + P_{\text{routed}}$$
  where $P_{\text{shared}}$ includes embeddings, self-attention, layer norms, and router gating matrices.
- Individual expert size: $P_{\text{expert}} = P_{\text{routed}} / E$.
- Active parameters per token: $P_{\text{active}} = P_{\text{shared}} + k \times P_{\text{expert}}$.

### Mathematical Formulations:
- **Expected Unique Experts**: Under uniform independent routing of top-$k$ distinct choices out of $E$ experts across $B$ tokens:
  $$\mathbb{E}[U] = E \times \left[1 - \left(1 - \frac{k}{E}\right)^B\right]$$
- **Resident vs. Non-Resident Partitioning**:
  $$\mathbb{E}[U_{\text{res}}] = E_{\text{res}} \times \left[1 - \left(1 - \frac{k}{E}\right)^B\right]$$
  $$\mathbb{E}[U_{\text{nonres}}] = (E - E_{\text{res}}) \times \left[1 - \left(1 - \frac{k}{E}\right)^B\right]$$
- **Separate Unoverlapped Ceilings**:
  The combined throughput is evaluated as the serialized minimum of card memory bandwidth and host link throughput:
  $$T_{\text{combined}} = \min(T_{\text{card}}, T_{\text{host}})$$
- **Routing Skew Sensitivity**:
  An optional multiplier $\sigma \ge 1.0$ (`routing_skew_factor`) is supported to model token concentration on popular experts, while recognizing that production-grade prediction requires trace-driven measurements.

---

## 6. Calibration & Provenance Standards
- **Unverified vs. Verified Records**:
  All entries in `benchmark_manifest.csv` must carry an explicit `verification_status`:
  - `unverified`: Literature or vendor-reported metrics where raw profiling traces and exact environment flags have not been independently reproduced locally.
  - `verified`: Independently audited local benchmark traces with locked clocks, confirmed power limits, and bit-level reproducible environments.
- **Empty Dataset Protection**:
  An empty comparison dataset cannot produce 0% MAPE. Evaluating MAPE against an empty dataset raises a clear `ValueError`.
- **Controlled Baselines**:
  Simulations must be calibrated against controlled physical GPU baselines (e.g. H100 SXM 80GB) to establish realistic sustained memory efficiency ($\eta_{\text{eff}} \approx 47\% - 50\%$), while acknowledging that calibrating against a GPU does not prove the performance of proposed unbuilt accelerators.
