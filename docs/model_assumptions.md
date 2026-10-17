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
- **Attention Type Constraints & Hybrid Architectures**:
  - The standard formula is mathematically valid for MHA, GQA, and MQA.
  - **Multi-Head Latent Attention (MLA)**:
    MLA (e.g., DeepSeek-V2/V3) compresses key-value representations into low-rank latent vectors and decouples RoPE keys. Applying standard GQA/MHA formulas to MLA workloads is invalid and strictly deferred via `UnsupportedAttentionTypeError`.
  - **Hybrid Linear Attention & Gated Attention (Qwen3.6-35B-A3B)**:
    Qwen3.6-35B-A3B employs a repeating hybrid pattern: $10 \times \big[3 \times (\text{Gated DeltaNet linear attention}) + 1 \times (\text{Gated Attention})\big]$ across its 40 layers.
    - 30 layers utilize **Gated DeltaNet** linear attention with a fixed-size recurrent state $O(1)$ independent of context length $S$.
    - 10 layers utilize standard **Gated Attention** with KV heads whose cache grows linearly $O(S)$ with context length $S$.
    - Applying a 40-layer standard GQA formula to Qwen is physically incorrect (it overstates context-dependent KV cache size and memory traffic by $4\times$ on the linear layers).
    - Cache capacity and decode bandwidth projections are strictly deferred via `UnsupportedAttentionTypeError` until dual recurrent-state and standard KV-cache modeling is implemented.
    - Parameter-storage calculations ($19.688\text{ GB}$ weights at 4.5 bits) do not depend on attention cache architecture and remain fully supported.

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
- **Enforced Workload Accounting & Reference Values**:
  - **Qwen 3.6 35B A3B**: $P_{\text{total}} = 35.0\text{B}$, $P_{\text{shared}} = 2.8\text{B}$, $P_{\text{routed}} = 32.2\text{B}$, $E = 256$, $k = 8$. Per expert: $32.2 / 256 = 0.12578\text{B}$. Active parameters per token: $2.8 + 8 \times 0.12578 = 3.80625\text{B} \approx 3.81\text{B}$ (hence A3B designation).
  - **Mixtral Parameter Reconciliation**: Previously configured as 12.9B active with 1.5B shared and 45.2B routed across 8 experts ($k=2$). Analytical active count is $1.5 + 2 \times (45.2 / 8) = 1.5 + 11.3 = 12.8\text{B}$. The 12.9B represented an inconsistent rounding discrepancy ($0.1\text{B} > \text{ROUNDING\_TOLERANCE\_B} = 0.05\text{B}$) and has been reconciled to 12.8B.

### Mathematical Formulations:
- **Expected Unique Experts**:
  - Under uniform independent routing of top-$k$ choices out of $E$ experts across batch $B$:
    $$\mathbb{E}[U] = E \times \left[1 - \left(1 - \frac{k}{E}\right)^B\right]$$
  - Under explicit marginal per-token expert inclusion probabilities $\mathbf{q} = [q_1, \dots, q_E]$, where $q_i \in [0.0, 1.0]$ represents the marginal probability that one token selects expert $i$, subject to $\sum_{i=1}^E q_i = k$:
    $$\mathbb{E}[U] = \sum_{i=1}^E \left[1 - (1 - q_i)^B\right]$$
  - **Routing Skew Semantics & Jensen's Inequality**:
    The function $f(q) = 1 - (1 - q)^B$ is strictly concave for $B > 1$ ($f''(q) = -B(B-1)(1-q)^{B-2} < 0$ on $q \in (0, 1)$). By Jensen's inequality:
    $$\frac{1}{E}\sum_{i=1}^E \left[1 - (1 - q_i)^B\right] \le 1 - \left(1 - \frac{\sum q_i}{E}\right)^B = 1 - \left(1 - \frac{k}{E}\right)^B$$
    $$\implies \mathbb{E}[U_{\text{skewed}}] \le \mathbb{E}[U_{\text{uniform}}]$$
    Token concentration onto popular experts mathematically **decreases** or maintains the expected number of distinct activated experts across the batch, reducing total distinct expert weight traffic while concentrating compute load on popular bottleneck experts.
- **Resident vs. Non-Resident Partitioning**:
  - Uniform routing:
    $$\mathbb{E}[U_{\text{res}}] = E_{\text{res}} \times \left[1 - \left(1 - \frac{k}{E}\right)^B\right]$$
    $$\mathbb{E}[U_{\text{nonres}}] = (E - E_{\text{res}}) \times \left[1 - \left(1 - \frac{k}{E}\right)^B\right]$$
  - Nonuniform routing with $\mathbf{q}$:
    $$\mathbb{E}[U_{\text{res}}] = \sum_{i \in \text{resident}} \left[1 - (1 - q_i)^B\right]$$
    $$\mathbb{E}[U_{\text{nonres}}] = \sum_{i \in \text{nonresident}} \left[1 - (1 - q_i)^B\right]$$
    where $\mathbb{E}[U_{\text{res}}] + \mathbb{E}[U_{\text{nonres}}] = \mathbb{E}[U_{\text{total}}]$.
- **Host-Offload Execution Bounds**:
  - **Serialized Bound** (conservative / non-overlapped): The stage execution times add sequentially per decoding step:
    $$t_{\text{step}} = t_{\text{card}} + t_{\text{host}}$$
    $$T_{\text{serialized}} = \frac{B}{t_{\text{card}} + t_{\text{host}}}$$
  - **Overlapped Bound** (optimistic / ideal pipelining upper bound): On-card memory retrieval and host PCIe transfers overlap perfectly; throughput is dictated solely by the bottleneck stage:
    $$T_{\text{overlapped}} = \min(T_{\text{card}}, T_{\text{host}})$$
  - Clearly, $T_{\text{serialized}} < T_{\text{overlapped}}$ for any non-zero offloaded traffic. Both bounds are reported explicitly in the simulator.

---

## 6. Calibration & Provenance Standards
- **Unverified vs. Verified Records**:
  All entries in `benchmark_manifest.csv` must carry an explicit `verification_status`:
  - `unverified`: Literature or vendor-reported metrics where raw profiling traces and exact environment flags have not been independently reproduced locally.
  - `verified`: Independently audited local benchmark traces with locked clocks, confirmed power limits, and bit-level reproducible environments.
- **Empty Dataset Protection**:
  An empty comparison dataset cannot produce 0% MAPE. Evaluating MAPE against an empty dataset raises a clear `ValueError`.
- **Algebraic Sanity Checks vs. Empirical Validation**:
  Comparisons against published literature baselines (e.g. H100 SXM5 80GB) provide an **algebraic sanity check** confirming equation consistency against realistic hardware efficiency gaps ($\eta_{\text{eff}} \approx 47\% - 50\%$). However, such checks do not constitute empirical validation of predictive accuracy for proposed unbuilt accelerators (A48 Base and A216 Advanced), whose figures remain analytical feasibility projections per `AGENTS.md`.
