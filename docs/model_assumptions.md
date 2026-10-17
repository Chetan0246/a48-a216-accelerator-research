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
    - **Standard Gated Attention (10 layers)**:
      - 2 KV heads, head dimension 256, FP16 precision ($P_{\text{attn}} = 16.0$ bits).
      - Bytes per token across 10 layers: $2 \times 10 \times 2 \times 256 \times (16/8) = 20,480\text{ bytes/token}$.
      - For context length $S = 8,192$: $8,192 \times 20,480 = 167,772,160\text{ bytes} = 0.16777\text{ decimal GB} \approx \mathbf{0.168\text{ GB}}$ per sequence.
      - **Context scaling**: Grows strictly linearly $O(S)$ with context length $S$.
      - **Decode memory traffic**: In each decode step, reading accumulated context history requires $B \times \text{KV}_{\text{seq}}(S)$ bytes read traffic ($O(S)$), plus $B \times 20,480$ bytes write traffic to append the new token's KV.
    - **Gated DeltaNet Linear Attention (30 layers)**:
      - **Recurrent State Matrix**: Each linear layer maintains a persistent recurrent state matrix $S_h \in \mathbb{R}^{d_k \times d_v}$ per value head, where $d_k = 128$, $d_v = 128$, and $H_v = 32$ value heads.
      - State elements per layer: $32 \times 128 \times 128 = 524,288$ elements.
      - Total recurrent elements across 30 layers: $30 \times 524,288 = 15,728,640$ elements per sequence.
      - State precision: FP32 ($P_{\text{state}} = 32.0$ bits = 4 bytes/elem, matching published `mamba_ssm_dtype: float32` required for numerical stability in the delta rule update).
      - Recurrent state capacity per sequence: $15,728,640 \times 4 = 62,914,560\text{ bytes} \approx \mathbf{0.0629\text{ decimal GB}}$.
      - Configurable precision: FP16 ($16.0$ bits) yields $31,457,280\text{ bytes} \approx 0.0315\text{ decimal GB}$.
      - **Causal 1D Convolution State**: Short depthwise causal convolution (kernel size $K = 4$, holding $K-1 = 3$ history tokens) across $\text{conv\_dim} = 2 \times (16 \times 128) + (32 \times 128) = 8,192$ channels.
      - Conv state elements across 30 layers: $30 \times 3 \times 8,192 = 737,280$ elements per sequence.
      - At baseline FP16 precision ($P_{\text{conv}} = 16.0$ bits = 2 bytes/elem): $737,280 \times 2 = 1,474,560\text{ bytes} \approx \mathbf{0.00147\text{ decimal GB}}$.
      - **Context scaling**: Both recurrent state and conv state are strictly independent of context length $S$ ($O(1)$ w.r.t. $S$), but scale linearly with batch size $B$.
      - **Decode memory traffic**: Each decode step updates the recurrent state ($S_t = S_{t-1}(I - \beta_t k_t k_t^T) + v_t k_t^T$), requiring **1 full read** of $S_{t-1}$ and **1 full write** of $S_t$ ($2 \times B \times \text{recurrent\_state\_bytes} = 2 \times B \times 0.0629\text{ GB}$). Similarly, updating the 1D convolution rolling buffer requires reading the past 3 tokens and writing the shifted buffer ($2 \times B \times \text{conv\_state\_bytes} = 2 \times B \times 0.00147\text{ GB}$).
    - **Total Persistent Inference State**:
      At $S = 8,192$, FP16 attention, FP32 recurrent state, FP16 conv state (batch $B = 1$):
      $$\text{State}_{\text{total}} = 0.16777 + 0.06291 + 0.00147 = \mathbf{0.23216\text{ decimal GB}}$$
    - **Published vs. Simulator Assumptions**:
      - *Published configuration parameters*: 10 full attention layers, 30 linear attention layers, 2 KV heads (dim 256), 16 linear key heads (dim 128), 32 linear value heads (dim 128), conv kernel size 4, `mamba_ssm_dtype: float32`, conv channels 8,192.
      - *Simulator assumptions*: Conv state precision baseline FP16 (configurable), allocation overhead baseline 0.0%, 1 read + 1 write per decode step for state matrices and conv buffers.

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
  - **Qwen 3.6 35B A3B Parameter Accounting Reconciliation**:
    - **Total parameters**: $35.0\text{B}$.
    - **Routed expert parameters**: 40 layers $\times 256$ experts $\times (3 \times 2048 \times 512) = 32,212,254,720$ parameters ($\mathbf{32.21\text{B}} \approx 32.2\text{B}$).
    - **Per-expert size**: $3 \times 2048 \times 512 = 3,145,728$ parameters ($0.00315\text{B}$).
    - **Active routed parameters per token** (top-8): $40 \times 8 \times 3,145,728 = 1,006,632,960$ parameters ($\mathbf{1.01\text{B}}$).
    - **Shared / non-routed parameter pool**: Total weights stored on device must house all $35.0\text{B}$ parameters. The non-routed pool is $35.0 - 32.21 = \mathbf{2.79\text{B}} \approx \mathbf{2.8\text{B}}$, composed of:
      1. Text backbone shared layers (10 full attention + 30 linear attention + 40 shared experts + routers): $\mathbf{1.43\text{B}}$.
      2. Input embedding table ($248,320 \times 2048$): $\mathbf{0.51\text{B}}$.
      3. Output LM head (`tie_word_embeddings: false`, $248,320 \times 2048$): $\mathbf{0.51\text{B}}$.
      4. Vision encoder tower (depth 27, hidden size 1152, intermediate 4304) included in multimodal conditional generation weights: $\mathbf{\approx 0.41\text{B}}$.
    - **Reconciling Conventions**:
      - **Simulator on-card active parameters**: In the simulator's strict parameter accounting, all resident non-routed weights ($2.8\text{B}$) remain in memory and are accounted as shared baseline parameters, yielding $P_{\text{active}} = 2.8 + 8 \times (32.2 / 256) = \mathbf{3.81\text{B}}$.
      - **Published Model Card ("A3B") Designation**: In open-weights model card conventions, active parameter counts reflect pure text forward-pass activation with tied/single embedding space:
        $$\text{Backbone Shared } (1.43\text{B}) + \text{Active Routed } (1.01\text{B}) + \text{Token Embedding } (0.51\text{B}) = \mathbf{2.94\text{B}} \approx \mathbf{3.0\text{B}}$$
      - Both conventions are explicitly documented. The simulator retains $3.81\text{B}$ for on-card weight memory tracking and conservative prefill FLOP bounding to preserve physical parameter conservation ($2.8\text{B} + 32.2\text{B} = 35.0\text{B}$).
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
