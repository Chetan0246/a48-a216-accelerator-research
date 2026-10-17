# Model Assumptions and Analytical Formulations

This document formalizes the mathematical equations, hardware simplifications, and analytical assumptions incorporated in the simulation models.

## 1. Roofline Model Formulations

The ridge point $\text{RP}$ (in FLOPs/Byte) determines whether a kernel is compute-bound or memory-bound:

$$\text{RP} = \frac{\text{Peak TFLOPS} \times 10^{12}}{\text{Memory Bandwidth (GB/s)} \times 10^9} = \frac{\text{Peak TFLOPS}}{\text{Memory Bandwidth (GB/s)}} \times 1000$$

For operational intensity $I = \frac{\text{Total FLOPs}}{\text{Total Bytes}}$:
- If $I < \text{RP}$, the operation is **memory bandwidth bound**.
- If $I \ge \text{RP}$, the operation is **compute bound**.

## 2. Prefill Phase (TTFT)

During the prompt processing phase:
- GEMM compute FLOPs:
  $$\text{FLOPs}_{\text{GEMM}} \approx 2 \times P_{\text{active}} \times B \times S_{\text{prompt}}$$
- Attention compute FLOPs:
  $$\text{FLOPs}_{\text{Attn}} \approx 4 \times L \times B \times S_{\text{prompt}}^2 \times D_{\text{hidden}}$$
- Weights are read once from HBM into on-chip SRAM/registers.
- KV Cache is written to HBM for each prompt token:
  $$\text{Bytes}_{\text{KV\_write}} = 2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times P_{\text{bytes}} \times B \times S_{\text{prompt}}$$

## 3. Decode Phase (TPOT)

For step $t$ in autoregressive generation:
- Each token step requires reading model weights:
  $$\text{Bytes}_{\text{weights}} = P_{\text{active}} \times P_{\text{bytes}}$$
- KV cache must be read for all preceding $S_t$ tokens:
  $$\text{Bytes}_{\text{KV\_read}} = B \times S_t \times \text{BytesPerToken}$$
- Compute per step:
  $$\text{FLOPs}_{\text{step}} \approx 2 \times P_{\text{active}} \times B$$
- For batch sizes $B < 16$, decode is strictly memory bandwidth bound; as batch size increases, operational intensity scales linearly with $B$ until reaching the ridge point.

## 4. Mixture of Experts (MoE)

- In MoE layers, only $K$ of $N_{\text{experts}}$ are activated per token.
- Communication overhead for All-to-All collective across $M$ devices:
  $$\text{Bytes}_{\text{All2All}} = 2 \times \frac{M - 1}{M} \times B \times S \times K \times D_{\text{hidden}} \times P_{\text{bytes}}$$
- Load Imbalance Multiplier: Real-world routing exhibits token distribution variance among experts, parameterized by default with factor $1.15\times$.
