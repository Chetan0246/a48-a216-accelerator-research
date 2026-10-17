# Architecture Decision Records (ADR)

## ADR-001: Analytical Simulation vs Cycle-Accurate Emulation
- **Status**: Accepted
- **Context**: For early-stage accelerator architecture research (comparing A48 and A216 against existing baselines like H100 and A100), full cycle-accurate gem5/timeloop models have excessive simulation runtime and parameter configuration friction.
- **Decision**: Adopt a modular analytical roofline model combined with phase-specific memory access accounting (prefill, decode, KV cache, and MoE all-to-all).
- **Consequences**: Enables rapid sweeps across dozens of batch sizes, sequence lengths, and precision settings while calibrating against real hardware traces.

## ADR-002: Modular Separation of Prefill and Decode
- **Status**: Accepted
- **Context**: LLM inference exhibits bimodal operational characteristics: prefill is compute-heavy and parallelized, whereas decode is sequential and memory-bandwidth bound.
- **Decision**: Implement separate simulation modules (`prefill.py` and `decode.py`) with shared hardware rooflines and KV cache specifications.
- **Consequences**: Clearer attribution of latency bottlenecks and clean mapping to serving metrics (TTFT vs TPOT).

## ADR-003: Attention Mechanisms and KV-Cache Overhead Accounting
- **Status**: Accepted
- **Context**: Modern architectures employ diverse attention strategies (GQA, MQA, MLA) and serving runtimes use paging algorithms (e.g., PagedAttention) that incur block table and internal fragmentation overhead.
- **Decision**:
  1. Distinguish pure analytical tensor footprint from runtime paging structures.
  2. The default mathematical baseline for `KVCacheShape` is set to `overhead_percent = 0.0%` (unpadded tensor size).
  3. A configurable `overhead_percent` parameter (with 5.0% recommended for PagedAttention allocation buffers) is available for serving-system capacity projections.
  4. Standard K/V cache formulas are restricted to MHA, GQA, and MQA. Compressed latent architectures such as MLA (e.g. DeepSeek-V2) must be explicitly flagged and deferred to avoid invalid capacity projections.
- **Consequences**: Eliminates silent discrepancy between analytical reference calculations (0% overhead) and systems projections, and protects against applying standard formulas to latent attention mechanisms.

