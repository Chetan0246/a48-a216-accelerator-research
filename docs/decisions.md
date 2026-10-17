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

## ADR-003: GQA and PagedAttention Overhead Inclusion
- **Status**: Accepted
- **Context**: Modern models (Llama-3, Mixtral) use Grouped-Query Attention (GQA), and inference servers use paging systems that incur metadata and block-fragmentation overhead.
- **Decision**: Model `num_kv_heads` independently from `num_attention_heads` and incorporate a default 5% paging memory overhead parameter in `KVCacheModel`.
- **Consequences**: Accurate capacity projection preventing optimistic out-of-memory underestimations.
