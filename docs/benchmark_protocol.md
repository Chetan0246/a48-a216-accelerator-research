# Benchmark Protocol

This document defines the standardized measurement protocol for collecting hardware benchmark data to calibrate the simulation framework.

## 1. Test Environment Setup
- **Hardware Isolation**: Lock GPU/NPU clocks to base frequencies to minimize thermal throttling jitter.
- **Warmup Cycles**: Run at least 10 warmup requests prior to recording timing numbers.
- **Batch Consistency**: Ensure fixed batch sizes without dynamic batch padding skew during microbenchmark evaluation.

## 2. Metrics to Measure
1. **Time to First Token (TTFT)**:
   - Measured from HTTP request receipt / input token arrival to emission of the first output token.
   - Measures prompt prefill efficiency.
2. **Time Per Output Token (TPOT)**:
   - Also known as Inter-Token Latency (ITL).
   - Average latency between consecutive generation tokens.
3. **End-to-End Latency**:
   - Total time elapsed from start of request to generation completion.
4. **Token Generation Throughput**:
   - Total generated output tokens divided by total generation duration (tokens/sec).

## 3. Data Recording Procedure
- Place raw command outputs, profiler exports, and JSON logs in `benchmarks/raw/<benchmark_id>/`.
- Append summary rows to `benchmarks/benchmark_manifest.csv` with standardized columns:
  - `benchmark_id`: Unique identifier (e.g., `bm_001`)
  - `system_name`: Must match an entry in `configs/systems.json`
  - `model_name`: Must match an entry in `configs/workloads.json`
  - `context_len`, `gen_len`, `batch_size`: Benchmark workload parameters
  - `measured_ttft_ms`, `measured_tpot_ms`, `measured_throughput`: Empirical outputs
