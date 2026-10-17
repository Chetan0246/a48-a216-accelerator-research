# Benchmark Manifest and Datasets

This directory holds empirical measurements from physical hardware benchmarks, used to calibrate and validate the analytical simulator models.

## Provenance and Verification Standards

In accordance with `AGENTS.md`:
1. **Never fabricate benchmark measurements**: Only sourced, reproducible physical measurements may be recorded.
2. **Provenance tracking**: Every entry must capture complete execution metadata including exact hardware details, software/backend versions, execution arguments, source attribution, and direct URLs.
3. **Verification status**:
   - `unverified`: Published or vendor-reported metrics where raw profiling traces and exact environment flags have not been independently reproduced locally.
   - `verified`: Independently audited local benchmark traces with confirmed clock locks, power limits, software environments, and raw log files in `raw/`.
   - Analytical models MUST warn and decline predictive accuracy claims when comparing against `unverified` records.

## Manifest Schema (`benchmark_manifest.csv`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `benchmark_id` | string | Unique identifier (e.g., `pub_h100_001`, `loc_h100_001`) |
| `model_name` | string | HuggingFace or canonical model identifier |
| `format_or_quant` | string | Precision / quantization format (e.g., `FP16`, `FP8`, `AWQ`, `GPTQ`) |
| `context_len` | integer | Input context length in tokens |
| `gen_len` | integer | Generated output tokens |
| `batch_size` | integer | Batch size / concurrency |
| `software_version` | string | Runtime engine and version (e.g., `vLLM v0.6.0`, `TensorRT-LLM v0.11`) |
| `hardware_name` | string | Accelerator marketing name |
| `hardware_details` | string | Form factor, peak bandwidth, TDP, and interconnect details |
| `execution_settings` | string | Exact serving flags (e.g., TP, PP, max model length, KV dtype) |
| `measured_ttft_ms` | float | Measured Time To First Token in milliseconds |
| `measured_tpot_ms` | float | Measured Time Per Output Token (or ITL) in milliseconds |
| `measured_throughput_tps` | float | Measured token throughput in tokens/second |
| `source` | string | Source publication, harness, or test report |
| `source_url` | string | Direct web URL or repository link for provenance audit |
| `benchmark_conditions` | string | Test harness conditions (clocks, power caps, isolated node) |
| `verification_status` | string | `verified` or `unverified` |
| `notes` | string | Additional context or audit observations |

## Directory Structure
- `benchmark_manifest.csv`: Registry of empirical benchmark runs.
- `raw/`: Unprocessed benchmark logs, profiler traces, and timing outputs from physical test harnesses.

