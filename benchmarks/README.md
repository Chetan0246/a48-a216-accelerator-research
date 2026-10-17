# Benchmark Manifest and Datasets

This directory holds empirical measurements from physical hardware benchmarks, used to calibrate and validate the analytical simulator models.

## Structure
- `benchmark_manifest.csv`: Registry of empirical benchmark runs containing tested configurations, operational metrics (TTFT, TPOT, throughput), and system metadata.
- `raw/`: Unprocessed benchmark logs, profiler traces, and timing outputs from physical test harnesses.

## Calibration Usage
Calibration scripts ingest entries from `benchmark_manifest.csv` to compute MFU scaling factors, memory bus efficiency, and kernel overheads across model families and batch regimes.
