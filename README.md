# A48 / A216 Accelerator Research Toolkit

An analytical simulation and evaluation framework designed for accelerator research, specifically tailored for evaluating dense and Mixture-of-Experts (MoE) large language model workloads on target architectures such as **A48** and **A216**.

## Overview

This repository provides tools for:
- **Memory Capacity Analysis**: Evaluating model weights, KV cache sizing, activation overheads, and tensor/expert parallel partitioning across devices.
- **Roofline & Performance Modeling**: Analytical prefill (TTFT) and decode (TPOT) performance modeling based on operational intensity and hardware characteristics.
- **Mixture-of-Experts (MoE) Modeling**: Estimating routing latency, active vs. inactive parameters, and all-to-all communication bottlenecks.
- **Empirical Calibration**: Aligning analytical models with real-world empirical benchmarks.
- **Scenario Evaluation**: Comparing workloads across system configurations with batch simulation runners.

## Directory Structure

```
a48-a216-accelerator-research/
├── AGENTS.md                  # Guidelines and instructions for autonomous agents
├── README.md                  # Project overview and documentation
├── pyproject.toml             # Project packaging and dependency specs
├── .gitignore                 # Git ignore file
├── configs/                   # System and workload specifications
│   ├── systems.json           # Hardware specs (A48, A216, baselines)
│   └── workloads.json         # Workload configs (Llama, DeepSeek, Mixtral)
├── src/accelerator_sim/       # Core simulator package
│   ├── __init__.py
│   ├── capacity.py            # Memory capacity & allocation modeling
│   ├── kv_cache.py            # KV cache footprint calculations
│   ├── decode.py              # Decode phase latency & throughput simulator
│   ├── prefill.py             # Prefill phase latency & throughput simulator
│   ├── moe.py                 # MoE routing & communication modeling
│   ├── roofline.py            # Roofline model implementation
│   ├── calibration.py         # Empirical calibration utilities
│   └── cli.py                 # Command line interface
├── tests/                     # Unit test suites
├── benchmarks/                # Empirical benchmark datasets & manifests
│   ├── README.md
│   ├── benchmark_manifest.csv
│   └── raw/
├── docs/                      # Technical documentation & ADRs
│   ├── architecture.md
│   ├── model_assumptions.md
│   ├── benchmark_protocol.md
│   └── decisions.md
├── scripts/                   # Evaluation and scenario run scripts
│   └── run_scenarios.py
└── results/                   # Simulation output directories
    ├── capacity/
    ├── decode/
    ├── prefill/
    └── calibration/
```

## Quickstart

### Installation

```bash
pip install -e .
```

For development dependencies:
```bash
pip install -e ".[dev]"
```

### Running Simulations via CLI

```bash
accelerator-sim --system A48 --workload llama-3-8b --batch-size 8
```

### Running Test Suite

```bash
pytest tests/
```
