# A48/A216 Accelerator Research Simulator

## Project objective

Build a reproducible Python research simulator for evaluating the feasibility of two proposed AI inference accelerators:

- A48 Base: cost-conscious local AI inference accelerator.
- A216 Advanced: premium, high-bandwidth AI inference accelerator.

The current scope is limited to Workstreams A and B:

1. Define representative workloads and acceptance criteria.
2. Model memory capacity and KV-cache requirements.
3. Estimate decode throughput ceilings.
4. Estimate prefill compute requirements.
5. Model MoE expert overlap and host-memory offloading.
6. Calibrate predictions against sourced, reproducible benchmarks.

## Explicit exclusions

Do not implement:

- RTL or hardware description languages.
- Chip floorplans or physical-design flows.
- FPGA implementation.
- Tape-out plans.
- GUI or web dashboard.
- A custom compiler or production inference runtime.
- Unverified hardware specifications presented as facts.
- Benchmark results invented to fill missing data.

Physical design, manufacturing cost, and tape-out remain outside the current scope.

## Engineering principles

1. Separate physical memory capacity from usable memory capacity.
2. Use decimal GB for memory-device capacities unless explicitly stated otherwise.
3. Use GiB and MiB where binary units are intended.
4. Keep units explicit in variable names and documentation.
5. Distinguish theoretical bandwidth from effective bandwidth.
6. Distinguish theoretical compute throughput from effective compute throughput.
7. Treat all proposed hardware performance figures as assumptions until validated.
8. Record the source, configuration, date, and limitations of every external benchmark.
9. Keep hardware parameters configurable.
10. Never silently change established assumptions or reference calculations.

## Modeling requirements

### Capacity

Account for model weights, quantization metadata, KV cache, runtime buffers, and configurable memory reserves.

### KV cache

Support configurable layer count, KV-head count, head dimension, context length, batch size, and KV precision.

Document assumptions about attention architecture and cache representation.

### Decode

Provide memory-bandwidth ceilings and compute ceilings separately.

State assumptions about weight reuse, batching, KV traffic, and effective bandwidth.

### Prefill

Estimate first-order model FLOPs and required effective compute throughput for a specified prompt length and latency target.

Clearly identify omitted operations and approximations.

### MoE

Model expected distinct experts activated across a batch.

Do not assume expert overlap is constant or that all experts are independently read for every token.

Keep expert weight residency and host-memory offloading assumptions explicit.

### Calibration

Never fabricate benchmark measurements.

Store measured values separately from simulated results.

Compare equivalent model architectures, quantization formats, context lengths, batch sizes, and execution configurations wherever possible.

## Testing requirements

Every modeling module must have automated tests.

Tests must cover known reference calculations, invalid inputs, boundary conditions, and unit consistency.

Before considering a change complete, run:

- pytest
- Python compilation checks
- Relevant command-line smoke tests

Do not suppress failures to make the test suite pass.

## Agent workflow

1. Inspect existing files before modifying them.
2. Explain the proposed changes before implementation.
3. Keep changes small and reviewable.
4. Do not overwrite user work without review.
5. Run tests after changes.
6. Summarize changed files, test results, assumptions, and remaining limitations.
7. Ask for approval before expanding project scope.

The user makes architecture decisions. The agent supplies calculations, evidence, tests, and clearly labeled recommendations.
