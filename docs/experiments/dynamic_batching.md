# Dynamic Batching Behavior

This document preserves the earlier burst vs staggered arrival experiment near the RTX 4090 saturation boundary.

## Setup

| Field | Value |
|-------|-------|
| GPU | NVIDIA RTX 4090, 24GB VRAM |
| Runtime | vLLM |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| API | OpenAI-compatible `/v1/completions` |
| Tested concurrency | 24, 32, 40 |
| Requested output length | 256 tokens |

## Arrival Patterns

| Pattern | Description |
|---------|-------------|
| burst | all requests start simultaneously |
| staggered_25ms | each worker delayed by 25 ms |
| staggered_50ms | each worker delayed by 50 ms |

## Throughput vs Arrival Pattern

![Throughput vs Arrival Pattern](../../results/batching/throughput_stagger_compare.png)

## Results

| Concurrency | Burst | Staggered 25 ms | Staggered 50 ms |
|-------------|-------|-----------------|-----------------|
| 24 | ~1308 tokens/s | ~1291 tokens/s | ~1244 tokens/s |
| 32 | ~1819 tokens/s | ~1663 tokens/s | ~1584 tokens/s |
| 40 | ~1684 tokens/s | ~1600 tokens/s | ~1556 tokens/s |

## Measured Observations

Burst arrivals produced the highest throughput in this run. Staggering requests caused a small but consistent throughput reduction, and the effect was most visible near concurrency 32.

## Interpretation

These results are consistent with vLLM's internal queueing and continuous batching being able to form efficient batches from bursty traffic in this workload. The experiment does not prove that burst traffic is universally optimal; it shows that external request smoothing did not improve throughput for this setup.

Modern inference engines use request queues and token-level schedulers to assemble batches during autoregressive decoding. For this workload, adding artificial inter-arrival delay reduced batching opportunity rather than improving system behavior.
