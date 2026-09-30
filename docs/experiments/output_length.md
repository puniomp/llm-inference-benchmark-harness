# Output-Length Sensitivity

This document preserves the earlier RTX 4090 output-length experiment and lightweight profiling notes.

## Setup

| Field | Value |
|-------|-------|
| GPU | NVIDIA RTX 4090, 24GB VRAM |
| Runtime | vLLM |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| API | OpenAI-compatible `/v1/completions` |
| Tested concurrency | 24, 32, 40 |
| Requested output lengths | 64, 128, 256, 512 tokens |

## Results at Concurrency 32

| max_tokens | tokens/sec | p95 latency |
|------------|------------|-------------|
| 64 | ~1740 | ~1.17 s |
| 128 | ~1763 | ~2.32 s |
| 256 | ~1768 | ~4.63 s |
| 512 | similar throughput | higher latency |

## Latency vs Output Length

![Latency vs Output Length](../../results/output_length/output_length_p95_latency.png)

## Throughput vs Output Length

![Throughput vs Output Length](../../results/output_length/output_length_throughput.png)

## Measured Observations

Throughput remained relatively stable near saturation across the tested output lengths. Latency increased approximately linearly with requested output length, and increasing concurrency beyond the saturation region did not produce a meaningful throughput gain.

## Interpretation

At the observed saturation boundary, the system maintained high aggregate throughput while longer generations increased tail latency. This is consistent with decode duration contributing heavily to end-to-end response time for long outputs. It is not, by itself, proof that decode was memory-bandwidth-bound.

## GPU Profiling Notes

A lightweight Nsight Systems profile was captured for a representative workload:

| Field | Value |
|-------|-------|
| Concurrency | 32 |
| max_tokens | 512 |

The profiling timeline showed sustained GPU activity, minimal visible idle gaps between kernel executions, and consistent kernel execution patterns during generation. This supports the measured saturation behavior, but raw profiling artifacts are not included in the repository.

Example Perfetto trace:

![Perfetto Trace](../../results/profiling/images/Perfetto_trace_file.png)

## Future Profiling

To make stronger bottleneck claims, collect SM occupancy, memory bandwidth counters, kernel-level timelines, vLLM scheduler queue depth, per-phase prefill/decode timing, and KV-cache utilization.
