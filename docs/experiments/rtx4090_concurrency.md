# RTX 4090 Concurrency Scaling

This document preserves the original local RTX 4090 concurrency experiment. The newer H100 streaming benchmark in the repository README is the primary current result because it measures TTFT, ITL, and per-user generation speed using streamed responses.

## System Configuration

| Component | Configuration |
|----------|--------------|
| GPU | NVIDIA RTX 4090, 24GB VRAM |
| Runtime | vLLM |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| API | OpenAI-compatible `/v1/completions` |
| Prompt workload | `prompts.json`, explanatory prompts |
| Generation lengths tested | 64, 256, 512 tokens |
| Benchmark driver | custom Python asyncio harness |

The workload emphasized autoregressive generation. The results are consistent with decode-heavy serving behavior, but this experiment did not collect hardware counters and should not be read as proof of a specific GPU bottleneck.

## Running the Legacy Benchmark

Start a local vLLM server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

Run a concurrency sweep:

```bash
python bench.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 1,2,4,8,16,24,32,48,64 \
  --max-tokens 256 \
  --requests-per-worker 5 \
  --out results/max_tokens_256.csv
```

Generate legacy plots:

```bash
python plot_results.py
```

## Workloads

| max_tokens | Workload type |
|------------|---------------|
| 64 | short responses |
| 256 | typical assistant responses |
| 512 | long generations |

## Results

### Throughput, max_tokens=256

![Throughput](../../results/concurrency/throughput_max_tokens_256.png)

### Latency Percentiles, max_tokens=256

![Latency Percentiles vs Concurrency](../../results/concurrency/latency_max_tokens_256.png)

## Measured Observations

Across the tested workloads, aggregate token throughput on the RTX 4090 plateaued around 1800 tokens/sec. Request throughput decreased as generation length increased, latency increased with longer generations, and saturation appeared around 32 concurrent requests in this setup.

## Interpretation

The observed curves show a transition from underutilized serving to a throughput plateau where adding more concurrent requests mostly increases waiting and completion time. This is consistent with an inference server reaching a saturation region, but the experiment did not include enough hardware-level profiling to identify the exact limiting resource.

## Practical Implication

Inference systems should be sized around expected output length distributions and latency SLOs, not only around peak aggregate tokens/sec.
