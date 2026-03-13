# LLM Inference Benchmark Harness

A lightweight Python harness for benchmarking LLM inference servers under increasing concurrency.

The goal is to measure **throughput scaling and latency behavior (p50 / p95 / p99)** as load increases and identify the **saturation point of an inference system**.

The harness targets **OpenAI-compatible endpoints**, including:

- vLLM  
- Triton Inference Server  
- TensorRT-LLM  
- OpenAI API-compatible gateways

This project focuses on understanding **how GPU inference systems behave under load**, including:

- batching efficiency
- scheduler behavior
- concurrency scaling
- throughput saturation

---

# What This Harness Measures

For each concurrency level the benchmark records:

- number of requests
- elapsed wall time
- requests/sec
- tokens/sec
- latency p50
- latency p95
- latency p99
- mean latency

The harness sweeps across increasing concurrency levels and produces plots showing:

1. **Throughput scaling**
2. **Tail latency growth**

These curves make it easy to identify when the system transitions from efficient utilization to **queueing and saturation**.

---

# Running the Benchmark

## Step 1 — Start an inference server

Example using **vLLM**:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

This exposes an OpenAI-compatible endpoint:

```
http://localhost:8000/v1/completions
```

---

## Step 2 — Run the concurrency sweep

```bash
python bench.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 1,2,4,8,16,24,32,48,64 \
  --max-tokens 256 \
  --requests-per-worker 5 \
  --out results/max_tokens_256.csv
```

This generates:

```
results/max_tokens_256.csv
```

---

## Step 3 — Generate plots

```bash
python plot_results.py
```

This produces:

```
results/throughput_*.png
results/latency_*.png
```

---

# System Configuration

All experiments were run with the following setup:

| Component | Configuration |
|----------|--------------|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| Runtime | vLLM |
| Model | Qwen/Qwen2.5-7B-Instruct |
| API | OpenAI-compatible `/v1/completions` |
| Prompt workload | prompts.json (explanatory prompts) |
| Generation lengths tested | 64, 256, 512 tokens |
| Benchmark driver | custom Python asyncio harness |
| Request scheduling | burst + staggered arrival experiments |

The benchmark focuses on **decoder-heavy inference workloads**, which are typically **memory bandwidth bound during autoregressive generation**.

# expirement 1 - Concurrency scaling

Experiments were run with three generation workloads:

| max_tokens | Workload Type |
|------------|--------------|
| 64 | short responses |
| 256 | typical assistant responses |
| 512 | long generations |

---

## Throughput (max_tokens = 256)

<img src="results/throughput_max_tokens_256.png" width="700">

---

## Latency Percentiles (max_tokens = 256)

<img src="results/latency_max_tokens_256.png" width="700">

---

# Key Observation

Across all workloads the system saturated at approximately:

```
~1800 tokens/sec
```

on an **RTX 4090**.

What I found / observed:

- token throughput remained nearly constant across workloads
- request throughput decreased as generation length increased
- latency scaled roughly linearly with `max_tokens`
- saturation occurred around **~32 concurrent requests**

This confirms that **GPU decoding throughput becomes the primary bottleneck once the model is fully utilized.**

---

## Expirement 2 - Dynamic Batching Behavior

To better understand how inference schedulers handle request arrival patterns, a second expirement compared **burst arrivals** vs **staggered arrivals** near the system saturation point.

The benchmark was run at concurrency levels:
24, 32, 40

with:
max_tokens = 256

### Arrival Patterns Tested

Three arrival patterns were tested:

| Pattern | Description |
|-------|-------------|
| burst | all requests start immediately |
| staggered_25ms | each worker delayed by 25ms |
| staggered_50ms | each worker delayed by 50ms |

---

### Throughput vs Arrival Pattern

<img src="results/batching/throughput_stagger_compare.png" width="700">

---

### Results

| Concurrency | Burst | Staggered 25ms | Staggered 50ms |
|-------------|------|---------------|---------------|
| 24 | ~1308 tokens/s | ~1291 tokens/s | ~1244 tokens/s |
| 32 | **~1819 tokens/s** | ~1663 tokens/s | ~1584 tokens/s |
| 40 | ~1684 tokens/s | ~1600 tokens/s | ~1556 tokens/s |

---

### Interpretation

Peak throughput occurs under **burst arrivals**.

Artificially staggering request arrivals slightly reduces throughput.

This suggests that **vLLM’s continuous batching scheduler already efficiently handles bursty workloads**, forming large batches internally without requiring externally smoothed traffic.

Modern inference engines rely on **request queues and token-level schedulers** to dynamically construct batches during decoding.

### my motivations to build this

LLM inference performance is often reported using **single-request latency**, which hides how systems behave under real load.

This harness focuses on **concurrency-driven saturation testing**, a more realistic way to evaluate:

- GPU utilization
- batching efficiency
- scheduler behavior
- inference server scalability

---

# Future Work

Next steps for this project:

- compare **vLLM vs TensorRT-LLM**
- analyze **GPU utilization during saturation**
- evaluate **multi-GPU inference scaling**
- test **longer generation workloads (1k+ tokens)**

---

# License

MIT License  
Marco Punio — 2026
