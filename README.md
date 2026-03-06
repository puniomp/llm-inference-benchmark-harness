# LLM Inference Benchmark Harness

A lightweight Python harness for benchmarking LLM inference servers under increasing concurrency.

The goal is to measure **throughput scaling and latency behavior (p50 / p95 / p99)** as load increases and identify the **saturation point of an inference system**.

The harness targets **OpenAI-compatible endpoints**, including:

- vLLM  
- Triton Inference Server  
- TensorRT-LLM  
- OpenAI API-compatible gateways  

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

# Example Benchmark Results

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

# My motivations to build this

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
- measure **dynamic batching effects**
- analyze **GPU utilization during saturation**

---

# License

MIT License  
Marco Punio — 2026
