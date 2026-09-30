# LLM Inference Benchmark Harness
> Investigating how workload shape, batching behavior, and concurrency impact LLM inference throughput and tail latency under GPU saturation.

A Python benchmarking and profiling harness for evaluating LLM inference systems under concurrency-driven load. The experiments focus on production-relevant inference characteristics including queueing behavior, decode-phase saturation, batching efficiency, and workload-sensitive latency growth.

The goal is to measure **throughput scaling and latency behavior (p50 / p95 / p99)** as load increases and identify the **saturation point of an inference system**.

The harness targets **OpenAI-compatible endpoints**, including:

- vLLM  
- Triton Inference Server  
- TensorRT-LLM  
- OpenAI API-compatible gateways  

This project studies how GPU-accelerated LLM inference systems behave under realistic, concurrency-driven load, with emphasis on:

- batching efficiency  
- scheduler behavior  
- concurrency scaling  
- throughput saturation  
- **workload-dependent latency (output length sensitivity)**  

---

# What This Harness Measures

In particular, this work demonstrates that **throughput alone is an incomplete metric for evaluating LLM inference systems under load**.

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

## Streaming Interactivity Benchmark

For TTFT, ITL, per-user generation speed, and throughput/interactivity analysis,
use `streaming_perf_bench.py`. This script uses streaming responses so it can
measure the time from request submission to the first generated token and the
generation interval after the first token arrives.

### Metric Formulas

For each request:

- Request latency = request completion timestamp - request submission timestamp
- TTFT = first non-empty streamed delta timestamp - request submission timestamp
- Decode duration = last streamed token timestamp - first streamed token timestamp
- ITL = decode duration / (`output_tokens - 1`)
- TPOT = (request latency - TTFT) / `output_tokens`
- Output tokens/sec/user = (`output_tokens - 1`) / decode duration

For each benchmark run:

- Aggregate output tokens/sec = total generated output tokens / benchmark
  wall-clock duration
- Requests/sec = completed requests / benchmark wall-clock duration
- Input tokens, output tokens, and total tokens are recorded separately where
  tokenization is available

Output token counts use the vLLM `/tokenize` endpoint when available. If that
endpoint is unavailable, the script falls back to counting streamed text events
and records the stream event count so the limitation is visible. Warmup runs
are intentionally excluded from result files.

### Start vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

### Warm Up

```bash
python streaming_perf_bench.py \
  --experiment warmup \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-type completions \
  --input-tokens 128 \
  --max-tokens 128 \
  --requests-per-worker 2
```

### Experiment 1: Concurrency Sweep

```bash
python streaming_perf_bench.py \
  --experiment concurrency \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-type completions \
  --concurrency 1,8,32,64,128,256 \
  --input-tokens 128 \
  --max-tokens 128 \
  --requests-per-worker 2 \
  --out-dir results/streaming/concurrency \
  --yes
```

This writes:

```text
results/streaming/concurrency/concurrency_raw.csv
results/streaming/concurrency/concurrency_summary.csv
results/streaming/concurrency/concurrency_metadata.json
```

### Experiment 2: Workload Shape

```bash
python streaming_perf_bench.py \
  --experiment workload-shape \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-type completions \
  --workload-concurrency 16 \
  --requests-per-worker 2 \
  --out-dir results/streaming/workload_shape \
  --yes
```

The workload-shape benchmark runs:

| Profile | Input tokens | Requested output tokens | Purpose |
|---------|--------------|-------------------------|---------|
| baseline_128in_128out | 128 | 128 | Balanced baseline |
| prefill_4096in_128out | ~4096 | 128 | Prefill-heavy workload |
| decode_128in_2048out | 128 | ~2048 | Decode-heavy workload |

The benchmark records observations only. Do not assume a workload is
compute-bound or memory-bound solely from throughput curves; treat such claims
as hypotheses that require additional profiling.

### Generate Plots and Report

```bash
python plot_streaming_results.py \
  --summary-csv results/streaming/concurrency/concurrency_summary.csv \
  --extra-summary-csv results/streaming/workload_shape/workload-shape_summary.csv \
  --out-dir results/streaming/plots \
  --report results/streaming/report.md
```

The generated plots include aggregate output tokens/sec vs concurrency,
per-user output tokens/sec vs concurrency, TTFT P50/P95 vs concurrency, ITL
P50/P95 vs concurrency, an interactivity-throughput frontier, and workload-shape
comparisons. The report template separates measured observations from bottleneck
hypotheses and lists additional profiling needed to validate those hypotheses.

## Running Against OpenAI-Compatible Hosted Endpoints

The harness can target local inference servers or hosted providers that expose
OpenAI-compatible APIs. Use:

- `--api-type completions` for `/v1/completions`
- `--api-type chat` for `/v1/chat/completions`

You can validate the chat request shape without any provider key:

```bash
python tests/test_llm_client.py
```

Before a benchmark run, validate auth, model access, and payload compatibility
with one tiny request. For Together AI:

```bash
export TOGETHER_API_KEY="your_key_here"

python validate_endpoint.py \
  --base-url https://api.together.xyz \
  --api-type chat \
  --api-key-env TOGETHER_API_KEY \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo
```

Then run a benchmark smoke test against the same endpoint:

```bash
python bench.py \
  --base-url https://api.together.xyz \
  --api-type chat \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --concurrency 1,2,4 \
  --max-tokens 128 \
  --requests-per-worker 2 \
  --out results/together_smoke.csv
```

If that succeeds, scale up gradually:

```bash
python bench_experiments.py \
  --base-url https://api.together.xyz \
  --api-type chat \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --concurrency 1,2,4,8,16 \
  --max-tokens 256 \
  --requests-per-worker 3 \
  --run-label together_llama_3_3_70b \
  --out results/together_llama_3_3_70b.csv
```

## Together AI Light Sweep

For Together AI specifically, `together_light_sweep.py` runs a conservative
hosted-API sweep with defaults intended for light-to-moderate load:

```bash
python together_light_sweep.py --yes
```

Before sending requests, the script prints the estimated total request count
and maximum generated-token budget. Without `--yes`, it asks for interactive
confirmation.

This measurement is different from the local dedicated-GPU saturation
experiments below. Together's hosted API runs on shared, rate-limited
infrastructure, so slowdowns, tail-latency growth, or HTTP 429 responses cannot
be attributed directly to GPU capacity or scheduler saturation the way they can
when you control the inference server and hardware. Treat this sweep as a
hosted endpoint behavior check under light-to-moderate client load, not as a
claim about Together's hardware saturation point.

### Example Run (Illustrative Only)

Model: `meta-llama/Llama-3.3-70B-Instruct-Turbo`  
Generation length: `max_tokens=256`

| Concurrency | tokens/sec | p50 latency | p95 latency | p99 latency |
|-------------|------------|-------------|-------------|-------------|
| 1 | 71 tok/s | 3.60s | 3.97s | 4.01s |
| 2 | 125 tok/s | 4.03s | 4.80s | 4.90s |
| 4 | 193 tok/s | 3.84s | 5.91s | 5.97s |
| 8 | 423 tok/s | 3.69s | 5.95s | 5.99s |
| 16 | 566 tok/s | 4.85s | 6.01s | 6.09s |

This is a single sample run against shared, third-party infrastructure,
included for illustration only. Unlike the local GPU experiments below, it is
not reproducible on demand, and the exact numbers will vary by time of day,
model, and current load on Together's platform. Do not present it with the same
permanence as Experiment 1/2/3.

Two observations are useful from this run. First, throughput scaled roughly
linearly across the tested range with no plateau, meaning this test did not
approach any real capacity ceiling. Second, p50 latency stayed relatively flat
across concurrency levels while p95 grew from about 4.0s to about 6.0s,
consistent with tail-latency queueing effects appearing before the median is
affected.

This is not a like-for-like comparison with the local experiments. The hosted
model here, `Llama-3.3-70B-Instruct-Turbo`, is roughly 10x larger than the
locally tested `Qwen2.5-7B-Instruct`, so higher absolute latency is expected
independent of any infrastructure difference. The meaningful comparison is the
shape of the curves, not the raw numbers.

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

## Experiment 1 - Concurrency Scaling

Experiments were run with three generation workloads:

| max_tokens | Workload Type |
|------------|--------------|
| 64 | short responses |
| 256 | typical assistant responses |
| 512 | long generations |

---

## Throughput (max_tokens = 256)

![Throughput](results/concurrency/throughput_max_tokens_256.png)

---

## Latency Percentiles (max_tokens = 256)

![Latency Percentiles vs Concurrency](results/concurrency/latency_max_tokens_256.png)

---

### Key Observations

Across all workloads the system saturated at approximately:

```
~1800 tokens/sec
```

on an **RTX 4090**.

- aggregate token throughput remained relatively stable
- request throughput decreased as generation length increased
- latency scaled roughly linearly with `max_tokens`
- saturation occurred around **~32 concurrent requests**

This indicates that once the GPU is fully utilized, the system transitions into a **decode-bound regime**, where throughput plateaus and latency becomes increasingly sensitive to request characteristics such as output length.

---

## Experiment 2 — Dynamic Batching Behavior

To better understand how inference schedulers handle request arrival patterns, we evaluated system performance under **burst** and **staggered** request arrivals near the saturation boundary.

Experiments were conducted at:

- concurrency: 24, 32, 40  
- max_tokens: 256  

---

### Arrival Patterns Tested

| Pattern | Description |
|--------|-------------|
| burst | all requests start simultaneously |
| staggered_25ms | each worker delayed by 25ms |
| staggered_50ms | each worker delayed by 50ms |

---

### Throughput vs Arrival Pattern

![Throughput vs Arrival Pattern](results/batching/throughput_stagger_compare.png)

---

### Results

| Concurrency | Burst | Staggered 25ms | Staggered 50ms |
|-------------|------|---------------|---------------|
| 24 | ~1308 tokens/s | ~1291 tokens/s | ~1244 tokens/s |
| 32 | **~1819 tokens/s** | ~1663 tokens/s | ~1584 tokens/s |
| 40 | ~1684 tokens/s | ~1600 tokens/s | ~1556 tokens/s |

---

### Key Observations

- Peak throughput is achieved under **burst arrivals**
- Staggering requests results in a **small but consistent reduction in throughput**
- The effect is most pronounced near the **saturation point (32 concurrency)**

---

### Interpretation

These results indicate that **vLLM’s continuous batching scheduler is already optimized for bursty traffic**.

Even when requests arrive simultaneously, the scheduler efficiently:

- queues incoming requests  
- dynamically forms large batches  
- maximizes GPU utilization during decoding  

Artificially smoothing request arrivals does not improve performance and can slightly reduce batching efficiency.

This reflects how modern inference engines operate:

> They rely on internal request queues and token-level schedulers to construct optimal batches during autoregressive decoding, rather than depending on externally controlled traffic shaping.

---

## Experiment 3 — Output Length Sensitivity Near Saturation

To understand how workload shape impacts system performance, we evaluated latency and throughput near the saturation boundary while varying generation length.

Experiments were conducted at:

- concurrency: 24, 32, 40  
- max_tokens: 64, 128, 256, 512  

---

### Results (32 Concurrency)

| max_tokens | tokens/sec | p95 latency |
|------------|-----------|------------|
| 64 | ~1740 | ~1.17s |
| 128 | ~1763 | ~2.32s |
| 256 | ~1768 | ~4.63s |
| 512 | ~similar throughput | ~higher latency |

---

### Latency vs Output Length

![Latency vs Output Length](results/output_length/output_length_p95_latency.png)

### Throughput vs Output Length

![Throughput vs Output Length](results/output_length/output_length_throughput.png)


### Key Observations

- Throughput remains **relatively stable near saturation** across different output lengths  
- Latency increases **approximately linearly with max_tokens**  
- Increasing concurrency beyond saturation does not improve throughput  

---

### Interpretation

At the saturation boundary, the system maintains high throughput due to efficient batching and GPU utilization.

However, **tail latency increases significantly for longer generations**, indicating that response time is dominated by decode duration rather than scheduling inefficiencies.

This reveals a key production insight:

> Throughput alone is not sufficient to evaluate inference performance. Workload characteristics, especially output length, directly impact latency and user experience.

---

### GPU Profiling (Nsight Systems)

To validate system behavior at saturation, a lightweight Nsight Systems profile was performed for a representative workload:

- concurrency: 32  
- max_tokens: 512  

The profiling timeline showed:

- sustained GPU activity throughout request processing  
- minimal idle gaps between kernel executions  
- consistent kernel execution patterns during decoding  

This supports the earlier observation that:

> Latency increases near saturation are driven by decode duration and queueing effects, rather than GPU underutilization.

Raw profiling artifacts are not included in the repository.

---

### Practical Implication

Inference systems should be sized based on:

- expected output length distributions  
- latency SLOs (p95 / p99)  
- not just peak tokens/sec capacity  

This is particularly important for applications such as:

- chat assistants (short responses)  
- agent workflows (medium responses)  
- code generation or summarization (long responses)

---

### Motivation

LLM inference performance is often evaluated using **single-request latency**, which does not reflect real production conditions.

This experiment focuses on **concurrency-driven load behavior**, providing a more realistic view of:

- GPU utilization under load  
- batching efficiency  
- scheduler effectiveness  
- system behavior near saturation  

By simulating bursty and staggered traffic patterns, we better approximate real-world request distributions and uncover how inference systems behave under sustained load.

---

# System Insights

This benchmark highlights several key properties of modern LLM inference systems:

- **Saturation occurs at a concurrency threshold**, beyond which additional load increases latency without improving throughput  
- **Throughput alone is not a sufficient performance metric** — tail latency reveals system stress earlier  
- **Modern schedulers (e.g., vLLM) effectively handle bursty traffic**, reducing the need for external request smoothing  
- **Decode-heavy workloads are memory-bandwidth bound**, leading to near-linear latency scaling with output length  
- **Workload shape (output length, concurrency)** directly impacts system behavior and must be considered in capacity planning  

These findings emphasize the importance of evaluating inference systems under realistic load conditions rather than relying solely on single-request benchmarks.

---

# PyTorch Profiler Analysis

To better understand operator-level execution behavior during inference, a lightweight PyTorch Profiler workflow was added using:

- PyTorch Profiler
- Perfetto trace visualization
- CUDA activity tracing

The profiling workflow captured:

- `aten::scaled_dot_product_attention`
- Flash Attention execution paths
- CUDA kernel dispatch activity
- matrix multiplication operators (`aten::matmul`)
- linear projection layers (`aten::linear`)

This establishes a baseline workflow for analyzing:
- operator-level bottlenecks
- attention-heavy execution regions
- CPU-to-GPU dispatch behavior
- decode-phase execution characteristics

Example Perfetto trace:

![Perfetto Trace](results/profiling/images/Perfetto_trace_file.png)

# Future Work

- compare vLLM vs TensorRT-LLM scheduling behavior
- analyze GPU kernel timelines with Nsight Systems
- correlate throughput saturation with GPU utilization metrics
- evaluate KV-cache pressure under long-context workloads
- benchmark multi-GPU inference scaling and NCCL communication overhead
- analyze batching efficiency under heterogeneous request mixes

---

# License

MIT License  
Marco Punio — 2026
