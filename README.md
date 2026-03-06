# LLM Inference Benchmark Harness
A lightweight Python harness for benchmarking LLM inference servers under increasing concurrency.

The goal is to measure throughput scaling and latency behavior (p50 / p95 / p99) as load increases and identify the saturation point of an inference system.

This harness targets OpenAI-compatible endpoints such as:
vLLM
Triton Inference Server
TensorRT-LLM
OpenAI API-compatible gateways

# What I measure
For each concurrency level the harness records:
number of requests
elapsed wall time
requests/sec
tokens/sec
latency p50
latency p95
latency p99
mean latency

The benchmark sweeps across increasing concurrency levels and produces plots showing:
1. Throughput scaling
2. Tail latency growth

These curves make it easy to identify when the system transitions from efficient utilization to queueing and saturation.

# Example Output
Throughput vs Concurrency

shows how token throughput scales with increased load

Latency Percentiles vs Concurrency

shows the growth of tail latency as concurrency increases

# Running the Benchmark
Step 1 — Start an inference server

Example using vLLM:

python -m vllm.entrypoints.openai.api_server
--model Qwen/Qwen2.5-7B-Instruct
--host 0.0.0.0
--port 8000

This exposes an OpenAI-compatible endpoint at:

http://localhost:8000/v1/completions

### Step 2 — Run the concurrency sweep
python bench.py
--model Qwen/Qwen2.5-7B-Instruct
--concurrency 1,2,4,8,16,24,32,48,64
--max-tokens 256
--requests-per-worker 5
--out results/run_2.csv

This generates:
results/run_2.csv

### Step 3 — Generate plots

python plot_results.py

This produces:
results/throughput.png
results/latency_percentiles.png

# Example Results
A typical benchmark pattern looks like:
- throughput scales nearly linearly at low concurrency
- eventually throughput plateaus
- tail latency (p95/p99) grows rapidly
- this indicates the system saturation region
- 
This harness makes it easy to visualize that transition.

# Why did I build this?
LLM inference performance is often reported using single-request latency, which hides how systems behave under real load.

This harness focuses on concurrency-driven saturation testing, a more realistic method for evaluating:
- GPU utilization
- batching efficiency
- scheduler behavior
- inference server scalability
  
# License
MIT
