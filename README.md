<<<<<<< HEAD
# llm-inference-benchmark-harness
A reproducible benchmarking framework for evaluating large language model inference performance across GPU architectures and serving stacks. The project measures throughput, latency, and scaling behavior across different model sizes using modern inference engines such as vLLM and TensorRT-LLM.
=======
# LLM Inference Benchmark Harness

A lightweight benchmarking harness for measuring LLM inference performance using an OpenAI-compatible API (vLLM).

This project measures:

- Latency (p50 / p95)
- Throughput (tokens/sec)
- Performance scaling with concurrent requests

The goal is to analyze how optimized inference runtimes scale under load.

---

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install vllm requests numpy pandas matplotlib
>>>>>>> df8800b (Initial LLM inference benchmark harness (vLLM))
