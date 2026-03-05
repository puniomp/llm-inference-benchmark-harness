# LLM Inference Benchmark Harness

A lightweight benchmarking harness for measuring **LLM inference throughput and latency** using an OpenAI-compatible endpoint (e.g., vLLM).

This project helps evaluate GPU inference performance across different concurrency levels.

## Features

- Measures request latency
- Estimates tokens/sec throughput
- Supports configurable concurrency
- Outputs results to CSV
- Optional visualization of scaling curves

## Requirements

- Python 3.10+
- vLLM running locally or remotely
- OpenAI-compatible API endpoint

Example server:

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000

## Install

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run Benchmark

python bench.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 1,2,4,8,16 \
  --max-tokens 256 \
  --requests-per-worker 3 \
  --out results/run_1.csv

## Plot Results

python plot_results.py

Outputs

results/run_1.csv  
results/throughput.png

## Future Work

- Compare vLLM vs HuggingFace Transformers
- Compare GPU types (4090 vs A100 vs H100)
- Add latency percentiles
- Add streaming benchmarks

