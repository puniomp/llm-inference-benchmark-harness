# Together AI Hosted Endpoint

The harness can target local inference servers or hosted providers that expose OpenAI-compatible APIs. Together AI should be treated as a hosted endpoint behavior check, not as a controlled GPU saturation experiment.

Use:

- `--api-type completions` for `/v1/completions`
- `--api-type chat` for `/v1/chat/completions`

## Validate Request Shape Locally

You can validate the client request construction without a provider key:

```bash
python tests/test_llm_client.py
```

## Validate Together Auth and Model Access

Run one tiny request before benchmarking:

```bash
export TOGETHER_API_KEY="your_key_here"

python validate_endpoint.py \
  --base-url https://api.together.xyz \
  --api-type chat \
  --api-key-env TOGETHER_API_KEY \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo
```

## Smoke Test

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

## Together AI Light Sweep

For Together AI specifically, `together_light_sweep.py` runs a conservative hosted-API sweep with defaults intended for light-to-moderate load:

```bash
python together_light_sweep.py --yes
```

Defaults:

| Field | Value |
|-------|-------|
| Base URL | `https://api.together.xyz` |
| API type | `chat` |
| API key env var | `TOGETHER_API_KEY` |
| Model | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Concurrency | `1,2,4,8,16` |
| max_tokens | 256 |
| requests_per_worker | 2 |

Before sending requests, the script prints the estimated total request count and maximum generated-token budget. Without `--yes`, it asks for interactive confirmation.

This measurement is different from the local dedicated-GPU saturation experiments. Together's hosted API runs on shared, rate-limited infrastructure, so slowdowns, tail-latency growth, or HTTP 429 responses cannot be attributed directly to GPU capacity or scheduler saturation the way they can when you control the inference server and hardware. Treat this sweep as a hosted endpoint behavior check under light-to-moderate client load, not as a claim about Together's hardware saturation point.

## Example Run, Illustrative Only

Model: `meta-llama/Llama-3.3-70B-Instruct-Turbo`  
Generation length: `max_tokens=256`

| Concurrency | tokens/sec | p50 latency | p95 latency | p99 latency |
|-------------|------------|-------------|-------------|-------------|
| 1 | 71 tok/s | 3.60 s | 3.97 s | 4.01 s |
| 2 | 125 tok/s | 4.03 s | 4.80 s | 4.90 s |
| 4 | 193 tok/s | 3.84 s | 5.91 s | 5.97 s |
| 8 | 423 tok/s | 3.69 s | 5.95 s | 5.99 s |
| 16 | 566 tok/s | 4.85 s | 6.01 s | 6.09 s |

This is a single sample run against shared, third-party infrastructure, included for illustration only. Unlike the local GPU experiments, it is not reproducible on demand, and the exact numbers will vary by time of day, model, and current load on Together's platform. Do not present it with the same permanence as controlled local experiments.

Two observations are useful from this run. First, throughput scaled roughly linearly across the tested range with no plateau, meaning this test did not approach any real capacity ceiling. Second, p50 latency stayed relatively flat across concurrency levels while p95 grew from about 4.0 s to about 6.0 s, consistent with tail-latency queueing effects appearing before the median is affected.

This is not a like-for-like comparison with the local experiments. The hosted model here, `Llama-3.3-70B-Instruct-Turbo`, is roughly 10x larger than the locally tested `Qwen2.5-7B-Instruct`, so higher absolute latency is expected independent of any infrastructure difference. The meaningful comparison is the shape of the curves, not the raw numbers.
