import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Optional

import pandas as pd
import requests

from llm_client import _headers, _raise_for_status


@dataclass
class GpuSample:
    ts_s: float
    gpu_index: int
    utilization_gpu_pct: Optional[float]
    utilization_memory_pct: Optional[float]
    memory_used_mb: Optional[float]
    memory_total_mb: Optional[float]


class GpuSampler:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[GpuSample] = []
        self.available = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not self._query_once():
            return
        self.available = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 1)

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self._query_once()

    def _query_once(self) -> bool:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return False

        now = time.time()
        for line in result.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 5:
                continue
            self.samples.append(
                GpuSample(
                    ts_s=now,
                    gpu_index=int(float(parts[0])),
                    utilization_gpu_pct=_float_or_none(parts[1]),
                    utilization_memory_pct=_float_or_none(parts[2]),
                    memory_used_mb=_float_or_none(parts[3]),
                    memory_total_mb=_float_or_none(parts[4]),
                )
            )
        return True

    def summary(self) -> dict:
        if not self.samples:
            return {
                "gpu_telemetry_available": False,
                "gpu_util_mean_pct": None,
                "gpu_util_max_pct": None,
                "gpu_mem_util_mean_pct": None,
                "gpu_mem_used_max_mb": None,
            }
        gpu_utils = [s.utilization_gpu_pct for s in self.samples if s.utilization_gpu_pct is not None]
        mem_utils = [
            s.utilization_memory_pct
            for s in self.samples
            if s.utilization_memory_pct is not None
        ]
        mem_used = [s.memory_used_mb for s in self.samples if s.memory_used_mb is not None]
        return {
            "gpu_telemetry_available": True,
            "gpu_util_mean_pct": statistics.mean(gpu_utils) if gpu_utils else None,
            "gpu_util_max_pct": max(gpu_utils) if gpu_utils else None,
            "gpu_mem_util_mean_pct": statistics.mean(mem_utils) if mem_utils else None,
            "gpu_mem_used_max_mb": max(mem_used) if mem_used else None,
        }


def _float_or_none(value: str) -> Optional[float]:
    if value in {"", "[N/A]", "N/A"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def endpoint(base_url: str, api_type: str) -> str:
    root = base_url.rstrip("/")
    if api_type == "chat":
        return f"{root}/v1/chat/completions"
    return f"{root}/v1/completions"


def extract_delta_text(event: dict, api_type: str) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if api_type == "chat":
        return (choice.get("delta") or {}).get("content") or ""
    return choice.get("text") or ""


def tokenize_vllm(base_url: str, model: str, text: str, api_key: Optional[str]) -> Optional[int]:
    url = f"{base_url.rstrip('/')}/tokenize"
    payload = {"model": model, "prompt": text}
    try:
        response = requests.post(
            url,
            headers=_headers(api_key),
            json=payload,
            timeout=30,
        )
        if response.status_code >= 400:
            return None
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    for key in ("count", "token_count", "num_tokens"):
        if isinstance(data.get(key), int):
            return data[key]
    tokens = data.get("tokens") or data.get("token_ids")
    if isinstance(tokens, list):
        return len(tokens)
    return None


def make_prompt(target_input_tokens: int, seed: int, base_url: str, model: str, api_key: Optional[str]) -> tuple[str, Optional[int]]:
    phrase = (
        f"Request seed {seed}. Explain how batching, prefill, and decode interact "
        "in transformer inference. Keep the reasoning concrete. "
    )
    prompt = phrase
    measured = tokenize_vllm(base_url, model, prompt, api_key)
    while measured is not None and measured < target_input_tokens:
        prompt += phrase
        measured = tokenize_vllm(base_url, model, prompt, api_key)
    if measured is not None:
        return prompt, measured

    approx_words = max(1, int(target_input_tokens * 0.75))
    prompt = " ".join([phrase] * max(1, approx_words // len(phrase.split())))
    return prompt, None


def package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def torch_cuda_version() -> Optional[str]:
    try:
        import torch
    except ImportError:
        return None
    return getattr(torch.version, "cuda", None)


def stream_once(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_type: str,
    api_key: Optional[str],
    request_id: str,
    input_tokens: Optional[int],
    target_input_tokens: int,
) -> dict:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    if api_type == "chat":
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt

    t0 = time.time()
    first_token_ts = None
    last_token_ts = None
    text_parts: list[str] = []
    stream_event_count = 0

    response = requests.post(
        endpoint(base_url, api_type),
        headers=_headers(api_key),
        json=payload,
        timeout=900,
        stream=True,
    )
    _raise_for_status(response)

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = extract_delta_text(event, api_type)
        if not delta:
            continue
        now = time.time()
        if first_token_ts is None:
            first_token_ts = now
        last_token_ts = now
        stream_event_count += 1
        text_parts.append(delta)

    t1 = time.time()
    output_text = "".join(text_parts)
    output_tokens = tokenize_vllm(base_url, model, output_text, api_key)
    if output_tokens is None:
        output_tokens = stream_event_count

    latency_s = t1 - t0
    ttft_s = (first_token_ts - t0) if first_token_ts is not None else None
    decode_s = (
        (last_token_ts - first_token_ts)
        if first_token_ts is not None and last_token_ts is not None
        else None
    )
    itl_mean_s = (
        decode_s / (output_tokens - 1)
        if decode_s is not None and output_tokens and output_tokens > 1
        else None
    )
    tpot_s = (
        (latency_s - (ttft_s or 0.0)) / output_tokens
        if output_tokens and output_tokens > 0
        else None
    )
    user_output_tokens_per_sec = (
        (output_tokens - 1) / decode_s
        if decode_s and output_tokens and output_tokens > 1
        else None
    )

    return {
        "request_id": request_id,
        "input_tokens": input_tokens,
        "target_input_tokens": target_input_tokens,
        "requested_output_tokens": max_tokens,
        "output_tokens": output_tokens,
        "stream_events": stream_event_count,
        "latency_s": latency_s,
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "itl_mean_s": itl_mean_s,
        "tpot_s": tpot_s,
        "user_output_tokens_per_sec": user_output_tokens_per_sec,
    }


async def run_profile(args, profile: dict, concurrency: int, api_key: Optional[str]) -> tuple[list[dict], dict]:
    prompt, input_tokens = make_prompt(
        profile["input_tokens"],
        args.seed,
        args.base_url,
        args.model,
        api_key,
    )
    loop = asyncio.get_event_loop()

    async def worker(worker_idx: int):
        rows = []
        for request_idx in range(args.requests_per_worker):
            request_id = f"{profile['name']}-c{concurrency}-w{worker_idx}-r{request_idx}"
            row = await loop.run_in_executor(
                None,
                stream_once,
                args.base_url,
                args.model,
                prompt,
                profile["output_tokens"],
                args.api_type,
                api_key,
                request_id,
                input_tokens,
                profile["input_tokens"],
            )
            rows.append(row)
        return rows

    sampler = GpuSampler(args.gpu_sample_interval_s)
    sampler.start()
    t0 = time.time()
    results = await asyncio.gather(
        *[asyncio.create_task(worker(i)) for i in range(concurrency)]
    )
    elapsed_s = time.time() - t0
    sampler.stop()

    raw_rows = [row for worker_rows in results for row in worker_rows]
    for row in raw_rows:
        row.update(
            {
                "experiment": args.experiment,
                "profile": profile["name"],
                "concurrency": concurrency,
                "model": args.model,
                "api_type": args.api_type,
            }
        )

    total_output_tokens = sum(row["output_tokens"] or 0 for row in raw_rows)
    gpu = sampler.summary()
    summary = {
        "experiment": args.experiment,
        "profile": profile["name"],
        "concurrency": concurrency,
        "n_requests": len(raw_rows),
        "input_tokens": input_tokens,
        "target_input_tokens": profile["input_tokens"],
        "tokenizer_available": input_tokens is not None,
        "requested_output_tokens": profile["output_tokens"],
        "elapsed_s": elapsed_s,
        "requests_per_sec": len(raw_rows) / elapsed_s if elapsed_s > 0 else None,
        "aggregate_output_tokens_per_sec": total_output_tokens / elapsed_s if elapsed_s > 0 else None,
        "user_output_tokens_per_sec_p50": pct(
            [row["user_output_tokens_per_sec"] for row in raw_rows], 50
        ),
        "user_output_tokens_per_sec_mean": statistics.mean(
            [
                row["user_output_tokens_per_sec"]
                for row in raw_rows
                if row["user_output_tokens_per_sec"] is not None
            ]
        )
        if any(row["user_output_tokens_per_sec"] is not None for row in raw_rows)
        else None,
        "latency_p50_s": pct([row["latency_s"] for row in raw_rows], 50),
        "latency_p95_s": pct([row["latency_s"] for row in raw_rows], 95),
        "ttft_p50_s": pct([row["ttft_s"] for row in raw_rows], 50),
        "ttft_p95_s": pct([row["ttft_s"] for row in raw_rows], 95),
        "itl_p50_s": pct([row["itl_mean_s"] for row in raw_rows], 50),
        "itl_p95_s": pct([row["itl_mean_s"] for row in raw_rows], 95),
        "tpot_p50_s": pct([row["tpot_s"] for row in raw_rows], 50),
        "tpot_p95_s": pct([row["tpot_s"] for row in raw_rows], 95),
        **gpu,
    }
    return raw_rows, summary


def workload_profiles(args) -> list[dict]:
    if args.experiment == "workload-shape":
        return [
            {"name": "baseline_128in_128out", "input_tokens": 128, "output_tokens": 128},
            {"name": "prefill_4096in_128out", "input_tokens": 4096, "output_tokens": 128},
            {"name": "decode_128in_2048out", "input_tokens": 128, "output_tokens": 2048},
        ]
    return [
        {
            "name": "baseline_128in_128out",
            "input_tokens": args.input_tokens,
            "output_tokens": args.max_tokens,
        }
    ]


def write_outputs(args, raw_rows: list[dict], summaries: list[dict]):
    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, f"{args.experiment}_raw.csv")
    summary_path = os.path.join(args.out_dir, f"{args.experiment}_summary.csv")
    metadata_path = os.path.join(args.out_dir, f"{args.experiment}_metadata.json")

    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "api_type": args.api_type,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vllm_version": package_version("vllm"),
        "pytorch_version": package_version("torch"),
        "cuda_version": torch_cuda_version(),
        "precision": args.precision,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "vllm_config": args.vllm_config,
        "seed": args.seed,
        "requests_per_worker": args.requests_per_worker,
        "notes": [
            "TTFT is measured from request submission to first non-empty streamed delta.",
            "ITL is request-level mean inter-token latency: decode duration divided by output_tokens - 1.",
            "Output token counts use the server /tokenize endpoint when available; otherwise stream event count is used as fallback.",
            "Warmup results are not written into benchmark result files.",
        ],
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote raw: {raw_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote metadata: {metadata_path}")


def confirm(args, total_requests: int):
    print(f"{args.experiment} benchmark")
    print(f"  model: {args.model}")
    print(f"  base_url: {args.base_url}")
    print(f"  api_type: {args.api_type}")
    print(f"  estimated measured requests: {total_requests}")
    if args.yes:
        return True
    return input("Proceed? [y/N] ").strip().lower() in {"y", "yes"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["warmup", "concurrency", "workload-shape"], required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-type", choices=["completions", "chat"], default="completions")
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--concurrency", default="1,8,32,64,128,256")
    ap.add_argument("--workload-concurrency", type=int, default=16)
    ap.add_argument("--requests-per-worker", type=int, default=2)
    ap.add_argument("--input-tokens", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", default="results/streaming")
    ap.add_argument("--gpu-sample-interval-s", type=float, default=1.0)
    ap.add_argument("--precision", default="")
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--vllm-config", default="")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    api_key = (os.getenv(args.api_key_env) or "").strip() if args.api_key_env else None

    if args.experiment == "warmup":
        args.requests_per_worker = max(1, args.requests_per_worker)
        profile = {"name": "warmup", "input_tokens": args.input_tokens, "output_tokens": args.max_tokens}
        asyncio.run(run_profile(args, profile, 1, api_key))
        print("Warmup complete; results were intentionally not written.")
        return 0

    profiles = workload_profiles(args)
    concurrencies = (
        parse_int_list(args.concurrency)
        if args.experiment == "concurrency"
        else [args.workload_concurrency]
    )
    total_requests = len(profiles) * sum(concurrencies) * args.requests_per_worker
    if not confirm(args, total_requests):
        print("Canceled.")
        return 1

    all_raw_rows: list[dict] = []
    summaries: list[dict] = []
    for profile in profiles:
        for concurrency in concurrencies:
            raw_rows, summary = asyncio.run(run_profile(args, profile, concurrency, api_key))
            all_raw_rows.extend(raw_rows)
            summaries.append(summary)
            print(summary)

    write_outputs(args, all_raw_rows, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
