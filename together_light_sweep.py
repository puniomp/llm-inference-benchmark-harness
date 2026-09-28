import argparse
import asyncio
import json
import os
import statistics
import time

import pandas as pd

from llm_client import call_once


def parse_concurrency(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def confirm_run(total_requests: int, args) -> bool:
    print("Together light sweep")
    print(f"  base_url: {args.base_url}")
    print(f"  model: {args.model}")
    print(f"  api_type: {args.api_type}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  requests_per_worker: {args.requests_per_worker}")
    print(f"  estimated_total_requests: {total_requests}")
    print(f"  estimated_max_output_tokens: {total_requests * args.max_tokens}")

    if args.yes:
        return True

    answer = input("Proceed with this hosted API sweep? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


async def run_concurrency(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    requests_per_worker: int,
    api_type: str,
    api_key: str | None,
):
    loop = asyncio.get_event_loop()

    async def worker():
        latencies, out_toks = [], []
        for _ in range(requests_per_worker):
            dt, ot = await loop.run_in_executor(
                None,
                call_once,
                base_url,
                model,
                prompt,
                max_tokens,
                api_type,
                api_key,
            )
            latencies.append(dt)
            out_toks.append(ot)
        return latencies, out_toks

    t0 = time.time()
    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    results = await asyncio.gather(*tasks)
    t1 = time.time()

    lat = [x for l, _ in results for x in l]
    toks = [x for _, t in results for x in t]
    total_out = sum(toks)
    elapsed = t1 - t0

    tokens_per_sec = (total_out / elapsed) if elapsed > 0 else 0.0
    requests_per_sec = (len(lat) / elapsed) if elapsed > 0 else 0.0

    return lat, toks, elapsed, requests_per_sec, tokens_per_sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://api.together.xyz")
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    ap.add_argument("--api-type", choices=["completions", "chat"], default="chat")
    ap.add_argument("--api-key-env", default="TOGETHER_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=str, default="1,2,4,8,16")
    ap.add_argument("--requests-per-worker", type=int, default=2)
    ap.add_argument("--prompt-idx", type=int, default=0)
    ap.add_argument("--run-label", type=str, default="together_light_sweep")
    ap.add_argument("--out", default="results/together_light_sweep.csv")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    concurrencies = parse_concurrency(args.concurrency)
    total_requests = sum(concurrencies) * args.requests_per_worker
    if not confirm_run(total_requests, args):
        print("Canceled.")
        return 1

    api_key = (os.getenv(args.api_key_env) or "").strip()
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set.")

    with open("prompts.json", "r") as f:
        prompts = json.load(f)
    prompt = prompts[args.prompt_idx]

    rows = []
    for c in concurrencies:
        lat, toks, elapsed, rps, tps = asyncio.run(
            run_concurrency(
                args.base_url,
                args.model,
                prompt,
                args.max_tokens,
                c,
                args.requests_per_worker,
                args.api_type,
                api_key,
            )
        )

        row = {
            "run_label": args.run_label,
            "prompt_idx": args.prompt_idx,
            "max_tokens": args.max_tokens,
            "concurrency": c,
            "n_requests": len(lat),
            "elapsed_s": elapsed,
            "req_per_sec": rps,
            "out_tokens_total": sum(toks),
            "tokens_per_sec": tps,
            "lat_p50_s": pct(lat, 50),
            "lat_p95_s": pct(lat, 95),
            "lat_p99_s": pct(lat, 99),
            "lat_mean_s": statistics.mean(lat) if lat else None,
        }
        rows.append(row)
        print(row)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
