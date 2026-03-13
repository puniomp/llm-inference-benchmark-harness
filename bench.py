import asyncio
import json
import time
import statistics
import argparse

import requests
import pandas as pd


def call_once(base_url: str, model: str, prompt: str, max_tokens: int):
    t0 = time.time()
    r = requests.post(
        f"{base_url}/v1/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=600,
    )
    r.raise_for_status()
    t1 = time.time()

    data = r.json()
    usage = data.get("usage", {})
    out_tokens = usage.get("completion_tokens", 0) or 0
    return (t1 - t0), out_tokens


async def run_concurrency(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    requests_per_worker: int,
):
    loop = asyncio.get_event_loop()

    async def worker():
        latencies, out_toks = [], []
        for _ in range(requests_per_worker):
            dt, ot = await loop.run_in_executor(
                None, call_once, base_url, model, prompt, max_tokens
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=str, default="1,2,4,8,16")
    ap.add_argument("--requests-per-worker", type=int, default=3)
    ap.add_argument("--prompt-idx", type=int, default=0)
    ap.add_argument("--out", default="results/run.csv")
    args = ap.parse_args()

    prompts = json.load(open("prompts.json"))
    prompt = prompts[args.prompt_idx]

    rows = []
    for c in [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]:
        lat, toks, elapsed, rps, tps = asyncio.run(
            run_concurrency(
                args.base_url,
                args.model,
                prompt,
                args.max_tokens,
                c,
                args.requests_per_worker,
            )
        )

        row = {
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

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
