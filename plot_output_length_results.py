import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/output_length_sweep.csv")
    ap.add_argument("--out-prefix", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_prefix, exist_ok=True)

    df = pd.read_csv(args.infile)
    df = df.sort_values(["concurrency", "max_tokens"])

    plt.figure()
    for concurrency, g in df.groupby("concurrency"):
        g = g.sort_values("max_tokens")
        plt.plot(g["max_tokens"], g["lat_p95_s"], marker="o", label=f"conc={concurrency}")
    plt.xlabel("Max Tokens")
    plt.ylabel("p95 Latency (seconds)")
    plt.title("p95 Latency vs Output Length Near Saturation")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.out_prefix}/output_length_p95_latency.png")

    plt.figure()
    for concurrency, g in df.groupby("concurrency"):
        g = g.sort_values("max_tokens")
        plt.plot(g["max_tokens"], g["lat_p99_s"], marker="o", label=f"conc={concurrency}")
    plt.xlabel("Max Tokens")
    plt.ylabel("p99 Latency (seconds)")
    plt.title("p99 Latency vs Output Length Near Saturation")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.out_prefix}/output_length_p99_latency.png")

    plt.figure()
    for concurrency, g in df.groupby("concurrency"):
        g = g.sort_values("max_tokens")
        plt.plot(g["max_tokens"], g["tokens_per_sec"], marker="o", label=f"conc={concurrency}")
    plt.xlabel("Max Tokens")
    plt.ylabel("Tokens / Second")
    plt.title("Throughput vs Output Length Near Saturation")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.out_prefix}/output_length_throughput.png")

    print("Saved plots:")
    print(f"{args.out_prefix}/output_length_p95_latency.png")
    print(f"{args.out_prefix}/output_length_p99_latency.png")
    print(f"{args.out_prefix}/output_length_throughput.png")


if __name__ == "__main__":
    main()
