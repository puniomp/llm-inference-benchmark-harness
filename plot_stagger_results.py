import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/batching/stagger_compare.csv")
    ap.add_argument("--out-prefix", default="results/batching")
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    df = df.sort_values(["arrival_pattern", "stagger_ms", "concurrency"])

    df["series"] = df.apply(
        lambda r: "burst"
        if r["arrival_pattern"] == "burst"
        else f"staggered_{int(r['stagger_ms'])}ms",
        axis=1,
    )

    plt.figure()
    for series_name, g in df.groupby("series"):
        plt.plot(g["concurrency"], g["tokens_per_sec"], marker="o", label=series_name)

    plt.xlabel("Concurrency")
    plt.ylabel("Tokens / Second")
    plt.title("Throughput vs Concurrency: Burst vs Staggered Arrivals")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.out_prefix}/throughput_stagger_compare.png")

    plt.figure()
    for series_name, g in df.groupby("series"):
        plt.plot(g["concurrency"], g["lat_p95_s"], marker="o", label=f"{series_name} p95")

    plt.xlabel("Concurrency")
    plt.ylabel("Latency (seconds)")
    plt.title("p95 Latency vs Concurrency: Burst vs Staggered Arrivals")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.out_prefix}/latency_stagger_compare.png")

    print("Saved plots:")
    print(f"{args.out_prefix}/throughput_stagger_compare.png")
    print(f"{args.out_prefix}/latency_stagger_compare.png")


if __name__ == "__main__":
    main()
