import argparse
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_plot(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Wrote: {path}")


def plot_concurrency(df: pd.DataFrame, out_dir: str):
    cdf = df[df["experiment"] == "concurrency"].sort_values("concurrency")
    if cdf.empty:
        return []

    paths = []
    plt.figure()
    plt.plot(cdf["concurrency"], cdf["aggregate_output_tokens_per_sec"], marker="o")
    plt.xlabel("Concurrency")
    plt.ylabel("Aggregate output tokens/sec")
    plt.grid(True)
    path = os.path.join(out_dir, "concurrency_aggregate_tokens_per_sec.png")
    save_plot(path)
    paths.append(path)

    plt.figure()
    plt.plot(cdf["concurrency"], cdf["user_output_tokens_per_sec_p50"], marker="o")
    plt.xlabel("Concurrency")
    plt.ylabel("P50 output tokens/sec/user")
    plt.grid(True)
    path = os.path.join(out_dir, "concurrency_tokens_per_sec_per_user.png")
    save_plot(path)
    paths.append(path)

    plt.figure()
    plt.plot(cdf["concurrency"], cdf["ttft_p50_s"], marker="o", label="TTFT p50")
    plt.plot(cdf["concurrency"], cdf["ttft_p95_s"], marker="o", label="TTFT p95")
    plt.xlabel("Concurrency")
    plt.ylabel("TTFT (s)")
    plt.legend()
    plt.grid(True)
    path = os.path.join(out_dir, "concurrency_ttft.png")
    save_plot(path)
    paths.append(path)

    plt.figure()
    plt.plot(cdf["concurrency"], cdf["itl_p50_s"], marker="o", label="ITL p50")
    plt.plot(cdf["concurrency"], cdf["itl_p95_s"], marker="o", label="ITL p95")
    plt.xlabel("Concurrency")
    plt.ylabel("ITL (s/token)")
    plt.legend()
    plt.grid(True)
    path = os.path.join(out_dir, "concurrency_itl.png")
    save_plot(path)
    paths.append(path)

    plt.figure()
    plt.scatter(cdf["user_output_tokens_per_sec_p50"], cdf["aggregate_output_tokens_per_sec"])
    for _, row in cdf.iterrows():
        plt.annotate(str(int(row["concurrency"])), (row["user_output_tokens_per_sec_p50"], row["aggregate_output_tokens_per_sec"]))
    plt.xlabel("P50 output tokens/sec/user")
    plt.ylabel("Aggregate output tokens/sec")
    plt.grid(True)
    path = os.path.join(out_dir, "interactivity_throughput_frontier.png")
    save_plot(path)
    paths.append(path)
    return paths


def plot_workloads(df: pd.DataFrame, out_dir: str):
    wdf = df[df["experiment"] == "workload-shape"]
    if wdf.empty:
        return []
    wdf = wdf.sort_values("profile")
    labels = wdf["profile"]
    paths = []

    plt.figure(figsize=(9, 4))
    plt.bar(labels, wdf["aggregate_output_tokens_per_sec"])
    plt.ylabel("Aggregate output tokens/sec")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y")
    path = os.path.join(out_dir, "workload_aggregate_tokens_per_sec.png")
    save_plot(path)
    paths.append(path)

    plt.figure(figsize=(9, 4))
    plt.plot(labels, wdf["ttft_p50_s"], marker="o", label="TTFT p50")
    plt.plot(labels, wdf["itl_p50_s"], marker="o", label="ITL p50")
    plt.ylabel("Seconds")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.grid(True)
    path = os.path.join(out_dir, "workload_ttft_itl.png")
    save_plot(path)
    paths.append(path)
    return paths


def write_report(df: pd.DataFrame, images: list[str], path: str):
    lines = [
        "# Streaming Inference Performance Report",
        "",
        "## Experimental Setup",
        "",
        "Document model, hardware, GPU count, vLLM version, PyTorch version, CUDA version, precision/quantization, tensor parallel size, max model length, and relevant vLLM configuration here.",
        "",
        "## Metric Formulas",
        "",
        "- Request latency: request completion timestamp - request submission timestamp.",
        "- TTFT: first non-empty streamed token timestamp - request submission timestamp.",
        "- ITL: request-level decode duration / (output tokens - 1), where decode duration is last streamed token timestamp - first streamed token timestamp.",
        "- TPOT: (request latency - TTFT) / output tokens.",
        "- Output tokens/sec/user: (output tokens - 1) / decode duration.",
        "- Aggregate output tokens/sec: total generated output tokens / benchmark wall-clock duration.",
        "- Requests/sec: completed requests / benchmark wall-clock duration.",
        "",
        "## Results",
        "",
        markdown_table(df),
        "",
        "## Plots",
        "",
    ]
    for image in images:
        rel = os.path.relpath(image, os.path.dirname(path))
        lines.append(f"![{os.path.basename(image)}]({rel})")
        lines.append("")
    lines.extend(
        [
            "## Observed Throughput/Interactivity Tradeoff",
            "",
            "Summarize measured changes in aggregate throughput, per-user generation speed, TTFT, and ITL as concurrency increases.",
            "",
            "## Prefill-Heavy vs Decode-Heavy Behavior",
            "",
            "Compare the baseline, prefill-heavy, and decode-heavy profiles using the measured TTFT, ITL, per-user tokens/sec, and aggregate tokens/sec.",
            "",
            "## Bottleneck Hypotheses",
            "",
            "Separate observations from hypotheses. Do not claim compute-bound or memory-bound behavior solely from throughput curves.",
            "",
            "## Additional Profiling Needed",
            "",
            "List profiling that would validate hypotheses, such as vLLM scheduler metrics, CUDA kernel traces, memory bandwidth counters, or queue-depth telemetry.",
        ]
    )
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {path}")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", default="results/streaming/concurrency_summary.csv")
    ap.add_argument("--extra-summary-csv", action="append", default=[])
    ap.add_argument("--out-dir", default="results/streaming/plots")
    ap.add_argument("--report", default="results/streaming/report.md")
    args = ap.parse_args()

    frames = [pd.read_csv(args.summary_csv)]
    frames.extend(pd.read_csv(path) for path in args.extra_summary_csv)
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    images = []
    images.extend(plot_concurrency(df, args.out_dir))
    images.extend(plot_workloads(df, args.out_dir))
    write_report(df, images, args.report)


if __name__ == "__main__":
    main()
