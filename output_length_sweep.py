import argparse
import subprocess
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", default="24,32,40")
    ap.add_argument("--max-tokens-list", default="64,128,256,512")
    ap.add_argument("--requests-per-worker", type=int, default=3)
    ap.add_argument("--prompt-idx", type=int, default=0)
    ap.add_argument("--out", default="results/output_length_sweep.csv")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    max_tokens_list = [int(x.strip()) for x in args.max_tokens_list.split(",") if x.strip()]

    if os.path.exists(args.out):
        os.remove(args.out)

    for mt in max_tokens_list:
        cmd = [
            "python",
            "bench_experiments.py",
            "--base-url", args.base_url,
            "--model", args.model,
            "--max-tokens", str(mt),
            "--concurrency", args.concurrency,
            "--requests-per-worker", str(args.requests_per_worker),
            "--prompt-idx", str(args.prompt_idx),
            "--run-label", "output_length_sweep",
            "--out", args.out,
            "--append",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"\nFinished output-length sweep. Results saved to: {args.out}")


if __name__ == "__main__":
    main()
