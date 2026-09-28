import argparse
import os
import sys

import requests

from llm_client import call_once


def _redact(value: str) -> str:
    if len(value) <= 8:
        return "<set>"
    return f"{value[:4]}...{value[-4:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-type", choices=["completions", "chat"], default="chat")
    ap.add_argument("--api-key-env", default="TOGETHER_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=4)
    args = ap.parse_args()

    api_key = (os.getenv(args.api_key_env) or "").strip()
    if api_key:
        print(
            f"Using {args.api_key_env}={_redact(api_key)} "
            f"(length={len(api_key)})"
        )
        if len(api_key) < 40:
            print(
                "Warning: this API key is unusually short. Make sure you copied "
                "the full secret value, not a key name or masked preview.",
                file=sys.stderr,
            )
    else:
        print(f"{args.api_key_env} is not set; sending request without auth.")

    try:
        latency_s, output_tokens = call_once(
            args.base_url,
            args.model,
            "Reply with exactly: ok",
            args.max_tokens,
            args.api_type,
            api_key=api_key or None,
        )
    except requests.HTTPError as exc:
        print(f"Endpoint validation failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Endpoint validation passed: "
        f"base_url={args.base_url}, model={args.model}, "
        f"api_type={args.api_type}, latency_s={latency_s:.3f}, "
        f"output_tokens={output_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
