import os
import time
from typing import Optional

import requests


def _raise_for_status(response: requests.Response):
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text.strip()
        if len(detail) > 1000:
            detail = f"{detail[:1000]}..."
        message = f"{exc}. Response body: {detail or '<empty>'}"
        raise requests.HTTPError(message, response=response) from exc


def _headers(api_key: Optional[str]) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def call_once(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_type: str = "completions",
    api_key: Optional[str] = None,
):
    api_key = api_key or os.getenv("TOGETHER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        api_key = api_key.strip()
    root = base_url.rstrip("/")
    t0 = time.time()

    if api_type == "chat":
        endpoint = f"{root}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    else:
        endpoint = f"{root}/v1/completions"
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }

    r = requests.post(
        endpoint,
        headers=_headers(api_key),
        json=payload,
        timeout=600,
    )
    _raise_for_status(r)
    t1 = time.time()

    data = r.json()
    usage = data.get("usage", {})
    out_tokens = usage.get("completion_tokens", 0) or 0
    return (t1 - t0), out_tokens
