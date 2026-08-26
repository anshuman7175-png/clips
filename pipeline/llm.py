"""Minimal OpenAI-compatible chat client (works with OpenAI, gateways, Ollama).

Kept dependency-free apart from `requests` so it runs anywhere
(Kaggle, Modal, local) without SDK version drift.
"""

from __future__ import annotations

import json
import re
import time

import requests

from .config import LLMEndpoint


class LLMError(RuntimeError):
    pass


def chat(
    endpoint: LLMEndpoint,
    system: str,
    user: str,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    url = endpoint.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    payload = {
        "model": endpoint.model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as err:  # noqa: BLE001 - retry any transport/API error
            last_err = err
            time.sleep(2**attempt)
    raise LLMError(f"LLM call failed after {max_retries} attempts: {last_err}")


def chat_json(endpoint: LLMEndpoint, system: str, user: str, temperature: float = 0.3):
    """Chat call that must return JSON; tolerates markdown fences."""
    raw = chat(endpoint, system + "\nRespond ONLY with valid JSON.", user, temperature)
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    text = match.group(1) if match else raw
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as err:
        raise LLMError(f"Model did not return valid JSON: {raw[:500]}") from err
