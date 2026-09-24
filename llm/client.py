"""Unified LLM access for every agent.

Order of attempts for each call:
  1. Groq (primary model, then secondary model) if GROQ_API_KEY is set.
  2. Gemini (free tier) if GOOGLE_API_KEY is set.
  3. Offline mock mode: a small deterministic template engine, so the whole
     orchestrator loop still runs (and is demoable) with zero network access.

Every call goes through retry_call(), which retries transient failures with
a short backoff before moving to the next provider. This is what keeps the
system inside free-tier rate limits and alive during a flaky connection.
"""
import json
import time
import random

import requests

import config


class LLMError(Exception):
    pass


def _post_json(url, headers, payload):
    resp = requests.post(
        url, headers=headers, json=payload, timeout=config.REQUEST_TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    return resp.json()


def _call_groq(prompt: str, system: str, model: str) -> str:
    if not config.GROQ_API_KEY:
        raise LLMError("no Groq key configured")
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    data = _post_json(config.GROQ_ENDPOINT, headers, payload)
    return data["choices"][0]["message"]["content"]


def _call_gemini(prompt: str, system: str) -> str:
    if not config.GOOGLE_API_KEY:
        raise LLMError("no Gemini key configured")
    url = f"{config.GEMINI_ENDPOINT}?key={config.GOOGLE_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": f"{system}\n\n{prompt}"}]}],
        "generationConfig": {"temperature": 0.3},
    }
    data = _post_json(url, {"Content-Type": "application/json"}, payload)
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _mock_response(prompt: str, system: str) -> str:
    """Deterministic offline fallback so the loop always completes.

    Not meant to be smart — it exists so a demo never dies just because no
    API key is present. It looks at a couple of keywords in the prompt to
    return something structurally usable (valid JSON when JSON was asked
    for) and otherwise returns a short templated sentence.
    """
    random.seed(len(prompt))
    if "respond only in json" in system.lower() or "respond only in json" in prompt.lower():
        # Best-effort generic structured stub; callers should treat this as
        # a last-resort filler, not a source of truth.
        return json.dumps(
            {
                "approach": "gradient_boosted_trees",
                "reasoning": "offline mock: no LLM key configured, using a safe generic default",
                "confidence": 0.5,
            }
        )
    return (
        "[offline mock response] No LLM API key is configured, so this is a "
        "placeholder. Add GROQ_API_KEY or GOOGLE_API_KEY to .env for real "
        "reasoning."
    )


def retry_call(fn, attempts=config.RETRY_ATTEMPTS, backoff=config.RETRY_BACKOFF_SECONDS):
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - deliberately broad, we retry any failure
            last_err = e
            if i < attempts - 1:
                time.sleep(backoff * (i + 1))
    raise last_err


def complete(prompt: str, system: str = "You are a careful, concise assistant.") -> dict:
    """Run a prompt through the provider chain.

    Returns a dict: {"text": str, "provider": str} so callers/UIs can show
    which provider actually answered (useful for the dashboard's
    transparency requirement).
    """
    providers = []
    if config.GROQ_API_KEY:
        providers.append(("groq-primary", lambda: _call_groq(prompt, system, config.GROQ_MODEL_PRIMARY)))
        providers.append(("groq-secondary", lambda: _call_groq(prompt, system, config.GROQ_MODEL_SECONDARY)))
    if config.GOOGLE_API_KEY:
        providers.append(("gemini", lambda: _call_gemini(prompt, system)))

    for name, fn in providers:
        try:
            text = retry_call(fn)
            return {"text": text, "provider": name}
        except Exception:
            continue

    return {"text": _mock_response(prompt, system), "provider": "offline-mock"}


def complete_json(prompt: str, system: str = "") -> dict:
    """Convenience wrapper: asks for JSON and parses it, with a safe fallback."""
    full_system = (
        system
        + "\nRespond only in JSON. No preamble, no markdown fences, no extra text."
    )
    result = complete(prompt, full_system)
    text = result["text"].strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"raw": text, "parse_error": True}
    parsed["_provider"] = result["provider"]
    return parsed
