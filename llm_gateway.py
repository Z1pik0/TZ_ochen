from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import sleep
from typing import TYPE_CHECKING

import httpx2
from openai import APIConnectionError, APIStatusError, DefaultHttpxClient, OpenAI
from starlette.config import Config

if TYPE_CHECKING:
    from litellm import Router


LOCAL_ENV = Path(__file__).resolve().with_name(".env")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_FALLBACK = "gemini-3.1-flash-lite"
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
RETRY_DELAY = 2.0
MAX_RETRY_DELAY = 5.0
router_lock = Lock()


class LLMConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ModelClient:
    api: OpenAI | Router
    model: str
    reasoning_effort: str = ""


@lru_cache(maxsize=1)
def gemini_router(keys: tuple[str, ...], model: str, fallback: str) -> Router:
    from litellm import Router

    model_names = (model, fallback) if fallback and fallback != model else (model,)
    deployments = [
        {
            "model_name": name,
            "litellm_params": {"model": f"gemini/{name}", "api_key": key},
        }
        for name in model_names
        for key in keys
    ]
    return Router(
        model_list=deployments,
        num_retries=2,
        max_fallbacks=1,
        fallbacks=[{model: [fallback]}] if len(model_names) == 2 else [],
        timeout=45,
    )


@contextmanager
def llm_client() -> Iterator[ModelClient]:
    settings = Config(LOCAL_ENV if LOCAL_ENV.is_file() else None, encoding="utf-8-sig")
    effort = settings("LLM_REASONING_EFFORT", default="").strip()
    keys = tuple(dict.fromkeys(
        key for i in range(1, 5)
        if (key := settings(f"LITELLM_API_KEY_{i}", default="").strip())
    ))
    if keys:
        model = settings("LITELLM_MODEL", default=DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
        fallback = settings("LITELLM_FALLBACK_MODEL", default=DEFAULT_GEMINI_FALLBACK).strip()
        if any(name and not name.startswith("gemini-") for name in (model, fallback)):
            raise LLMConfigurationError("Для встроенного LiteLLM укажите ID моделей Gemini.")
        with router_lock:
            api = gemini_router(keys, model, fallback)
        yield ModelClient(api, model, effort)
        return

    api_key = settings("LLM_API_KEY", default="").strip()
    model = settings("LLM_MODEL", default="").strip()
    if not api_key or not model:
        raise LLMConfigurationError("Для генерации укажите LLM_API_KEY и LLM_MODEL в .env.")
    url = settings("LLM_BASE_URL", default=DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    try:
        base_url = httpx2.URL(url)
    except httpx2.InvalidURL as error:
        raise LLMConfigurationError("Укажите корректный LLM_BASE_URL в .env.") from error
    if base_url.scheme not in {"http", "https"} or not base_url.host:
        raise LLMConfigurationError("Укажите корректный LLM_BASE_URL в .env.")
    loopback = base_url.host in {"127.0.0.1", "localhost", "::1"}
    with OpenAI(
        api_key=api_key, base_url=str(base_url), timeout=90, max_retries=0,
        http_client=DefaultHttpxClient(trust_env=False) if loopback else None,
    ) as api:
        yield ModelClient(api=api, model=model, reasoning_effort=effort)


def complete_json(client: ModelClient, messages: list[dict], response_format: dict) -> str:
    payload = {
        "model": client.model,
        "messages": messages,
        "response_format": response_format,
    }
    if client.reasoning_effort:
        payload["reasoning_effort"] = client.reasoning_effort
    api = client.api
    if isinstance(api, OpenAI):
        completion = openai_completion(api, payload)
    else:
        completion = api.completion(**payload)
    if not getattr(completion, "choices", None):
        raise ValueError("The model returned no choices")
    choice = completion.choices[0]
    content = getattr(getattr(choice, "message", None), "content", None)
    if getattr(choice, "finish_reason", None) != "stop" or not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned an incomplete response")
    return content


def openai_completion(api: OpenAI, payload: dict):
    try:
        return api.chat.completions.create(**payload)
    except APIConnectionError as error:
        if not isinstance(error.__cause__, (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.RemoteProtocolError)):
            raise
        delay = RETRY_DELAY
    except APIStatusError as error:
        if error.status_code not in RETRYABLE_STATUSES:
            raise
        delay = retry_delay(error.response.headers.get("Retry-After"))
        if delay > MAX_RETRY_DELAY:
            raise
    sleep(delay)
    return api.chat.completions.create(**payload)


def retry_delay(retry_after: str | None) -> float:
    if not retry_after:
        return RETRY_DELAY
    try:
        delay = int(retry_after)
        return max(RETRY_DELAY, delay)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, OverflowError):
            return RETRY_DELAY
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(RETRY_DELAY, (retry_at - datetime.now(UTC)).total_seconds())
