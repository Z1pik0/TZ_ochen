from __future__ import annotations

import json
from base64 import b64encode
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
from xml.sax.saxutils import quoteattr

import httpx2
from openai import APIConnectionError, APIStatusError, DefaultHttpxClient, OpenAI
from resvg_py import svg_to_bytes
from starlette.config import Config

if TYPE_CHECKING:
    from litellm import Router


LOCAL_ENV = Path(__file__).resolve().with_name(".env")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_FALLBACK = "gemini-3.1-flash-lite"
GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_IMAGE_LIMIT = 3
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
    groq_fallback: bool = False


@lru_cache(maxsize=1)
def provider_router(
    gemini_keys: tuple[str, ...], model: str, fallback: str, groq_keys: tuple[str, ...],
) -> Router:
    from litellm import Router

    gemini_models = (model, fallback) if fallback and fallback != model else (model,)
    deployments = [
        {
            "model_name": name,
            "litellm_params": {"model": f"gemini/{name}", "api_key": key},
        }
        for name in gemini_models
        for key in gemini_keys
    ]
    deployments.extend(
        {
            "model_name": GROQ_MODEL,
            "litellm_params": {"model": f"groq/{GROQ_MODEL}", "api_key": key},
        }
        for key in groq_keys
    )
    return Router(
        model_list=deployments,
        num_retries=2,
        max_fallbacks=1,
        fallbacks=[{model: [fallback]}] if gemini_keys and len(gemini_models) == 2 else [],
        timeout=45,
    )


def read_groq_keys(value: str) -> tuple[str, ...]:
    value = value.strip()
    if not value:
        return ()
    if not value.startswith("["):
        return (value,)
    problem = "GROQ_API_KEY должен содержать ключ или JSON-массив ключей Groq."
    try:
        keys = json.loads(value)
    except json.JSONDecodeError as error:
        raise LLMConfigurationError(problem) from error
    if not isinstance(keys, list) or any(not isinstance(key, str) or not key.strip() for key in keys):
        raise LLMConfigurationError(problem)
    return tuple(dict.fromkeys(key.strip() for key in keys))


@contextmanager
def llm_client() -> Iterator[ModelClient]:
    settings = Config(LOCAL_ENV if LOCAL_ENV.is_file() else None, encoding="utf-8-sig")
    effort = settings("LLM_REASONING_EFFORT", default="").strip()
    gemini_keys = tuple(dict.fromkeys(
        key for i in range(1, 5)
        if (key := settings(f"LITELLM_API_KEY_{i}", default="").strip())
    ))
    groq_keys = read_groq_keys(settings("GROQ_API_KEY", default=""))
    if gemini_keys or groq_keys:
        model = settings("LITELLM_MODEL", default=DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
        fallback = settings("LITELLM_FALLBACK_MODEL", default=DEFAULT_GEMINI_FALLBACK).strip()
        if gemini_keys and any(name and not name.startswith("gemini-") for name in (model, fallback)):
            raise LLMConfigurationError("Для встроенного LiteLLM укажите ID моделей Gemini.")
        with router_lock:
            api = provider_router(gemini_keys, model, fallback, groq_keys)
        yield ModelClient(api, model if gemini_keys else GROQ_MODEL, effort, bool(groq_keys and gemini_keys))
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
        completion = router_completion(api, payload, client.groq_fallback)
    if not getattr(completion, "choices", None):
        raise ValueError("The model returned no choices")
    choice = completion.choices[0]
    content = getattr(getattr(choice, "message", None), "content", None)
    if getattr(choice, "finish_reason", None) != "stop" or not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned an incomplete response")
    return content


def router_completion(api: Router, payload: dict, groq_fallback: bool):
    if payload["model"] != GROQ_MODEL:
        try:
            return api.completion(**payload)
        except APIStatusError as error:
            if not groq_fallback or error.status_code not in RETRYABLE_STATUSES:
                raise
        except APIConnectionError:
            if not groq_fallback:
                raise
    return api.completion(**{
        **payload,
        "model": GROQ_MODEL,
        "messages": groq_messages(payload["messages"]),
        "extra_body": {"reasoning_format": "hidden"},
    })


def groq_messages(messages: list[dict]) -> list[dict]:
    return [
        {**message, "content": groq_image_content(message["content"])}
        if isinstance(message.get("content"), list) else message
        for message in messages
    ]


def groq_image_content(parts: list[dict]) -> list[dict]:
    urls = [part["image_url"]["url"] for part in parts if part["type"] == "image_url"]
    if len(urls) <= GROQ_IMAGE_LIMIT:
        return parts
    # Placement labels already occur in the same order as their previews.
    columns = 3
    tile = 336
    width = columns * tile
    height = ((len(urls) + columns - 1) // columns) * tile
    cells = []
    for index, url in enumerate(urls):
        row, column = divmod(index, columns)
        x, y = column * tile + 8, row * tile + 8
        cells.append(f'<image x="{x}" y="{y}" width="320" height="320" href={quoteattr(url)}/>')
    sheet = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="white"/>{"".join(cells)}</svg>'
    )
    png = b64encode(svg_to_bytes(svg_string=sheet)).decode("ascii")
    return [
        *(part for part in parts if part["type"] != "image_url"),
        {
            "type": "text",
            "text": "Превью собраны в сетку по три в строке. Подписи выше идут слева направо, сверху вниз.",
        },
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
    ]


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
