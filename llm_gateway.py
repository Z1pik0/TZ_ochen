from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import sleep

import httpx2
from openai import APIConnectionError, APIStatusError, DefaultHttpxClient, OpenAI
from starlette.config import Config


LOCAL_ENV = Path(__file__).resolve().with_name(".env")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
RETRY_DELAY = 2.0
MAX_RETRY_DELAY = 5.0


class LLMConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ModelClient:
    api: OpenAI
    model: str
    reasoning_effort: str = ""


@contextmanager
def llm_client() -> Iterator[ModelClient]:
    settings = Config(LOCAL_ENV if LOCAL_ENV.is_file() else None, encoding="utf-8-sig")
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
    effort = settings("LLM_REASONING_EFFORT", default="").strip()
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
    delay = None
    try:
        completion = client.api.chat.completions.create(**payload)
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
    if delay is not None:
        sleep(delay)
        completion = client.api.chat.completions.create(**payload)
    if not getattr(completion, "choices", None):
        raise ValueError("The model returned no choices")
    choice = completion.choices[0]
    content = getattr(getattr(choice, "message", None), "content", None)
    if getattr(choice, "finish_reason", None) != "stop" or not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned an incomplete response")
    return content


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
