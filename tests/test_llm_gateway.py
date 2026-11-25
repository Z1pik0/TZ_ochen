import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx2
from openai import APIStatusError, OpenAI

from llm_gateway import ModelClient, complete_json, gemini_router, llm_client


class GatewayTests(unittest.TestCase):
    def test_retry_is_bounded_and_skips_permanent_errors(self):
        for status, headers, expected_attempts in ((503, {}, 2), (401, {}, 1), (429, {"Retry-After": "60"}, 1)):
            with self.subTest(status=status):
                attempts = []

                def reject(request):
                    attempts.append(request)
                    return httpx2.Response(status, headers=headers)

                transport = httpx2.Client(transport=httpx2.MockTransport(reject))
                with (
                    OpenAI(api_key="test", base_url="https://provider.invalid/v1", http_client=transport, max_retries=0) as api,
                    patch("llm_gateway.sleep") as wait,
                ):
                    with self.assertRaises(APIStatusError) as error:
                        complete_json(ModelClient(api, "demo-model"), [], {"type": "json_object"})
                self.assertEqual(error.exception.status_code, status)
                self.assertEqual(len(attempts), expected_attempts)
                self.assertEqual(wait.call_count, expected_attempts - 1)

    def test_settings_supply_endpoint_and_model_and_omit_empty_reasoning(self):
        requests = []

        def completion(request):
            requests.append(request)
            return httpx2.Response(200, json={
                "id": "test", "object": "chat.completion", "created": 0, "model": "my-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}],
            })

        with TemporaryDirectory() as directory, patch.dict(os.environ, clear=True):
            env_file = Path(directory) / ".env"
            env_file.write_text("LLM_BASE_URL=http://127.0.0.1:9999/v1\nLLM_API_KEY=test-key\nLLM_MODEL=my-model\n", encoding="utf-8")
            transport = httpx2.Client(transport=httpx2.MockTransport(completion))
            with (
                patch("llm_gateway.LOCAL_ENV", env_file),
                patch("llm_gateway.DefaultHttpxClient", return_value=transport),
                llm_client() as client,
            ):
                self.assertEqual(complete_json(client, [], {"type": "json_object"}), "{}")

        request, = requests
        self.assertEqual(str(request.url), "http://127.0.0.1:9999/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer test-key")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "my-model")
        self.assertNotIn("reasoning_effort", body)

    def test_litellm_uses_numbered_keys_and_preserves_high_thinking_on_fallback(self):
        import httpx

        attempts = []
        primary, fallback = "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"
        keys = [f"test-key-{index}" for index in range(1, 5)]
        settings = {
            **{f"LITELLM_API_KEY_{index}": key for index, key in enumerate(keys, start=1)},
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LLM_MODEL": "ignored-direct-model",
            "LLM_REASONING_EFFORT": "high",
        }

        def respond(request):
            model = request.url.path.rsplit("/", 1)[-1].removesuffix(":generateContent")
            payload = json.loads(request.content)
            attempts.append((model, request.headers.get("x-goog-api-key"), payload["generationConfig"]))
            if model == primary:
                return httpx.Response(503, request=request, json={
                    "error": {"code": 503, "status": "UNAVAILABLE", "message": "Test overload"},
                })
            return httpx.Response(200, request=request, json={
                "candidates": [{
                    "content": {"role": "model", "parts": [{"text": "{\"ok\": true}"}]},
                    "index": 0,
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 6, "totalTokenCount": 14},
            })

        def send(_client, request, **_):
            return respond(request)

        async def async_send(_client, request, **_):
            return respond(request)

        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "Check",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        }
        self.addCleanup(gemini_router.cache_clear)
        with (
            TemporaryDirectory() as directory,
            patch("llm_gateway.LOCAL_ENV", Path(directory) / ".env"),
            patch.dict(os.environ, settings, clear=True),
            patch.object(httpx.Client, "send", send),
            patch.object(httpx.AsyncClient, "send", async_send),
            llm_client() as client,
        ):
            reply = complete_json(client, [{"role": "user", "content": "Return JSON"}], schema)

        self.assertEqual(json.loads(reply), {"ok": True})
        models = [model for model, _, _ in attempts]
        self.assertEqual(models, [primary, primary, primary, fallback])
        for _, key, config in attempts:
            self.assertIn(key, keys)
            self.assertEqual(config["thinkingConfig"]["thinkingLevel"].lower(), "high")
            self.assertEqual(config["response_mime_type"], "application/json")
