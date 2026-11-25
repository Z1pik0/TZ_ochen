import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx2
from openai import APIStatusError, OpenAI

from llm_gateway import ModelClient, complete_json, llm_client


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
