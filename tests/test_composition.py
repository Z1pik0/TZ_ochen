import json
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
from fastapi import HTTPException, Request
from openai import OpenAI
from pydantic import ValidationError

from asset_catalog import Anchor, Role, Size
from composition import CompositionRequest, LayerPlan, Position, fit_canvas, generate_composition, variants_for_size
from llm_gateway import ModelClient
from main import Asset, Variant, compose


class LayerPlanTests(unittest.TestCase):
    def test_background_and_main_roles_are_retained_in_layer_order(self):
        context = {
            "instance_ids": {"book", "pen"},
            "roles": {"book": {Role.MAIN, Role.BACKGROUND}, "pen": {Role.MAIN}},
        }
        plan = LayerPlan.model_validate_json(
            '{"arrangement":"The pen is next to the book.","layers":[{"id":"book","role":"bg","x":0.4,"y":0.5},'
            '{"id":"pen","role":"main","x":0.8,"y":0.6}]}',
            context=context,
        )
        self.assertEqual([(item.id, item.role) for item in plan.layers], [("book", Role.BACKGROUND), ("pen", Role.MAIN)])

    def test_role_that_is_missing_for_the_selected_size_is_rejected(self):
        with self.assertRaises(ValidationError):
            LayerPlan.model_validate_json(
                '{"arrangement":"A single pen.","layers":[{"id":"pen","role":"bg","x":0.5,"y":0.5}]}',
                context={"instance_ids": {"pen"}, "roles": {"pen": {Role.MAIN}}},
            )

    def test_generation_keeps_pen_on_the_page_as_placed_by_the_model(self):
        book = Variant("L_main", Size.L, Role.MAIN, 1446, 977, (0, 0, 1446, 977), "/book.svg")
        pen = Variant("L_main", Size.L, Role.MAIN, 509, 286, (0, 0, 509, 286), "/pen.svg")
        assets = [
            Asset("book", "Книга", None, {"on_right_page": Anchor(0.65, 0.32)}, [book]),
            Asset("pen", "Ручка", None, {}, [pen]),
        ]
        request = CompositionRequest(prompt="Книга, на правой страничке лежит ручка", size=Size.L, width=1000, height=1000)
        layers = [
            {"id": "book", "role": "main", "x": 0.5, "y": 0.5},
            {"id": "pen", "role": "main", "x": 0.608, "y": 0.412},
        ]
        raw_assets = Path(__file__).resolve().parents[1] / "raw_assets"
        asset_paths = {(Size.L, Role.MAIN, asset.id): raw_assets / "L" / "main" / f"{asset.id}.svg" for asset in assets}
        with (
            patch("composition.llm_client", return_value=nullcontext()),
            patch("composition.complete_json", side_effect=[
                '{"asset_ids":["book","pen"],"missing_objects":[]}',
                json.dumps({"arrangement": "The pen rests on the right page.", "layers": [
                    dict(layer, id=f"l{index}") for index, layer in enumerate(layers, start=1)
                ]}),
            ]) as complete,
        ):
            result = generate_composition(request, variants_for_size(assets, request.size), asset_paths)

        placement_content = complete.call_args.args[1][1]["content"]
        placement_request = json.loads(placement_content[0]["text"])
        previews = [part for part in placement_content if part["type"] == "image_url"]
        self.assertEqual(placement_request["objects"][0]["anchors"], {"on_right_page": {"x": 0.65, "y": 0.32}})
        self.assertEqual(len(previews), 2)
        self.assertTrue(all(part["image_url"]["url"].startswith("data:image/png;base64,") for part in previews))
        self.assertEqual(result.model_dump(mode="json")["layers"], layers)

    def test_fitting_to_canvas_moves_overlapping_layers_together(self):
        book = Position(id="book", role=Role.MAIN, x=0.30, y=0.20)
        pen = Position(id="pen", role=Role.MAIN, x=0.40, y=0.13)
        footprints = {
            "book": {Role.MAIN: {"width": 0.72, "height": 0.49}},
            "pen": {Role.MAIN: {"width": 0.25, "height": 0.14}},
        }
        factor, (fitted_book, fitted_pen) = fit_canvas([book, pen], footprints)
        self.assertAlmostEqual(fitted_book.x - 0.36 * factor, 0.04)
        self.assertAlmostEqual(fitted_book.y - 0.245 * factor, 0.04)
        self.assertAlmostEqual(fitted_pen.x - fitted_book.x, (pen.x - book.x) * factor)
        self.assertAlmostEqual(fitted_pen.y - fitted_book.y, (pen.y - book.y) * factor)

    def test_duplicate_objects_are_placed_and_only_the_failed_step_retries(self):
        book = Asset("book", "Книга", None, {}, [
            Variant("L_main", Size.L, Role.MAIN, 500, 300, (0, 0, 500, 300), "/book.svg"),
        ])
        coin = Asset("coin", "Монета", None, {}, [
            Variant("L_main", Size.L, Role.MAIN, 400, 300, (0, 0, 400, 300), "/coin.svg"),
        ])
        layers = [
            {"id": "l2", "role": "main", "x": 0.18, "y": 0.5},
            {"id": "l1", "role": "main", "x": 0.5, "y": 0.5},
            {"id": "l3", "role": "main", "x": 0.82, "y": 0.5},
        ]
        content = [
            {"asset_ids": ["book", "coin", "coin"], "missing_objects": []},
            None,
            {"arrangement": "Two coins on either side of the book.", "layers": layers},
        ]
        requests = []

        def completion(request):
            requests.append(json.loads(request.content))
            reply = content.pop(0)
            if reply is None:
                return httpx2.Response(503, headers={"Retry-After": "3"})
            return httpx2.Response(200, json={"choices": [{
                "index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(reply)},
            }], "id": "test", "object": "chat.completion", "created": 0, "model": "demo-model"})

        request = CompositionRequest(prompt="Книга и две монеты", size=Size.L, width=800, height=700)
        transport = httpx2.Client(transport=httpx2.MockTransport(completion))
        api = OpenAI(api_key="test", base_url="https://provider.invalid/v1/", http_client=transport, max_retries=0)
        gateway = ModelClient(api, "demo-model", "high")
        with api, patch("composition.llm_client", return_value=nullcontext(gateway)), patch("llm_gateway.sleep") as wait:
            result = generate_composition(request, variants_for_size([book, coin], request.size), {})

        self.assertEqual([body["response_format"]["json_schema"]["name"] for body in requests], [
            "AssetSelection", "LayerPlan", "LayerPlan",
        ])
        self.assertTrue(all(body["model"] == "demo-model" and body["reasoning_effort"] == "high" for body in requests))
        self.assertEqual(requests[1], requests[2])
        placement = json.loads(requests[1]["messages"][1]["content"])
        self.assertEqual([item["asset_id"] for item in placement["instances"]], ["book", "coin", "coin"])
        self.assertEqual(len(placement["objects"]), 2)
        coin_width = next(item["variants"]["main"]["width"] for item in placement["objects"] if item["id"] == "coin")
        self.assertAlmostEqual(coin_width * 800 / 400, (0.55 * 800 * 700 / (500 * 300 + 2 * 400 * 300)) ** 0.5)
        self.assertEqual([layer.id for layer in result.layers], ["coin", "book", "coin"])
        self.assertLess(result.layers[0].x, result.layers[1].x)
        self.assertLess(result.layers[1].x, result.layers[2].x)
        wait.assert_called_once_with(3)

    def test_unmatched_requests_stop_before_layout(self):
        book = Asset("book", "Книга", None, {}, [
            Variant("L_main", Size.L, Role.MAIN, 500, 300, (0, 0, 500, 300), "/book.svg"),
        ])
        cases = [
            ({"asset_ids": ["book"], "missing_objects": ["кот"]}, "«кот»"),
            ({"asset_ids": [], "missing_objects": []}, "нет подходящих объектов"),
        ]
        state = SimpleNamespace(catalog=[book], asset_paths={})
        request = Request({"type": "http", "app": SimpleNamespace(state=state)})
        data = CompositionRequest(prompt="Тестовый запрос", size=Size.L, width=1000, height=1000)
        for reply, message in cases:
            with (
                self.subTest(reply=reply),
                patch("composition.llm_client", return_value=nullcontext()),
                patch("composition.complete_json", return_value=json.dumps(reply)) as complete,
            ):
                with self.assertRaises(HTTPException) as error:
                    compose(request, data)
                self.assertEqual(error.exception.status_code, 422)
                self.assertIn(message, error.exception.detail)
                complete.assert_called_once()
