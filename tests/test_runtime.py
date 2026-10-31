import unittest
from unittest.mock import patch
import runtime


class RuntimeTests(unittest.TestCase):
    def test_rerank_maps_ids_and_order(self):
        with patch(
            "runtime.call",
            return_value={
                "id": "req-1",
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ],
            },
        ):
            result = runtime.execute(
                "rerank",
                {
                    "query": "q",
                    "documents": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}],
                },
                "key",
            )
        self.assertEqual([r["id"] for r in result["results"]], ["b", "a"])
        self.assertEqual(result["request_id"], "req-1")

    def test_embed_counts_and_dimensions(self):
        with patch(
            "runtime.call",
            return_value={
                "id": "r",
                "embeddings": {"float": [[1.0] * 1536, [3.0] * 1536]},
            },
        ):
            result = runtime.execute(
                "embed", {"texts": ["a", "b"], "input_type": "search_document"}, "key"
            )
        self.assertEqual(result["dimensions"], 1536)
        self.assertEqual(len(result["vectors"]), 2)

    def test_invalid_inputs(self):
        for op, args in [
            ("embed", {"texts": [], "input_type": "search_document"}),
            (
                "rerank",
                {
                    "query": "q",
                    "documents": [{"id": "x", "text": "a"}, {"id": "x", "text": "b"}],
                },
            ),
        ]:
            with self.assertRaises(runtime.Failure):
                runtime.execute(op, args, "key")

    def test_credentials(self):
        with self.assertRaises(runtime.Failure) as ctx:
            runtime.execute("embed", {"texts": ["a"], "input_type": "search_query"}, "")
        self.assertEqual(ctx.exception.code, "not_configured")

    def test_invalid_response(self):
        with patch("runtime.call", return_value={"embeddings": {"float": [[1.0]]}}):
            with self.assertRaises(runtime.Failure):
                runtime.execute(
                    "embed", {"texts": ["a", "b"], "input_type": "search_query"}, "key"
                )


class FailureTests(unittest.TestCase):
    def test_batch_and_dimension_validation(self):
        cases = [
            ("embed", {"texts": ["x"] * 97, "input_type": "search_document"}),
            ("embed", {"texts": ["x"], "input_type": "image"}),
            (
                "embed",
                {
                    "texts": ["x"],
                    "input_type": "search_document",
                    "model": "embed-v4.0",
                    "output_dimension": 768,
                },
            ),
            (
                "rerank",
                {"query": "x", "documents": [{"id": "a", "text": "x"}], "top_n": 2},
            ),
        ]
        for op, args in cases:
            with (
                self.subTest(op=op, args=args),
                self.assertRaises(runtime.Failure) as exc,
            ):
                runtime.execute(op, args, "key")
            self.assertEqual(exc.exception.code, "invalid_payload")

    def test_timeout_and_http_error_do_not_leak_secret(self):
        import sys
        import types
        from urllib.error import HTTPError

        for error, expected in [
            (TimeoutError(), "timeout"),
            (
                HTTPError("https://api.cohere.com", 401, "bad", {}, None),
                "authentication_failed",
            ),
        ]:
            fake = types.SimpleNamespace(
                open_pinned_url=lambda *args, **kwargs: (_ for _ in ()).throw(error)
            )
            with (
                patch.dict(sys.modules, {"flowsteward_extension_sdk": fake}),
                self.assertRaises(runtime.Failure) as exc,
            ):
                runtime.call("embed", {}, "secret-value")
            self.assertEqual(exc.exception.code, expected)
            self.assertNotIn("secret-value", str(exc.exception))


class HostTests(unittest.TestCase):
    def test_selected_connection_key_and_invalid_scope(self):
        import main

        payload = {
            "contract_version": "extension_host_v2",
            "mode": "action",
            "runtime_context": {
                "group_id": None,
                "for_each_id": None,
                "step_id": "embed",
                "step_execution_id": "123",
            },
            "action": {
                "action_id": "embed",
                "input": {"texts": ["a"], "input_type": "search_query"},
                "target": {"connection": {"secrets": {"api_key": "selected"}}},
            },
        }
        with patch(
            "runtime.call",
            return_value={"id": "r", "embeddings": {"float": [[1.0] * 1536]}},
        ) as upstream:
            self.assertTrue(main.handle(payload)["ok"])
            self.assertEqual(upstream.call_args.args[2], "selected")
            payload["runtime_context"] = {}
            self.assertFalse(main.handle(payload)["ok"])
            self.assertEqual(upstream.call_count, 1)


class DimensionTests(unittest.TestCase):
    def test_dimension_select_is_converted_to_integer(self):
        with patch(
            "runtime.call", return_value={"embeddings": {"float": [[0.0] * 256]}}
        ) as upstream:
            result = runtime.execute(
                "embed",
                {
                    "texts": ["a"],
                    "input_type": "search_document",
                    "output_dimension": "256",
                },
                "key",
            )
        assert result["dimensions"] == 256
        assert upstream.call_args.args[1]["output_dimension"] == 256


class ReviewRegressionTests(unittest.TestCase):
    def test_default_dimensions_reject_wrong_width_for_each_model(self):
        for model in ("embed-v4.0", "embed-v5.0-fast"):
            with (
                self.subTest(model=model),
                patch(
                    "runtime.call",
                    return_value={"embeddings": {"float": [[0.0] * 256]}},
                ),
                self.assertRaises(runtime.Failure),
            ):
                runtime.execute(
                    "embed",
                    {"texts": ["x"], "input_type": "search_query", "model": model},
                    "key",
                )

    def test_output_budget_rejected_before_call(self):
        with patch("runtime.call") as call, self.assertRaises(runtime.Failure):
            runtime.execute(
                "embed",
                {
                    "texts": ["x"] * 96,
                    "input_type": "search_query",
                    "model": "embed-v5.0-fast",
                },
                "key",
            )
        call.assert_not_called()

    def test_large_id_rejected_before_call(self):
        with patch("runtime.call") as call, self.assertRaises(runtime.Failure):
            runtime.execute(
                "rerank",
                {"query": "q", "documents": [{"id": "x" * 1025, "text": "x"}]},
                "key",
            )
        call.assert_not_called()

    def test_wrapped_timeout(self):
        import sys
        import types
        from urllib.error import URLError

        fake = types.SimpleNamespace(
            open_pinned_url=lambda *a, **kw: (_ for _ in ()).throw(
                URLError(TimeoutError())
            )
        )
        with (
            patch.dict(sys.modules, {"flowsteward_extension_sdk": fake}),
            self.assertRaises(runtime.Failure) as exc,
        ):
            runtime.call("embed", {}, "key")
        self.assertEqual(exc.exception.code, "timeout")


class OutputBoundaryTests(unittest.TestCase):
    def test_maximum_budget_in_host_envelope(self):
        import json
        import main

        payload = {
            "contract_version": "extension_host_v2",
            "mode": "action",
            "runtime_context": {
                "group_id": None,
                "for_each_id": None,
                "step_id": "embed",
                "step_execution_id": "123",
            },
            "action": {
                "action_id": "embed",
                "input": {
                    "texts": ["x"] * 48,
                    "input_type": "search_query",
                    "model": "embed-v5.0-pro",
                },
                "target": {"connection": {"secrets": {"api_key": "selected"}}},
            },
        }
        with patch(
            "runtime.call",
            return_value={
                "id": "r",
                "embeddings": {
                    "float": [[0.12345678901234568] * 2048 for _ in range(48)]
                },
            },
        ):
            result = main.handle(payload)
        self.assertTrue(result["ok"])
        self.assertLess(len(json.dumps(result).encode()), 5 * 1024 * 1024)


class CommonDimensionTests(unittest.TestCase):
    def test_explicit_v5_only_dimensions_are_not_accepted(self):
        for dim in (768, 2048):
            with (
                self.subTest(dim=dim),
                patch("runtime.call") as call,
                self.assertRaises(runtime.Failure),
            ):
                runtime.execute(
                    "embed",
                    {
                        "texts": ["x"],
                        "input_type": "search_query",
                        "model": "embed-v5.0-pro",
                        "output_dimension": str(dim),
                    },
                    "key",
                )
            call.assert_not_called()
