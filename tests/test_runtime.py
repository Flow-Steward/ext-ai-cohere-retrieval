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
            return_value={"id": "r", "embeddings": {"float": [[1.0, 2.0], [3.0, 4.0]]}},
        ):
            result = runtime.execute(
                "embed", {"texts": ["a", "b"], "input_type": "search_document"}, "key"
            )
        self.assertEqual(result["dimensions"], 2)
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
