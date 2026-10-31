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
                "external_effect_id": "claim-1",
            },
            "action": {
                "action_id": "embed",
                "input": {
                    "connection_ref": "selected-connection",
                    "texts": ["a"],
                    "input_type": "search_query",
                },
                "target": {
                    "connection": {
                        "connection_id": "selected-connection",
                        "connection_type": "cohere_api",
                        "secrets": {"api_key": "selected"},
                        "config": {"request_timeout_seconds": 10},
                    }
                },
            },
        }
        with patch(
            "runtime.call",
            return_value={"id": "r", "embeddings": {"float": [[1.0] * 1536]}},
        ) as upstream:
            self.assertTrue(main.handle(payload)["ok"])
            self.assertEqual(upstream.call_args.args[2], "selected")
            self.assertEqual(upstream.call_args.kwargs["timeout_seconds"], 10)
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
                "external_effect_id": "claim-1",
            },
            "action": {
                "action_id": "embed",
                "input": {
                    "connection_ref": "selected-connection",
                    "texts": ["x"] * 4,
                    "input_type": "search_query",
                    "model": "embed-v5.0-pro",
                },
                "target": {
                    "connection": {
                        "connection_id": "selected-connection",
                        "connection_type": "cohere_api",
                        "secrets": {"api_key": "selected"},
                    }
                },
            },
        }
        with patch(
            "runtime.call",
            return_value={
                "id": "r",
                "embeddings": {
                    "float": [[0.12345678901234568] * 2048 for _ in range(4)]
                },
            },
        ):
            result = main.handle(payload)
        self.assertTrue(result["ok"])
        self.assertLess(len(json.dumps(result).encode()), 512 * 1024)


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


class ConnectionTimeoutTests(unittest.TestCase):
    def test_invalid_timeout_is_rejected_before_network(self):
        with patch("runtime.call") as call, self.assertRaises(runtime.Failure):
            runtime.execute(
                "embed",
                {"texts": ["x"], "input_type": "search_query"},
                "key",
                {"request_timeout_seconds": 61},
            )
        call.assert_not_called()

    def test_selected_timeout_is_passed_to_request(self):
        with patch(
            "runtime.call", return_value={"embeddings": {"float": [[0.0] * 1536]}}
        ) as call:
            runtime.execute(
                "embed",
                {"texts": ["x"], "input_type": "search_query"},
                "key",
                {"request_timeout_seconds": 10},
            )
        self.assertEqual(call.call_args.kwargs["timeout_seconds"], 10)


class EffectTests(unittest.TestCase):
    def payload(self):
        return {
            "contract_version": "extension_host_v2",
            "mode": "action",
            "runtime_context": {
                "group_id": None,
                "for_each_id": None,
                "step_id": "embed",
                "step_execution_id": "123",
                "external_effect_id": "claim-1",
            },
            "action": {
                "action_id": "embed",
                "input": {
                    "connection_ref": "conn-1",
                    "texts": ["x"],
                    "input_type": "search_query",
                },
                "target": {
                    "connection": {
                        "connection_id": "conn-1",
                        "connection_type": "cohere_api",
                        "secrets": {"api_key": "key"},
                    }
                },
            },
        }

    def test_claim_test_mode_and_connection_checked_before_upstream(self):
        import main

        for change in ("claim", "test_mode", "connection_id", "connection_type"):
            payload = self.payload()
            if change == "claim":
                payload["runtime_context"].pop("external_effect_id")
            if change == "test_mode":
                payload["runtime_context"]["test_mode"] = True
            if change == "connection_id":
                payload["action"]["target"]["connection"]["connection_id"] = "other"
            if change == "connection_type":
                payload["action"]["target"]["connection"]["connection_type"] = "other"
            with self.subTest(change=change), patch("runtime.call") as call:
                self.assertFalse(main.handle(payload)["ok"])
                call.assert_not_called()

    def test_success_carries_provider_id(self):
        import main

        with patch(
            "runtime.call",
            return_value={"id": "provider-1", "embeddings": {"float": [[0.0] * 1536]}},
        ):
            self.assertEqual(
                main.handle(self.payload())["provider_message_id"], "provider-1"
            )

    def test_replay_budget_rejected_before_paid_request(self):
        with patch("runtime.call") as call, self.assertRaises(runtime.Failure):
            runtime.execute(
                "embed",
                {
                    "texts": ["x"] * 10,
                    "input_type": "search_query",
                    "model": "embed-v5.0-pro",
                },
                "key",
            )
        call.assert_not_called()


class CorrelationTests(unittest.TestCase):
    def test_http_request_id_and_no_automatic_retry(self):
        import sys
        import types
        from urllib.error import HTTPError

        error = HTTPError(
            "https://api.cohere.com",
            429,
            "limited",
            {"x-request-id": "cohere-request-1"},
            None,
        )
        fake = types.SimpleNamespace(
            open_pinned_url=lambda *a, **kw: (_ for _ in ()).throw(error)
        )
        with (
            patch.dict(sys.modules, {"flowsteward_extension_sdk": fake}),
            self.assertRaises(runtime.Failure) as exc,
        ):
            runtime.call("embed", {}, "key")
        self.assertEqual(exc.exception.request_id, "cohere-request-1")
        self.assertEqual(exc.exception.code, "rate_limited")

    def test_ambiguous_failure_is_not_safe_to_repeat(self):
        import main

        payload = EffectTests().payload()
        with patch("runtime.call", side_effect=runtime.Failure("timeout", "uncertain")):
            response = main.handle(payload)
        self.assertFalse(response["retryable"])
        self.assertFalse(response["definitely_no_external_effect"])


class PostResponseCorrelationTests(unittest.TestCase):
    def test_invalid_embedding_response_preserves_provider_id(self):
        import main

        with patch(
            "runtime.call",
            return_value={"id": "provider-received", "embeddings": {"float": [[0.0]]}},
        ):
            response = main.handle(EffectTests().payload())
        self.assertFalse(response["ok"])
        self.assertEqual(response["provider_message_id"], "provider-received")
        self.assertFalse(response["definitely_no_external_effect"])

    def test_invalid_rerank_response_preserves_provider_id(self):
        import main

        payload = EffectTests().payload()
        payload["action"]["action_id"] = "rerank"
        payload["action"]["input"] = {
            "connection_ref": "conn-1",
            "query": "q",
            "documents": [{"id": "a", "text": "x"}],
        }
        with patch(
            "runtime.call",
            return_value={
                "id": "provider-rerank",
                "results": [{"index": 5, "relevance_score": 0.2}],
            },
        ):
            response = main.handle(payload)
        self.assertFalse(response["ok"])
        self.assertEqual(response["provider_message_id"], "provider-rerank")
        self.assertFalse(response["definitely_no_external_effect"])


class ResponseHeaderTests(unittest.TestCase):
    def test_malformed_body_retains_response_header_request_id(self):
        import sys
        import types

        class Response:
            headers = {"x-request-id": "header-id"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                return b"bad json"

        with (
            patch.dict(
                sys.modules,
                {
                    "flowsteward_extension_sdk": types.SimpleNamespace(
                        open_pinned_url=lambda *a, **kw: Response()
                    )
                },
            ),
            self.assertRaises(runtime.Failure) as exc,
        ):
            runtime.call("embed", {}, "key")
        self.assertEqual(exc.exception.request_id, "header-id")


class ResponseHeaderSuccessTests(unittest.TestCase):
    def test_response_header_id_used_when_body_omits_id(self):
        import sys
        import types

        class Response:
            headers = {"x-request-id": "header-success"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                return b'{"results": []}'

        with patch.dict(
            sys.modules,
            {
                "flowsteward_extension_sdk": types.SimpleNamespace(
                    open_pinned_url=lambda *a, **kw: Response()
                )
            },
        ):
            self.assertEqual(runtime.call("rerank", {}, "key")["id"], "header-success")
