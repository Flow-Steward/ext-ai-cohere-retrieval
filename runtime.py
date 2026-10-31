"""Bounded Cohere v2 retrieval requests. No retry on uncertain outcomes."""

import json
import math
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request

RERANK_MODELS = {"rerank-v3.5", "rerank-v5.0-pro", "rerank-v5.0-fast"}
EMBED_MODELS = {"embed-v4.0", "embed-v5.0-pro", "embed-v5.0-fast"}
INPUT_TYPES = {"search_document", "search_query", "classification", "clustering"}
DIMENSIONS = {
    "embed-v4.0": {256, 512, 1024, 1536},
    "embed-v5.0-pro": {256, 512, 1024, 1536},
    "embed-v5.0-fast": {256, 512, 1024, 1536},
}


class Failure(Exception):
    def __init__(self, code, message, request_id=""):
        super().__init__(message)
        self.code = code
        self.request_id = request_id


def invalid(field, hint):
    raise Failure("invalid_payload", f"{field}: {hint}")


def call(endpoint, data, key, *, timeout_seconds=60):
    from flowsteward_extension_sdk import open_pinned_url

    encoded = json.dumps(data).encode()
    if len(encoded) > 2 * 1024 * 1024:
        invalid("input", "request exceeds the 2 MiB size limit; reduce the batch.")
    request = Request(
        "https://api.cohere.com/v2/" + endpoint,
        data=encoded,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with open_pinned_url(
            request, timeout_seconds=timeout_seconds, purpose="Cohere retrieval"
        ) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024:
                raise Failure(
                    "provider_unavailable", "Cohere response exceeded the size limit."
                )
            return json.loads(raw)
    except HTTPError as exc:
        code = (
            "authentication_failed"
            if exc.code in (401, 403, 498)
            else "rate_limited"
            if exc.code == 429
            else "provider_unavailable"
        )
        upstream_id = str(exc.headers.get("x-request-id", "")) if exc.headers else ""
        if len(upstream_id) > 256:
            upstream_id = ""
        raise Failure(
            code,
            f"Cohere request failed with HTTP {exc.code}; review credentials, limits, and input.",
            upstream_id,
        ) from None
    except (TimeoutError, socket.timeout):
        raise Failure(
            "timeout", "Cohere request timed out; check its outcome before retrying."
        ) from None
    except URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise Failure(
                "timeout",
                "Cohere request timed out; check its outcome before retrying.",
            ) from None
        raise Failure("provider_unavailable", "Cohere request failed.") from None
    except (ValueError, OSError):
        raise Failure(
            "provider_unavailable",
            "Cohere request failed or returned an invalid response.",
        ) from None


def request_id(response):
    value = response.get("id")
    if value is not None and (not isinstance(value, str) or len(value) > 256):
        raise Failure("provider_unavailable", "Cohere returned an invalid request ID.")
    return value


def execute(operation, args, key, config=None):
    config = config or {}
    timeout = config.get("request_timeout_seconds", 60)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or not 1 <= timeout <= 60
    ):
        invalid(
            "request_timeout_seconds",
            "set the connection timeout to an integer from 1 to 60 seconds.",
        )
    if not key:
        raise Failure(
            "not_configured",
            "Connection API key is missing; add it to the selected Cohere connection.",
        )
    if operation == "rerank":
        model = args.get("model", "rerank-v3.5")
        if model not in RERANK_MODELS:
            invalid("model", "choose a supported rerank model.")
        query, docs = args.get("query"), args.get("documents")
        if not isinstance(query, str) or not query.strip():
            invalid("query", "enter a non-empty query.")
        if not isinstance(docs, list) or not 1 <= len(docs) <= 1000:
            invalid("documents", "provide 1–1000 documents.")
        ids, texts = [], []
        for i, doc in enumerate(docs):
            if (
                not isinstance(doc, dict)
                or not isinstance(doc.get("id"), str)
                or not doc["id"].strip()
                or not isinstance(doc.get("text"), str)
                or not doc["text"].strip()
            ):
                invalid(f"documents[{i}]", "provide non-empty id and text strings.")
            if len(doc["id"]) > 128:
                invalid(f"documents[{i}].id", "use at most 128 characters.")
            ids.append(doc["id"])
            texts.append(doc["text"])
        if len(set(ids)) != len(ids):
            invalid("documents", "each id must be unique.")
        if sum(len(json.dumps(doc_id).encode()) + 100 for doc_id in ids) > 256 * 1024:
            invalid(
                "documents", "IDs exceed the replay output budget; split the batch."
            )
        data = {"model": model, "query": query, "documents": texts}
        top = args.get("top_n")
        if top is not None:
            if (
                isinstance(top, bool)
                or not isinstance(top, int)
                or not 1 <= top <= len(docs)
            ):
                invalid("top_n", "enter an integer between 1 and the document count.")
            data["top_n"] = top
        response = call("rerank", data, key, timeout_seconds=timeout)
        rows = response.get("results") if isinstance(response, dict) else None
        if not isinstance(rows, list) or len(rows) != (top or len(docs)):
            raise Failure(
                "provider_unavailable", "Cohere returned an incomplete rerank response."
            )
        result, seen = [], set()
        for row in rows:
            index = row.get("index") if isinstance(row, dict) else None
            score = row.get("relevance_score") if isinstance(row, dict) else None
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(ids)
                or index in seen
                or isinstance(score, bool)
                or not isinstance(score, (float, int))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise Failure(
                    "provider_unavailable", "Cohere returned invalid rerank results."
                )
            seen.add(index)
            result.append({"id": ids[index], "index": index, "score": score})
        result.sort(key=lambda row: row["score"], reverse=True)
        return {"model": model, "results": result, "request_id": request_id(response)}
    if operation == "embed":
        model = args.get("model", "embed-v4.0")
        if model not in EMBED_MODELS:
            invalid("model", "choose a supported embed model.")
        texts = args.get("texts")
        if (
            not isinstance(texts, list)
            or not 1 <= len(texts) <= 96
            or any(not isinstance(t, str) or not t.strip() for t in texts)
        ):
            invalid("texts", "provide 1–96 non-empty text strings.")
        input_type = args.get("input_type")
        if input_type not in INPUT_TYPES:
            invalid("input_type", "choose a supported text input type.")
        data = {
            "model": model,
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        dim = args.get("output_dimension")
        if isinstance(dim, str) and dim.isascii() and dim.isdecimal():
            dim = int(dim)
        if dim is not None:
            if (
                isinstance(dim, bool)
                or not isinstance(dim, int)
                or dim not in DIMENSIONS[model]
            ):
                invalid(
                    "output_dimension",
                    "choose 256, 512, 1024 or 1536, or omit for the model default.",
                )
            data["output_dimension"] = dim
        expected_dim = (
            dim if dim is not None else (1536 if model == "embed-v4.0" else 2048)
        )
        if len(texts) * expected_dim * 32 > 256 * 1024:
            invalid(
                "texts",
                "batch is too large for the selected dimensions; split it into smaller batches.",
            )
        response = call("embed", data, key, timeout_seconds=timeout)
        vectors = (
            response.get("embeddings", {}).get("float")
            if isinstance(response, dict)
            and isinstance(response.get("embeddings"), dict)
            else None
        )
        if (
            not isinstance(vectors, list)
            or len(vectors) != len(texts)
            or not vectors
            or not isinstance(vectors[0], list)
            or not vectors[0]
        ):
            raise Failure(
                "provider_unavailable", "Cohere returned incomplete embeddings."
            )
        dimensions = len(vectors[0])
        if (dimensions != expected_dim) or any(
            not isinstance(v, list)
            or len(v) != dimensions
            or any(
                isinstance(n, bool)
                or not isinstance(n, (int, float))
                or not math.isfinite(n)
                for n in v
            )
            for v in vectors
        ):
            raise Failure(
                "provider_unavailable",
                "Cohere returned invalid embedding dimensions or values.",
            )
        if len(json.dumps(vectors)) > 384 * 1024:
            raise Failure(
                "provider_unavailable",
                "Cohere returned embeddings too large for the workflow output.",
            )
        return {
            "model": model,
            "vectors": vectors,
            "dimensions": dimensions,
            "request_id": request_id(response),
        }
    invalid("operation", "choose rerank or embed.")
