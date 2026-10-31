"""Offline smoke for host validation and both Cohere operations."""

import pathlib
import sys
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import main

scope = {
    "group_id": None,
    "for_each_id": None,
    "step_id": "smoke",
    "step_execution_id": "test",
    "external_effect_id": "claim-1",
}


def request(operation, values):
    return {
        "contract_version": "extension_host_v2",
        "mode": "action",
        "runtime_context": scope,
        "action": {
            "action_id": operation,
            "input": {"connection_ref": "cohere-smoke", **values},
            "target": {
                "connection": {
                    "connection_id": "cohere-smoke",
                    "connection_type": "cohere_api",
                    "secrets": {"api_key": "mock"},
                }
            },
        },
    }


with patch(
    "runtime.call",
    return_value={"id": "test", "results": [{"index": 0, "relevance_score": 0.5}]},
):
    assert main.handle(
        request("rerank", {"query": "q", "documents": [{"id": "a", "text": "text"}]})
    )["ok"]
with patch(
    "runtime.call", return_value={"id": "test", "embeddings": {"float": [[0.5] * 1536]}}
):
    assert main.handle(
        request("embed", {"texts": ["text"], "input_type": "search_document"})
    )["ok"]
print("offline smoke passed")
