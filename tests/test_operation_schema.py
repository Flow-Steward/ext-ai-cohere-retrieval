"""Examples the workflow input schemas must accept and reject."""

from pathlib import Path

import jsonschema
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def schema(operation, field):
    data = yaml.safe_load((ROOT / "contracts/operation_manifest.yaml").read_text())
    op = next(row for row in data["operations"] if row["operation_id"] == operation)
    return next(row["schema"] for row in op["inputs"] if row["name"] == field)


@pytest.mark.parametrize(
    ("operation", "field", "value", "valid"),
    [
        ("rerank", "top_n", 1, True),
        ("rerank", "top_n", 1.5, False),
        ("rerank", "top_n", 1001, False),
        ("rerank", "query", "   ", False),
        ("rerank", "model", "rerank-v3.5", True),
        ("rerank", "model", "unknown", False),
        ("rerank", "documents", [{"id": "a", "text": "hello"}], True),
        ("rerank", "documents", [1], False),
        ("rerank", "documents", [{"id": " ", "text": "hello"}], False),
        ("rerank", "documents", [{"id": "x" * 129, "text": "hello"}], False),
        ("embed", "texts", ["hello"], True),
        ("embed", "texts", ["x"] * 97, False),
        ("embed", "texts", [1], False),
        ("embed", "texts", ["   "], False),
        ("embed", "input_type", "search_query", True),
        ("embed", "input_type", "image", False),
        ("embed", "model", "embed-v4.0", True),
        ("embed", "model", "other", False),
        ("embed", "output_dimension", "256", True),
        ("embed", "output_dimension", 256, True),
        ("embed", "output_dimension", "768", False),
    ],
)
def test_operation_schema_examples(operation, field, value, valid):
    validator = jsonschema.Draft202012Validator(schema(operation, field))
    assert validator.is_valid(value) is valid
