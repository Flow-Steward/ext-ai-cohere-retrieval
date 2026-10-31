"""Descriptor references must preserve workflow contracts through installation."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_descriptor_runtime_references_match_manifest():
    manifest = yaml.safe_load((ROOT / "extension.yaml").read_text())
    descriptor = yaml.safe_load((ROOT / "extension-descriptor.yaml").read_text())
    manifest_contract = manifest["runtime"]["extension_contract_v2"]
    descriptor_contract = descriptor["runtime"]["extension_contract_v2"]
    assert descriptor_contract == manifest_contract
    assert all((ROOT / ref).is_file() for ref in descriptor_contract.values())
    operations = yaml.safe_load(
        (ROOT / descriptor_contract["operation_manifest"]).read_text()
    )
    assert {row["operation_id"] for row in operations["operations"]} == {
        "rerank",
        "embed",
    }
