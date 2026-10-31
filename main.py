"""Flow Steward host v2 subprocess entrypoint."""

import json
import sys
from flowsteward_extension_sdk import runtime_context_from_payload
import runtime


def handle(payload):
    try:
        runtime_context_from_payload(payload)
        action = payload.get("action") or {}
        target = action.get("target") or {}
        connection = target.get("connection") or {}
        secrets = connection.get("secrets") or {}
        result = runtime.execute(
            action.get("action_id"),
            action.get("input") or {},
            secrets.get("api_key", ""),
            connection.get("config") or {},
        )
        return {"ok": True, "result": result, "errors": []}
    except runtime.Failure as exc:
        return {
            "ok": False,
            "error_code": exc.code,
            "error": str(exc),
            "errors": [{"code": exc.code, "message": str(exc)}],
        }
    except (ValueError, TypeError, AttributeError):
        return {
            "ok": False,
            "error_code": "invalid_payload",
            "error": "Invalid host envelope or action input.",
            "errors": [
                {
                    "code": "invalid_payload",
                    "message": "Invalid host envelope or action input.",
                }
            ],
        }

    except Exception:
        return {
            "ok": False,
            "error_code": "provider_unavailable",
            "error": "Cohere request failed.",
            "errors": [
                {"code": "provider_unavailable", "message": "Cohere request failed."}
            ],
        }


def main():
    try:
        payload = json.load(sys.stdin)
        response = handle(payload)
    except (ValueError, TypeError):
        response = {
            "ok": False,
            "error_code": "invalid_payload",
            "error": "Invalid JSON request.",
            "errors": [],
        }
    print(json.dumps(response, allow_nan=False))
    return 0 if response["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
