"""Flow Steward host v2 subprocess entrypoint."""

import json
import sys
from flowsteward_extension_sdk import runtime_context_from_payload
import runtime


def handle(payload):
    try:
        runtime_context_from_payload(payload)
        context = payload.get("runtime_context") or {}
        test_mode = context.get("test_mode", False)
        if not isinstance(test_mode, bool):
            raise runtime.Failure("invalid_payload", "test_mode must be a boolean.")
        if test_mode:
            raise runtime.Failure(
                "test_mode_suppressed", "Cohere calls are suppressed in test mode."
            )
        claim = context.get("external_effect_id")
        if not isinstance(claim, str) or not claim.strip() or len(claim) > 512:
            raise runtime.Failure(
                "effect_claim_required",
                "A durable effect claim is required before calling Cohere.",
            )
        if payload.get("mode") != "action":
            raise runtime.Failure("invalid_payload", "An action is required.")
        action = payload.get("action") or {}
        target = action.get("target") or {}
        connection = target.get("connection") or {}
        args = action.get("input") or {}
        if (
            not isinstance(args, dict)
            or not isinstance(connection, dict)
            or not args.get("connection_ref")
            or args.get("connection_ref") != connection.get("connection_id")
            or connection.get("connection_type") != "cohere_api"
        ):
            raise runtime.Failure(
                "invalid_connection", "Select the matching Cohere API connection."
            )
        secrets = connection.get("secrets") or {}
        result = runtime.execute(
            action.get("action_id"),
            args,
            secrets.get("api_key", ""),
            connection.get("config") or {},
        )
        return {
            "ok": True,
            "result": result,
            "errors": [],
            "provider_message_id": result.get("request_id") or "",
        }
    except runtime.Failure as exc:
        return {
            "ok": False,
            "error_code": exc.code,
            "provider_message_id": exc.request_id,
            "retryable": False,
            "definitely_no_external_effect": exc.code
            in {
                "test_mode_suppressed",
                "effect_claim_required",
                "invalid_payload",
                "invalid_connection",
                "not_configured",
            }
            and not exc.request_id,
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
