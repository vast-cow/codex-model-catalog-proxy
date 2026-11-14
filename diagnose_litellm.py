"""Read-only LiteLLM metadata check; never print request settings or credentials."""
import json
import os
import sys

import httpx

import codex_catalog_proxy as proxy


def diagnose(payload, intended_names):
    # A neutral seed avoids requiring a local Codex CLI. Seeds cannot change
    # LiteLLM effort choices/defaults; configured exclusions and overrides apply.
    catalog = proxy.convert_models(
        payload, [{"slug": "__diagnostic_seed__"}], proxy.get_overrides(), mode="litellm"
    )
    models = {model["slug"]: model for model in catalog["models"]}
    groups = {}
    for entry in payload["data"]:
        groups.setdefault(entry["model_name"].strip(), []).append(entry.get("model_info") or {})

    rows = []
    for name in intended_names or groups:
        deployments = groups.get(name, [])
        model = models.get(name)
        choices = [level["effort"] for level in model["supported_reasoning_levels"]] if model else []
        causes = []
        if not deployments:
            causes.append("Public model name absent from /model/info.")
        elif name in proxy.EXCLUDE_MODEL_IDS:
            causes.append("Excluded by EXCLUDE_MODEL_IDS.")
        elif model is None:
            causes.append("Excluded: at least one deployment has unknown or conflicting conversational metadata.")
        elif not choices:
            for index, info in enumerate(deployments, 1):
                prefix = f"Deployment {index}: "
                if info.get("supports_reasoning") is False:
                    causes.append(prefix + "reasoning explicitly unsupported (supports_reasoning=false).")
                elif not isinstance(info.get("reasoning_effort_levels"), list):
                    causes.append(prefix + "explicit reasoning_effort_levels missing, null, or not a list.")
                elif not info["reasoning_effort_levels"]:
                    causes.append(prefix + "explicit reasoning_effort_levels is empty.")
                elif not any(value in proxy.REASONING_EFFORTS for value in info["reasoning_effort_levels"]
                             if isinstance(value, str)):
                    causes.append(prefix + "explicit levels contain no valid project effort values.")
            if len(deployments) > 1:
                causes.append("Choices are intersected across all deployments sharing this public name; the intersection is empty.")
        rows.append({
            "model_name": name,
            "deployments": [{field: info.get(field, "<missing>") for field in (
                "mode", "supported_endpoints", "reasoning_effort_levels", "default_reasoning_effort"
            )} for info in deployments],
            "codex_reasoning_choices": choices,
            "codex_default_reasoning_level": model["default_reasoning_level"] if model else None,
            "explanation": causes,
        })
    return rows


def main():
    # Require an explicit URL so running diagnostics cannot contact a guessed service.
    if not os.environ.get("UPSTREAM_MODEL_INFO_URL"):
        print("Set UPSTREAM_MODEL_INFO_URL to the full LiteLLM /model/info URL.", file=sys.stderr)
        return 1
    headers = {"Authorization": f"Bearer {proxy.UPSTREAM_API_KEY}"} if proxy.UPSTREAM_API_KEY else {}
    try:
        with httpx.Client(timeout=4.0, follow_redirects=False) as client:
            response = client.get(proxy.UPSTREAM_MODEL_INFO_URL, headers=headers)
            response.raise_for_status()
            rows = diagnose(response.json(), sys.argv[1:])
    except (httpx.HTTPError, ValueError, OSError) as exc:
        # Exception messages may contain URLs or response bodies. Print only type.
        print(f"Diagnostic failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
