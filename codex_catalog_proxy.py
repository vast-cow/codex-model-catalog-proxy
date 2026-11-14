"""OpenAI-compatible /v1/models -> Codex model_catalog_url adapter.

Install: pip install fastapi httpx uvicorn
Run: UPSTREAM_MODELS_URL=http://127.0.0.1:11434/v1/models \
     uvicorn codex_catalog_proxy:app --host 127.0.0.1 --port 8000

This is a catalog adapter, NOT a reverse proxy for inference requests.
The local `codex debug models` catalog supplies initial model metadata.
"""

import asyncio
from copy import deepcopy
import json
import os
import subprocess
import threading
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Codex Model Catalog Proxy")

UPSTREAM_MODELS_URL = os.getenv(
    "UPSTREAM_MODELS_URL", "http://127.0.0.1:11434/v1/models"
)
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
CODEX_INSTRUCTIONS = os.getenv(
    "CODEX_INSTRUCTIONS",
    "You are a coding assistant. Follow the user's instructions and use available "
    "tools when appropriate. Never claim a tool action succeeded unless it did.",
)
SHELL_TYPE = os.getenv("CODEX_SHELL_TYPE", "unified_exec")
if SHELL_TYPE not in ("unified_exec", "disabled"):
    raise ValueError("CODEX_SHELL_TYPE must be 'unified_exec' or 'disabled'")

REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)

REASONING_DESCRIPTIONS = {
    "none": "No reasoning",
    "minimal": "Minimal reasoning",
    "low": "Fast responses with lighter reasoning",
    "medium": "Balances speed and reasoning depth",
    "high": "Greater reasoning depth for complex problems",
    "xhigh": "Extra high reasoning depth for complex problems",
}

# Unknown upstream models must not inherit capabilities from an unrelated Codex model.
MATCHED_SEED_FIELDS = (
    "default_reasoning_summary", "support_verbosity", "default_verbosity",
    "apply_patch_tool_type", "web_search_tool_type", "truncation_policy",
    "effective_context_window_percent", "additional_speed_tiers",
    "service_tiers", "default_service_tier", "available_access_programs",
    "supports_reasoning_summary_parameter", "supports_image_detail_original",
    "max_context_window", "context_window", "experimental_supported_tools",
    "input_modalities", "supports_search_tool", "supports_experimental_context",
    "use_responses_lite", "supports_reasoning_effort_updates",
    "node_repl_disabled", "tool_mode", "multi_agent_version",
    "include_skills_usage_instructions", "include_plugin_usage_instructions",
    "include_apps_usage_instructions",
)

DEBUG_MODELS_FILE = os.getenv("CODEX_DEBUG_MODELS_FILE", "")
MODEL_OVERRIDES_FILE = os.getenv("MODEL_OVERRIDES_FILE", "")
DEBUG_CACHE_SECONDS = max(0, int(os.getenv("CODEX_DEBUG_CACHE_SECONDS", "900")))
EXCLUDE_MODEL_IDS = {
    x.strip() for x in os.getenv("EXCLUDE_MODEL_IDS", "").split(",") if x.strip()
}
_debug_cache: tuple[float, list[dict[str, Any]]] | None = None
_debug_lock = threading.Lock()


def _load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_overrides() -> dict[str, dict[str, Any]]:
    """Optional per-model overrides: {"model-id": {"field": value}}."""
    if not MODEL_OVERRIDES_FILE:
        return {}
    value = _load_json_file(MODEL_OVERRIDES_FILE)
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or not isinstance(v, dict)
        for k, v in value.items()
    ):
        raise ValueError("MODEL_OVERRIDES_FILE must contain an object of model objects")
    return value


def get_debug_models() -> list[dict[str, Any]]:
    """Read and cache the local Codex model catalog, optionally from a JSON file."""
    global _debug_cache
    with _debug_lock:
        if _debug_cache and time.monotonic() - _debug_cache[0] < DEBUG_CACHE_SECONDS:
            return deepcopy(_debug_cache[1])
        try:
            if DEBUG_MODELS_FILE:
                catalog = _load_json_file(DEBUG_MODELS_FILE)
            else:
                completed = subprocess.run(
                    ["codex", "debug", "models"], check=True,
                    capture_output=True, text=True, timeout=60,
                )
                catalog = json.loads(completed.stdout)
            if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
                raise ValueError("debug catalog must contain a 'models' array")
            models = [m for m in catalog["models"] if isinstance(m, dict)
                      and isinstance(m.get("slug"), str) and m["slug"].strip()]
            if not models:
                raise ValueError("debug catalog has no usable models")
            _debug_cache = (time.monotonic(), models)
            return deepcopy(models)
        except (OSError, ValueError, subprocess.SubprocessError):
            # A previously successful load can serve as a last-known-good snapshot.
            if _debug_cache:
                return deepcopy(_debug_cache[1])
            raise


def _reasoning_descriptions(model: dict[str, Any]) -> dict[str, str]:
    descriptions = {}
    for preset in model.get("supported_reasoning_levels", []):
        if not isinstance(preset, dict):
            continue
        effort = preset.get("effort")
        description = preset.get("description")
        if effort in REASONING_EFFORTS and isinstance(description, str):
            descriptions[effort] = description
    return descriptions


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def convert_models(
    payload: dict[str, Any],
    debug_models: list[dict[str, Any]] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert an upstream catalog, seeded from `codex debug models`."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("upstream must return an object with a 'data' array")
    if debug_models is None:
        debug_models = get_debug_models()
    if not debug_models:
        raise ValueError("debug model catalog is empty")

    debug_by_slug = {model["slug"].strip(): model for model in debug_models}
    overrides = overrides if overrides is not None else get_overrides()

    models = []
    seen = set()
    for item in payload["data"]:
        if not isinstance(item, dict):
            continue
        slug = item.get("id")
        if not isinstance(slug, str) or not slug.strip():
            continue
        slug = slug.strip()
        if slug in seen or slug in EXCLUDE_MODEL_IDS:
            continue
        seen.add(slug)
        matching_debug = debug_by_slug.get(slug)
        seed = matching_debug or {}

        name = item.get("name")
        description = item.get("description")
        if not isinstance(name, str) or not name.strip():
            name = (matching_debug or {}).get("display_name") or slug
        if not isinstance(description, str):
            description = (matching_debug or {}).get("description")

        # Supply required fields, then overlay safe metadata from Codex's catalog.
        # Do not copy its model_messages or base_instructions into another model:
        # those are model-specific prompts, not general-purpose defaults.
        model = {
            "slug": slug,
            "display_name": name,
            "description": description,
            "shell_type": SHELL_TYPE,
            "visibility": "list",
            "supported_in_api": True,
            "priority": len(models) + 1,
            "availability_nux": None,
            "upgrade": None,
            "support_verbosity": False,
            "default_reasoning_summary": "none",
            "effective_context_window_percent": 90,
            "tool_mode": "direct",
            "supports_search_tool": False,
            "web_search_tool_type": "text",
            "default_verbosity": None,
            "apply_patch_tool_type": None,
            "truncation_policy": {"mode": "bytes", "limit": 10000},
            "experimental_supported_tools": [],
            "supports_reasoning_summary_parameter": False,
            "input_modalities": ["text"],
            # Accepted by both older and newer Codex catalog deserializers.
            "base_instructions": CODEX_INSTRUCTIONS,
        }

        fields = MATCHED_SEED_FIELDS if matching_debug else ()
        for field in fields:
            if field in seed:
                model[field] = deepcopy(seed[field])

        # The caller's shell setting and instructions always take precedence.
        model["shell_type"] = SHELL_TYPE
        model["base_instructions"] = CODEX_INSTRUCTIONS

        # The picker always advertises the six requested efforts. Use descriptions
        # from Codex metadata when available, then fall back to generic descriptions.
        descriptions = _reasoning_descriptions(seed)
        descriptions.update(_reasoning_descriptions(item))
        model["supported_reasoning_levels"] = [
            {
                "effort": effort,
                "description": descriptions.get(effort) or REASONING_DESCRIPTIONS[effort],
            }
            for effort in REASONING_EFFORTS
        ]

        model["default_reasoning_level"] = next(
            (
                value
                for value in (
                    item.get("default_reasoning_level"),
                    item.get("defaultReasoningEffort"),
                    seed.get("default_reasoning_level"),
                    "medium",
                )
                if value in REASONING_EFFORTS
            ),
            "medium",
        )

        # Context size may be advertised by nonstandard upstream fields.
        context = item.get("context_window")
        if not _positive_int(context):
            context = item.get("context_length")
        if _positive_int(context):
            model["context_window"] = context
            # Explicit upstream context metadata overrides the debug template.
            max_context = item.get("max_context_window")
            model["max_context_window"] = (
                max_context
                if _positive_int(max_context) and max_context >= context
                else context
            )
        # User overrides are authoritative; do not allow changing the upstream model ID.
        user_settings = deepcopy(overrides.get(slug, {}))
        user_settings.pop("slug", None)
        model.update(user_settings)
        if model.get("default_reasoning_level") not in REASONING_EFFORTS:
            raise ValueError(f"invalid default_reasoning_level override for {slug}")
        if "supported_reasoning_levels" in user_settings:
            raise ValueError("supported_reasoning_levels cannot be overridden; six levels required")
        models.append(model)

    return {"models": models}


@app.get("/models")
@app.get("/models.json")
async def get_codex_models():
    headers = {}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    try:
        # subprocess.run is blocking, so perform it outside FastAPI's event loop.
        debug_models = await asyncio.to_thread(get_debug_models)
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=False) as client:
            response = await client.get(UPSTREAM_MODELS_URL, headers=headers)
            response.raise_for_status()
            payload = response.json()
        return convert_models(payload, debug_models, get_overrides())
    except (httpx.HTTPError, ValueError, OSError, subprocess.SubprocessError) as exc:
        # Do not silently fall back to an OpenAI model catalog.
        raise HTTPException(
            status_code=502,
            detail=f"Failed to build Codex catalog: {type(exc).__name__}",
        ) from exc
