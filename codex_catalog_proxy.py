"""OpenAI-compatible /v1/models -> Codex model_catalog_url adapter.

Install: pip install fastapi httpx uvicorn

Default (legacy behavior, no model capabilities JSON needed):
    UPSTREAM_MODELS_URL=http://127.0.0.1:8001/v1/models \
    uvicorn codex_catalog_proxy:app --host 127.0.0.1 --port 8000

Opt in to model-aware capabilities filtering and reasoning presets:
    MODEL_CATALOG_MODE=capabilities \
    MODEL_CAPABILITIES_FILE=./vllm_model_capabilities.json \
    UPSTREAM_MODELS_URL=http://127.0.0.1:8001/v1/models \
    uvicorn codex_catalog_proxy:app --host 127.0.0.1 --port 8000

Legacy mode lists every upstream model and advertises six fixed reasoning
levels (none/minimal/low/medium/high/xhigh), as in the original adapter.
This does not mean all these levels are understood by the upstream model.

Capabilities mode requires a JSON file with the shape:
{"models": {"model-id": {"chat_supported": bool|null, "thinking": {...},
                         "reasoning_effort": {"parameter": ..., "options": ..., "default": ...}}}}
Only models with chat_supported=true are shown, with supported reasoning options.

This adapter serves ONLY the model catalog, not inference traffic. It cannot
translate Codex reasoning.effort into arbitrary vLLM chat_template_kwargs
or a model-specific numeric thinking budget.
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
MODEL_CAPABILITIES_FILE = os.getenv("MODEL_CAPABILITIES_FILE", "vllm_model_capabilities.json")
MODEL_CATALOG_MODE = os.getenv("MODEL_CATALOG_MODE", "legacy").strip().lower()
CATALOG_MODES = ("legacy", "capabilities")
if MODEL_CATALOG_MODE not in CATALOG_MODES:
    raise ValueError("MODEL_CATALOG_MODE must be 'legacy' or 'capabilities'")
CODEX_INSTRUCTIONS = os.getenv(
    "CODEX_INSTRUCTIONS",
    "You are a coding assistant. Follow the user's instructions and use available "
    "tools when appropriate. Never claim a tool action succeeded unless it did.",
)
SHELL_TYPE = os.getenv("CODEX_SHELL_TYPE", "unified_exec")
if SHELL_TYPE not in ("unified_exec", "disabled"):
    raise ValueError("CODEX_SHELL_TYPE must be 'unified_exec' or 'disabled'")

# Fixed levels advertised by the original catalog adapter (legacy mode).
LEGACY_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
# Accepted metadata values in capabilities mode. Not all models support these.
REASONING_EFFORTS = LEGACY_REASONING_EFFORTS + ("max",)
REASONING_DESCRIPTIONS = {
    "none": "Thinking disabled",
    "minimal": "Minimal reasoning",
    "low": "Fast responses with lighter reasoning",
    "medium": "Balances speed and reasoning depth",
    "high": "Greater reasoning depth for complex problems",
    "xhigh": "Extra-high reasoning depth",
    "max": "Maximum reasoning depth",
}

# A matched Codex model may supply sensible field defaults, but unrelated
# Codex models must not be used as a template for unknown upstream models.
MATCHED_SEED_FIELDS = (
    "default_reasoning_summary", "support_verbosity", "default_verbosity",
    "apply_patch_tool_type", "web_search_tool_type", "truncation_policy",
    "effective_context_window_percent", "additional_speed_tiers",
    "service_tiers", "default_service_tier", "available_access_programs",
    "supports_image_detail_original", "max_context_window", "context_window",
    "experimental_supported_tools", "input_modalities", "supports_search_tool",
    "supports_experimental_context", "use_responses_lite", "node_repl_disabled",
    "supports_reasoning_summary_parameter", "supports_reasoning_effort_updates",
    "tool_mode", "multi_agent_version", "include_skills_usage_instructions",
    "include_plugin_usage_instructions", "include_apps_usage_instructions",
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


def get_capabilities() -> dict[str, dict[str, Any]]:
    """Read the capabilities on each request so edits apply without restarting.

    Reject a missing / malformed file instead of advertising all upstream models
    with guessed capabilities. Models absent from the file are excluded.
    """
    value = _load_json_file(MODEL_CAPABILITIES_FILE)
    if not isinstance(value, dict) or not isinstance(value.get("models"), dict):
        raise ValueError("MODEL_CAPABILITIES_FILE must contain a 'models' object")
    result = value["models"]
    for model_id, capability in result.items():
        if not isinstance(model_id, str) or not model_id.strip() or not isinstance(capability, dict):
            raise ValueError("capability entries must be keyed by model ID and contain objects")
        chat = capability.get("chat_supported")
        if chat is not None and type(chat) is not bool:
            raise ValueError(f"{model_id}: chat_supported must be true/false/null")
        if chat is True:
            if not isinstance(capability.get("thinking"), dict):
                raise ValueError(f"{model_id}: missing thinking object")
            if not isinstance(capability.get("reasoning_effort"), dict):
                raise ValueError(f"{model_id}: missing reasoning_effort object")
            _validate_reasoning(model_id, capability)
    return result


def _validate_reasoning(model_id: str, capability: dict[str, Any]) -> None:
    thinking = capability["thinking"]
    t_supported = thinking.get("supported")
    if t_supported is not None and type(t_supported) is not bool:
        raise ValueError(f"{model_id}: thinking.supported must be true/false/null")
    t_default = thinking.get("default")
    if t_default is not None and type(t_default) is not bool:
        raise ValueError(f"{model_id}: thinking.default must be true/false/null")
    toggle = thinking.get("toggle")
    if toggle is not None and (
        not isinstance(toggle, dict)
        or not isinstance(toggle.get("parameter"), str)
        or not toggle["parameter"].strip()
    ):
        raise ValueError(f"{model_id}: invalid thinking.toggle")
    reasoning = capability["reasoning_effort"]
    options = reasoning.get("options")
    if options is not None and (
        not isinstance(options, list)
        or any(not isinstance(v, str) or v not in REASONING_EFFORTS for v in options)
        or len(set(options)) != len(options)
    ):
        raise ValueError(f"{model_id}: invalid reasoning_effort.options")
    default = reasoning.get("default")
    if default is not None and (not isinstance(default, str) or default not in REASONING_EFFORTS):
        raise ValueError(f"{model_id}: invalid reasoning_effort.default")
    param = reasoning.get("parameter")
    if param is not None and not isinstance(param, str):
        raise ValueError(f"{model_id}: invalid reasoning_effort.parameter")


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
            models = [
                m for m in catalog["models"]
                if isinstance(m, dict) and isinstance(m.get("slug"), str) and m["slug"].strip()
            ]
            if not models:
                raise ValueError("debug catalog has no usable models")
            _debug_cache = (time.monotonic(), models)
            return deepcopy(models)
        except (OSError, ValueError, subprocess.SubprocessError):
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


def _effective_reasoning(capability: dict[str, Any]) -> tuple[list[str], str | None]:
    """Build only Codex selections that vLLM can receive as reasoning.effort.

    'none' is added for toggles handled by recent vLLM versions. A custom
    thinking_budget needs inference request translation, which this catalog-only
    adapter cannot perform. Unknown effort options are NOT guessed.
    """
    reasoning = capability["reasoning_effort"]
    thinking = capability["thinking"]
    raw = reasoning.get("options")
    if not raw or thinking.get("supported") is False:
        return [], None

    parameter = reasoning.get("parameter")
    if parameter not in ("reasoning_effort", "chat_template_kwargs.reasoning_effort"):
        # The Codex model picker emits reasoning.effort, not arbitrary parameters.
        return [], None

    options = list(raw)
    toggle = thinking.get("toggle")
    if (
        thinking.get("supported") is True
        and isinstance(toggle, dict)
        and toggle.get("parameter") in (
            "chat_template_kwargs.enable_thinking", "reasoning_effort",
        )
        and "none" not in options
    ):
        options.insert(0, "none")

    explicit_default = reasoning.get("default")
    if explicit_default is not None and explicit_default not in options:
        raise ValueError(f"reasoning_effort.default is absent from options: {explicit_default}")
    # If the model defaults to non-thinking, prefer 'none' when possible.
    if thinking.get("default") is False and "none" in options:
        default = "none"
    else:
        default = explicit_default
    return options, default


def convert_models(
    payload: dict[str, Any],
    debug_models: list[dict[str, Any]] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    capabilities: dict[str, dict[str, Any]] | None = None,
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    """Convert upstream models using legacy or opt-in capabilities metadata.

    Providing `capabilities` alone does not switch modes: explicitly choose
    `mode="capabilities"`, or set MODEL_CATALOG_MODE in the environment.
    """
    mode = MODEL_CATALOG_MODE if mode is None else mode
    if mode not in CATALOG_MODES:
        raise ValueError("mode must be 'legacy' or 'capabilities'")
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("upstream must return an object with a 'data' array")
    if debug_models is None:
        debug_models = get_debug_models()
    if not debug_models:
        raise ValueError("debug model catalog is empty")
    if mode == "capabilities" and capabilities is None:
        capabilities = get_capabilities()
    if overrides is None:
        overrides = get_overrides()

    debug_by_slug = {model["slug"].strip(): model for model in debug_models}
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

        if mode == "capabilities":
            # Strict filtering: never guess whether unknown models can chat.
            capability = capabilities.get(slug)
            if not isinstance(capability, dict) or capability.get("chat_supported") is not True:
                continue
            _validate_reasoning(slug, capability)
            supported, default = _effective_reasoning(capability)
        else:
            # Original adapter's behavior: no capabilities file or filtering.
            supported = list(LEGACY_REASONING_EFFORTS)

        matching_debug = debug_by_slug.get(slug)
        seed = matching_debug or {}
        if mode == "legacy":
            default = next(
                (
                    level for level in (
                        item.get("default_reasoning_level"),
                        item.get("defaultReasoningEffort"),
                        seed.get("default_reasoning_level"),
                        "medium",
                    )
                    if level in LEGACY_REASONING_EFFORTS
                ),
                "medium",
            )

        name = item.get("name")
        description = item.get("description")
        if not isinstance(name, str) or not name.strip():
            name = (matching_debug or {}).get("display_name") or slug
        if not isinstance(description, str):
            description = (matching_debug or {}).get("description")

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
            "supports_reasoning_effort_updates": False,
            "input_modalities": ["text"],
            "base_instructions": CODEX_INSTRUCTIONS,
        }

        fields = MATCHED_SEED_FIELDS if matching_debug else ()
        for field in fields:
            if field in seed:
                model[field] = deepcopy(seed[field])
        model["shell_type"] = SHELL_TYPE
        model["base_instructions"] = CODEX_INSTRUCTIONS
        descriptions = _reasoning_descriptions(seed)
        if mode == "legacy":
            # Match the original precedence: upstream descriptions override seed.
            descriptions.update(_reasoning_descriptions(item))
        model["supported_reasoning_levels"] = [
            {"effort": effort,
             "description": (
                 descriptions.get(effort)
                 or REASONING_DESCRIPTIONS[effort]
             )}
            for effort in supported
        ]
        # Capabilities mode may return null for unverified defaults.
        model["default_reasoning_level"] = default

        context = item.get("context_window")
        if not _positive_int(context):
            context = item.get("context_length")
        if _positive_int(context):
            model["context_window"] = context
            max_context = item.get("max_context_window")
            model["max_context_window"] = (
                max_context
                if _positive_int(max_context) and max_context >= context
                else context
            )

        user_settings = deepcopy(overrides.get(slug, {}))
        if mode == "capabilities":
            # Capability metadata is authoritative for model selection and efforts.
            protected = {
                "slug", "supported_reasoning_levels", "default_reasoning_level",
                "visibility", "supported_in_api", "priority",
                "supports_reasoning_effort_updates",
            }
            attempted = protected.intersection(user_settings)
            if attempted:
                raise ValueError(f"cannot override capability-controlled fields for {slug}: "
                                 f"{', '.join(sorted(attempted))}")
        else:
            # Preserve the legacy override contract.
            user_settings.pop("slug", None)
            if "supported_reasoning_levels" in user_settings:
                raise ValueError("supported_reasoning_levels cannot be overridden; six levels required")
        model.update(user_settings)
        if mode == "legacy" and model.get("default_reasoning_level") not in LEGACY_REASONING_EFFORTS:
            raise ValueError(f"invalid default_reasoning_level override for {slug}")
        models.append(model)

    return {"models": models}


@app.get("/models")
@app.get("/models.json")
async def get_codex_models():
    headers = {}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    try:
        # Loading debug metadata can block; keep it off the event loop.
        tasks = [
            asyncio.to_thread(get_debug_models),
            asyncio.to_thread(get_overrides),
        ]
        if MODEL_CATALOG_MODE == "capabilities":
            tasks.append(asyncio.to_thread(get_capabilities))
        results = await asyncio.gather(*tasks)
        debug_models, overrides = results[:2]
        capabilities = results[2] if MODEL_CATALOG_MODE == "capabilities" else None
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=False) as client:
            response = await client.get(UPSTREAM_MODELS_URL, headers=headers)
            response.raise_for_status()
            payload = response.json()
        return convert_models(payload, debug_models, overrides, capabilities, mode=MODEL_CATALOG_MODE)
    except (httpx.HTTPError, ValueError, OSError, subprocess.SubprocessError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to build Codex catalog: {type(exc).__name__}",
        ) from exc
