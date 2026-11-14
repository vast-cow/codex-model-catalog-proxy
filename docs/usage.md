# Codex Model Catalog Proxy

A lightweight [FastAPI](https://fastapi.tiangolo.com/) service that converts an OpenAI-compatible `GET /v1/models` response into the model catalog format expected by [OpenAI Codex](https://github.com/openai/codex).

The service helps Codex discover models exposed by an OpenAI-compatible provider. It uses metadata from the locally installed Codex CLI where available, and supplies conservative defaults for model IDs that Codex does not recognize.

> [!IMPORTANT]
> **This is a model catalog adapter, not an inference reverse proxy.** It does not handle chat, Responses API, or other inference requests. Codex sends inference traffic directly to the provider configured in its `base_url`.

## Features

- Converts an upstream `{"data": [{"id": "..."}]}` model list into Codex's `{"models": [...]}` format.
- Uses `codex debug models` as a source of metadata for matching model IDs.
- Provides baseline metadata for unknown model IDs without copying unrelated model capabilities.
- Advertises the six reasoning-effort choices: `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`.
- Accepts optional per-model metadata overrides and an exclusion list.
- Reads a Codex catalog from a JSON file instead of invoking the CLI, if desired.
- Caches Codex CLI metadata and retains the last successfully loaded catalog if a later refresh fails.
- Serves the catalog at both `/models` and `/models.json`.

**Note:** Advertising a reasoning level in the model picker does **not** mean the upstream model actually supports that level. Verify reasoning, context length, tool calling, and other capabilities with your inference provider.

## Requirements

- Python **3.10 or newer**.
- The packages `fastapi`, `httpx`, and `uvicorn`.
- An OpenAI-compatible upstream endpoint returning a JSON object with a `data` array containing model objects with string `id` fields.
- Either:
  - a Codex CLI installation for which `codex debug models` returns a JSON catalog with a `models` array, or
  - a compatible JSON catalog file supplied using `CODEX_DEBUG_MODELS_FILE`.
- For actual inference through Codex, a **separate** provider/gateway supporting the Codex-required Responses API behavior. A working `/v1/models` endpoint alone is not sufficient.

## Quick start

Clone the repository:

```bash
git clone https://github.com/naoto-aoki-fy/codex-model-catalog-proxy.git
cd codex-model-catalog-proxy
```

Install dependencies:

```bash
python -m pip install fastapi httpx uvicorn
```

Start the adapter (example: a local OpenAI-compatible model-list endpoint):

```bash
UPSTREAM_MODELS_URL="http://127.0.0.1:11434/v1/models" \
  uvicorn codex_catalog_proxy:app --host 127.0.0.1 --port 8000
```

If `UPSTREAM_MODELS_URL` is omitted, the same `http://127.0.0.1:11434/v1/models` URL is used by default.

Check the catalog:

```bash
curl -fsS http://127.0.0.1:8000/models | python -m json.tool
```

The response has a top-level `models` array. Each entry includes a `slug`, `display_name`, `base_instructions`, reasoning-level metadata, and other Codex catalog fields.

You can also request `http://127.0.0.1:8000/models.json`.

## Configure Codex

Set up Codex to use the **inference gateway** as its provider and the **adapter** as the source of model metadata.

For an API-key-authenticated, Responses-compatible gateway, add settings similar to these to `~/.codex/config.toml`:

```toml
model = "your-upstream-model-id"
model_provider = "my_gateway"

[model_providers.my_gateway]
name = "My inference gateway"
base_url = "https://your-inference-gateway.example/v1"
wire_api = "responses"
env_key = "MY_GATEWAY_API_KEY"
model_catalog_url = "http://127.0.0.1:8000/models"

[features]
api_key_model_discovery = true
```

Provide the gateway's API key to the Codex process through the `MY_GATEWAY_API_KEY` environment variable. Replace the example provider URL and model ID with real values; the selected `model` must match an upstream `data[].id` value.

**How these URLs differ:**

| Setting | Purpose |
| --- | --- |
| `base_url` | The separate inference provider that receives Codex Responses API requests. |
| `model_catalog_url` | This adapter's full catalog URL, used to discover model metadata. |
| `UPSTREAM_MODEL_INFO_URL` | The full LiteLLM `/model/info` URL, used only in `litellm` mode. |
| `UPSTREAM_MODELS_URL` | The OpenAI-compatible model-list endpoint read in `legacy` and `capabilities` modes. |

Codex may send the selected provider's authentication headers to `model_catalog_url`. This application does not validate incoming credentials; keep it bound to loopback or add authentication in front of it before deploying beyond a trusted local environment. Do not send provider credentials to an untrusted catalog endpoint.

Codex's remote catalog discovery depends on the client's configuration and authentication mode. The `model_catalog_json` setting, if active, takes precedence over a remote catalog URL. Restart Codex after changing configuration.

See the [Codex gateway documentation](https://learn.chatgpt.com/docs/enterprise/roll-out-a-gateway) for supported authentication and catalog-discovery configurations.

## Configuration

This reference describes legacy discovery unless stated otherwise. For strict file or LiteLLM discovery, see [discovery modes](discovery-modes.md).

All settings are read from environment variables at process startup.

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_CATALOG_MODE` | `legacy` | Discovery mode: `legacy`, `capabilities`, or `litellm`. |
| `MODEL_CAPABILITIES_FILE` | `vllm_model_capabilities.json` | Required only in `capabilities` mode. |
| `UPSTREAM_MODEL_INFO_URL` | `http://127.0.0.1:4000/model/info` | Full LiteLLM metadata URL, used only in `litellm` mode. |
| `UPSTREAM_MODELS_URL` | `http://127.0.0.1:11434/v1/models` | URL of the upstream OpenAI-compatible model list. |
| `UPSTREAM_API_KEY` | Empty | Optional Bearer token sent to the selected discovery endpoint. |
| `CODEX_DEBUG_MODELS_FILE` | Empty | JSON file to use instead of running `codex debug models`. |
| `CODEX_DEBUG_CACHE_SECONDS` | `900` | Cache duration in seconds for the locally loaded Codex metadata; `0` requests a refresh each time. |
| `MODEL_OVERRIDES_FILE` | Empty | Path to a JSON object of overrides keyed by model ID. |
| `EXCLUDE_MODEL_IDS` | Empty | Comma-separated, exact model IDs to omit. |
| `CODEX_INSTRUCTIONS` | Built-in coding-assistant instruction | Base instructions assigned to every exposed model. |
| `CODEX_SHELL_TYPE` | `unified_exec` | Shell type to advertise: `unified_exec` or `disabled`. |

### Use a saved Codex catalog

If `codex debug models` is unavailable at runtime, first generate a compatible snapshot on a machine where it works:

```bash
codex debug models > codex-models.json
```

Then start the adapter with:

```bash
CODEX_DEBUG_MODELS_FILE="./codex-models.json" \
UPSTREAM_MODELS_URL="http://127.0.0.1:11434/v1/models" \
uvicorn codex_catalog_proxy:app --host 127.0.0.1 --port 8000
```

The file must contain an object with a nonempty `models` array and entries with nonempty `slug` strings. Keep it in sync with the Codex version whose metadata you want to use.

### Override model metadata

Create `model-overrides.json`:

```json
{
  "your-upstream-model-id": {
    "display_name": "My Custom Model",
    "default_reasoning_level": "medium",
    "context_window": 32768
  }
}
```

Start the service with `MODEL_OVERRIDES_FILE="./model-overrides.json"`.

Overrides are applied after the catalog metadata is assembled. For safety and consistency:

- A `slug` override is ignored; the upstream model ID stays authoritative.
- `supported_reasoning_levels` cannot be overridden; the service always publishes six effort choices.
- `default_reasoning_level` must be one of `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`.
- Set capability-related fields only to values you have verified against the actual inference provider.

To hide specific model IDs, set, for example:

```bash
EXCLUDE_MODEL_IDS="model-to-hide,another-model"
```

## How it works

1. Load the local Codex model catalog by running `codex debug models` or reading `CODEX_DEBUG_MODELS_FILE`.
2. Make an HTTP GET request to `UPSTREAM_MODELS_URL` (with an upstream Bearer token if configured).
3. Read model IDs from the upstream response's `data` array, skipping invalid, duplicated, or explicitly excluded entries.
4. For a model ID that matches a local Codex `slug`, copy selected capability metadata from that exact entry. Otherwise, start from conservative, text-only defaults.
5. Set the shell type and base instructions, populate the six displayed reasoning levels, honor valid upstream context metadata, and apply per-model overrides.
6. Return a JSON object with a `models` array.

The upstream can optionally supply fields such as `name`, `description`, `context_window`, `context_length`, `max_context_window`, `default_reasoning_level`, `defaultReasoningEffort`, and `supported_reasoning_levels`.

The adapter does not copy model-specific prompts (`model_messages` or upstream `base_instructions`) from one model to another. `CODEX_INSTRUCTIONS` supplies the effective `base_instructions` for every emitted model.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| HTTP `502` from `/models` | Confirm the upstream URL is reachable, returns HTTP 2xx and JSON with a `data` array, and responds within the 4-second upstream timeout. |
| HTTP `502` on initial startup requests | Make sure `codex debug models` is executable and returns a valid catalog, or configure `CODEX_DEBUG_MODELS_FILE`. |
| HTTP `502` after configuring overrides | Check that `MODEL_OVERRIDES_FILE` is valid JSON and contains an object of per-model objects. Also check allowed `default_reasoning_level` values. |
| No models appear | Verify the upstream `data[].id` entries and `EXCLUDE_MODEL_IDS`; inspect the adapter response with `curl`. |
| Codex does not use the remote catalog | Verify the exact `model_catalog_url`, authentication/discovery configuration, and absence of an overriding `model_catalog_json`. Restart Codex. |
| A model appears but inference fails | Check the separate inference `base_url`, credentials, model ID, and Responses API compatibility. The adapter does not proxy inference. |
| Incorrect model capabilities | Confirm the upstream model's real support, then use verified per-model overrides. A displayed reasoning level is not proof of support. |

The adapter returns a generic `502` error when it cannot build a catalog. For additional diagnostics, inspect the service logs and test the upstream endpoint directly.

## Security and deployment notes

- **Not an authentication gateway:** the FastAPI endpoints have no inbound authentication or access controls. The example binds Uvicorn to `127.0.0.1` intentionally.
- **Protect secrets:** `UPSTREAM_API_KEY` is sent only to the configured upstream model-list endpoint. Codex may independently send provider credentials to the catalog URL.
- **Catalog versus authorization:** publishing an ID does not make a model callable or grant access to it. Enforce authorization at the inference provider.
- **Version compatibility:** the service uses metadata from its locally installed Codex catalog, not the requesting client's `client_version` parameter. Check compatibility when clients and the server use different Codex versions.
- **Upstream failure behavior:** a previously successful local Codex metadata snapshot may be reused when its refresh fails, but the upstream HTTP model-list request must still succeed.
- **No inference forwarding:** streaming, tool-call behavior, and Responses API compatibility are the responsibility of the inference provider.

## API

| Method | Path | Response |
| --- | --- | --- |
| `GET` | `/models` | Codex-compatible JSON model catalog (`{"models": [...]}`). |
| `GET` | `/models.json` | Same catalog response. |

The routes ignore query parameters, including `client_version`, and generate the catalog using the process's configured local metadata and upstream endpoint.

## References

- [OpenAI Codex repository](https://github.com/openai/codex)
- [Codex gateway and model-catalog guidance](https://learn.chatgpt.com/docs/enterprise/roll-out-a-gateway)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
