# Codex Model Catalog Proxy

A FastAPI adapter that serves an upstream model list in Codex catalog format at
`/models` and `/models.json`. It discovers models only; Codex sends inference
requests directly to a separately configured Responses-compatible provider.

## Quick start

Requires Python 3.10+ and either a working `codex debug models` command or a saved
Codex catalog configured with `CODEX_DEBUG_MODELS_FILE`.

```bash
python -m pip install fastapi httpx uvicorn
UPSTREAM_MODELS_URL=http://127.0.0.1:11434/v1/models \
  uvicorn codex_catalog_proxy:app --host 127.0.0.1 --port 8000
```

Configure Codex's provider `base_url` for inference and set its `model_catalog_url`
to `http://127.0.0.1:8000/models`. See [setup and configuration](docs/usage.md)
for a complete Codex example, overrides, authentication, and troubleshooting.
Keep the adapter on a trusted interface; it does not authenticate incoming requests.

## Discovery modes

`MODEL_CATALOG_MODE` defaults to `legacy`: read `UPSTREAM_MODELS_URL` and advertise
six fixed reasoning choices, without verifying support. The existing
`capabilities` mode reads `MODEL_CAPABILITIES_FILE` for strict filtering.

LiteLLM discovery is explicitly opt-in and requires no capabilities file:

```bash
MODEL_CATALOG_MODE=litellm \
UPSTREAM_MODEL_INFO_URL=http://127.0.0.1:4000/model/info \
UPSTREAM_API_KEY=your-litellm-key \
  uvicorn codex_catalog_proxy:app --host 127.0.0.1 --port 8000
```

`UPSTREAM_MODEL_INFO_URL` is the full URL (default shown above).
`UPSTREAM_API_KEY` supplies the existing upstream Bearer token. Other modes
continue to use `UPSTREAM_MODELS_URL`.

LiteLLM mode uses `data[].model_name` as the public slug and `model_info.mode`
and `supported_endpoints` to establish chat or Responses support. Multimodal chat
models qualify; unknown, conflicting, and completion-only capabilities are excluded.
`model_info.reasoning_effort_levels` supplies validated reasoning choices;
`default_reasoning_effort` supplies a default only if it belongs to those choices.
Missing or insufficient reasoning metadata produces an empty list and null default.
Duplicate deployments use the intersection of their choices. Discovery failures
return HTTP 502 without falling back to legacy.

See [discovery rules and metadata limitations](docs/discovery-modes.md) for the
exact fields, duplicate handling, and LiteLLM source references. Catalog discovery
does not verify tool calling or end-to-end Codex inference compatibility.
