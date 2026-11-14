# Discovery modes

Select a mode with `MODEL_CATALOG_MODE`. Settings are read at process startup.
Both `/models` and `/models.json` return the same existing `{"models": [...]}`
shape. All modes use the local Codex metadata source and optional overrides.

| Mode | Discovery source | Reasoning choices |
| --- | --- | --- |
| `legacy` (default) | `UPSTREAM_MODELS_URL` | Fixed `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; support is not verified. |
| `capabilities` | `UPSTREAM_MODELS_URL` plus `MODEL_CAPABILITIES_FILE` | Verified file options and existing vLLM toggle conversion. |
| `litellm` | `UPSTREAM_MODEL_INFO_URL` only | Explicit deployment metadata, intersected across duplicates. |

## File capabilities

```bash
MODEL_CATALOG_MODE=capabilities \
MODEL_CAPABILITIES_FILE=./vllm_model_capabilities.json \
UPSTREAM_MODELS_URL=http://127.0.0.1:8001/v1/models \
  uvicorn codex_catalog_proxy:app --host 127.0.0.1 --port 8000
```

The file has a `models` object keyed by upstream ID. Entries with
`chat_supported: true` require `thinking` and `reasoning_effort` objects:

```json
{"models": {"public-model": {
  "chat_supported": true,
  "thinking": {"supported": true},
  "reasoning_effort": {
    "parameter": "reasoning_effort", "options": ["low", "high"], "default": "low"
  }
}}}
```

Missing files or malformed metadata fail the request. Absent or unknown chat
support excludes models. Existing thinking toggles can add `none` when compatible
with vLLM; this conversion does not apply to LiteLLM metadata.

## LiteLLM response and model selection

Configure the full `/model/info` URL with `UPSTREAM_MODEL_INFO_URL`
(default `http://127.0.0.1:4000/model/info`). The selected endpoint receives
`Authorization: Bearer <UPSTREAM_API_KEY>` when configured. No `/v1/models`
lookup or local capabilities file is needed in this mode.

The actual deployment response is a `data` array:

```json
{"data": [{
  "model_name": "public-model",
  "litellm_params": {"model": "provider/internal-model"},
  "model_info": {
    "id": "internal-deployment-id",
    "mode": "chat",
    "supported_endpoints": ["/v1/chat/completions", "/v1/responses"],
    "supports_reasoning": true,
    "reasoning_effort_levels": ["low", "high"],
    "default_reasoning_effort": "low"
  }
}]}
```

Only the public `model_name` becomes the slug. Internal IDs and
`litellm_params.model` are never substituted. `EXCLUDE_MODEL_IDS` and overrides
are keyed by public name. Ordering follows first appearance of each public name;
priorities remain consecutive. Every deployment sharing a public name must
establish conversational support, or that name is excluded.

Selection rules:

- `mode: chat` or `mode: responses` establishes conversational support when
  `supported_endpoints` is absent or null.
- A non-null endpoint list must be a list of strings containing
  `/chat/completions`, `/v1/chat/completions`, `/responses`, or `/v1/responses`.
  It may establish support when mode is absent or null.
- Any other non-null mode conflicts with conversational support and excludes
  the deployment, even if its endpoints contain a conversational route.
- An empty, invalid, or exclusively non-conversational endpoint list excludes
  the deployment even when its mode says chat or Responses.
- Vision and other multimodal capabilities do not disqualify conversational
  models. Names, text input support, and tool capability flags never establish
  chat support on their own.

Completion-only models are excluded, including those accepting text prompts.
Text generation does not establish the chat/Responses behavior required by Codex.
Even included chat models still need a separate gateway that implements the
Responses behavior, streaming, and tools Codex requires; discovery is no inference
compatibility test.

## LiteLLM reasoning rules

Only a JSON list in `model_info.reasoning_effort_levels` establishes choices.
Values are filtered and deduplicated against this project's vocabulary, in order:
`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`.
Unknown values are discarded. An explicitly empty list stays empty. A missing,
null, or malformed list yields no choices. `supports_reasoning: false` overrides
even an explicit list and yields no choices.

For duplicate deployments, choices are intersected across every deployment.
Missing or unknown choices on one deployment empty the intersection. A default
is published only if every deployment reports the same
`model_info.default_reasoning_effort` and it belongs to the final intersection.
Otherwise the default is null. Matched Codex metadata and overrides cannot supply
unverified choices or defaults in this mode. Protected override fields follow
the same restrictions as file capabilities mode.

These fields do **not** enumerate choices:

- `supports_reasoning: true`.
- `supported_openai_params` containing `reasoning_effort`.
- `litellm_params.reasoning_effort` (a request setting/default).
- Per-level `supports_*_reasoning_effort` flags alone.

Flags-only discovery is deliberately unsupported. LiteLLM's capability resolver
uses different missing-value behavior for different flags and Chat versus
Responses, plus provider-specific exceptions and model-map lookups. Treating
all flags uniformly would invent choices. The adapter does not reproduce that
resolver or fetch a local LiteLLM model map. It never falls back to six legacy
choices or adds `none` through the vLLM thinking-toggle code.

## Errors and metadata limitations

HTTP failures, timeouts, invalid JSON, invalid top-level response shape, and
malformed deployment objects return the existing generic HTTP 502 error. There
is no legacy fallback. Valid metadata that cannot establish capabilities yields
an excluded model or empty reasoning choices rather than a request failure.
A missing/null `model_info` is treated as unknown; a non-object value is malformed.

LiteLLM versions and deployments vary in the fields they expose. Operators can
supply explicit metadata in their LiteLLM deployment's `model_info`, but should
verify it against the real inference path. The adapter consumes only metadata
actually returned by `/model/info`; it neither calls `/model_group/info` nor
assumes that endpoint's aggregated fields exist on deployment responses.

The implementation was checked against current LiteLLM documentation and source:

- [Model management documentation](https://docs.litellm.ai/docs/proxy/model_management):
  `/model/info`, custom metadata passthrough, and informational endpoint metadata.
- [Proxy source](https://github.com/BerriAI/litellm/blob/main/litellm/proxy/proxy_server.py):
  `model_info_v1` response and `_enrich_model_info_with_litellm_data` deployment enrichment.
- [Metadata types](https://github.com/BerriAI/litellm/blob/main/litellm/types/utils.py):
  modes, endpoints, explicit levels, defaults, and reasoning flags.
- [Reasoning capability resolver](https://github.com/BerriAI/litellm/blob/main/litellm/litellm_core_utils/reasoning_effort_capability.py):
  explicit-list precedence, intersections, and the differing flag semantics.
