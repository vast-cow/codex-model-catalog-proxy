#!/usr/bin/env python3
"""Retrieve Codex model catalogs from a remote, already-running App Server.

Requires:
    pip install 'websockets>=14'

Examples:
    # Enumerate all providers using normal model/list RPC
    python remote_codex_models.py --method normal

    # Enumerate all providers using debug models
    python remote_codex_models.py --method debug

    # Try normal RPC for each provider, fallback to debug
    python remote_codex_models.py --method auto

    # Specific provider
    python remote_codex_models.py --provider rivault --method normal

    # Multiple providers
    python remote_codex_models.py \
        --provider rivault \
        --provider openai \
        --method auto

    # Remote server
    python remote_codex_models.py \
        --url ws://127.0.0.1:4500 \
        --method normal \
        --output all_models.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import websockets


class RpcClient:
    def __init__(self, websocket):
        self.websocket = websocket
        self.next_id = 0

    async def call(self, method, params=None, timeout=135):
        self.next_id += 1
        request_id = self.next_id

        await self.websocket.send(
            json.dumps({
                "id": request_id,
                "method": method,
                "params": {} if params is None else params,
            })
        )

        async with asyncio.timeout(timeout):
            while True:
                message = json.loads(
                    await self.websocket.recv()
                )

                # Ignore unsolicited App Server notifications.
                if message.get("id") != request_id:
                    continue

                if "error" in message:
                    raise RuntimeError(
                        f"{method}: {message['error']}"
                    )

                return message.get("result", {})

    async def initialize(self):
        await self.call(
            "initialize",
            {
                "clientInfo": {
                    "name": "remote-model-enumerator",
                    "version": "1.0.0",
                },
            },
        )

        await self.websocket.send(
            json.dumps({
                "method": "initialized",
                "params": {},
            })
        )

    async def list_provider_ids(self):
        """Enumerate all providers through modelProvider/list."""
        provider_ids = []
        cursor = None
        seen_cursors = set()

        while True:
            params = (
                {"cursor": cursor}
                if cursor
                else {}
            )

            result = await self.call(
                "modelProvider/list",
                params,
            )

            if isinstance(result, list):
                providers = result
                cursor = None

            elif isinstance(result, dict):
                providers = result.get("data")

                if providers is None:
                    providers = result.get(
                        "providers",
                        [],
                    )

                cursor = result.get("nextCursor")

            else:
                raise RuntimeError(
                    "unexpected modelProvider/list response: "
                    f"{type(result).__name__}"
                )

            if not isinstance(providers, list):
                raise RuntimeError(
                    "modelProvider/list did not return a list"
                )

            for provider in providers:
                if not isinstance(provider, dict):
                    continue

                provider_id = provider.get("id")

                if (
                    isinstance(provider_id, str)
                    and provider_id
                ):
                    provider_ids.append(provider_id)

            if not cursor:
                break

            if cursor in seen_cursors:
                raise RuntimeError(
                    "modelProvider/list pagination loop detected"
                )

            seen_cursors.add(cursor)

        return list(dict.fromkeys(provider_ids))

    async def normal_models(self, provider_id):
        """Retrieve models for one provider using model/list RPC.

        Requires the remote App Server implementation to support
        the modelProvider parameter on model/list.
        """
        models = []
        cursor = None
        seen_cursors = set()

        while True:
            params = {
                "modelProvider": provider_id,
            }

            if cursor:
                params["cursor"] = cursor

            result = await self.call(
                "model/list",
                params,
            )

            if isinstance(result, list):
                page = result
                cursor = None

            elif isinstance(result, dict):
                page = result.get("data")

                if page is None:
                    page = result.get(
                        "models",
                        [],
                    )

                cursor = result.get("nextCursor")

            else:
                raise RuntimeError(
                    "unexpected model/list response: "
                    f"{type(result).__name__}"
                )

            if not isinstance(page, list):
                raise RuntimeError(
                    "model/list did not return a list"
                )

            models.extend(page)

            if not cursor:
                break

            if cursor in seen_cursors:
                raise RuntimeError(
                    "model/list pagination loop detected"
                )

            seen_cursors.add(cursor)

        return models

    async def debug_models(
        self,
        provider_id,
        codex_binary="codex",
    ):
        """Retrieve models through remote codex debug models."""
        params = {
            "command": [
                codex_binary,
                "debug",
                "models",
                "-c",
                f"model_provider={provider_id}",
            ],
            "timeoutMs": 120_000,
            "outputBytesCap": 8 * 1024 * 1024,
        }

        # Inherit the server's sandbox policy.
        result = await self.call(
            "command/exec",
            params,
        )

        if not isinstance(result, dict):
            raise RuntimeError(
                "unexpected command/exec response"
            )

        code = result.get("exitCode")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        if code != 0:
            raise RuntimeError(
                f"remote codex exited with code {code}\n"
                f"stderr: {stderr.strip()}\n"
                f"stdout: {stdout[:500]}"
            )

        try:
            catalog = json.loads(stdout)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "remote output is not valid JSON "
                f"({exc})\n"
                f"stderr: {stderr[:500]}\n"
                f"stdout: {stdout[:500]}"
            ) from exc

        if not isinstance(catalog, dict):
            raise RuntimeError(
                "unexpected debug catalog type: "
                f"{type(catalog).__name__}"
            )

        models = catalog.get("models")

        if not isinstance(models, list):
            raise RuntimeError(
                "unexpected catalog format: "
                f"{list(catalog)}"
            )

        return models

    async def get_models(
        self,
        provider_id,
        method="normal",
        codex_binary="codex",
    ):
        """Retrieve models for one provider using selected method.

        Returns:
            (models, actual_method)
        """
        if method == "normal":
            models = await self.normal_models(
                provider_id
            )
            return models, "normal"

        if method == "debug":
            models = await self.debug_models(
                provider_id,
                codex_binary,
            )
            return models, "debug"

        if method == "auto":
            try:
                models = await self.normal_models(
                    provider_id
                )
                return models, "normal"

            except Exception as exc:
                print(
                    f"  Normal RPC failed: {exc}",
                    file=sys.stderr,
                )

                print(
                    "  Falling back to debug method...",
                    file=sys.stderr,
                )

                models = await self.debug_models(
                    provider_id,
                    codex_binary,
                )

                return models, "debug"

        raise ValueError(
            f"unsupported method: {method}"
        )


def normalize_model(model, provider_id):
    """Normalize model entries into a common output format."""
    if not isinstance(model, dict):
        return None

    name = (
        model.get("slug")
        or model.get("id")
        or model.get("model")
    )

    if not isinstance(name, str) or not name:
        return None

    display_name = (
        model.get("displayName")
        or model.get("display_name")
        or name
    )

    return {
        "provider": provider_id,
        "model": name,
        "displayName": display_name,
    }


async def run(args):
    token = os.getenv(
        "CODEX_APP_SERVER_TOKEN"
    )

    headers = (
        {"Authorization": f"Bearer {token}"}
        if token
        else None
    )

    async with websockets.connect(
        args.url,
        additional_headers=headers,
        max_size=16 * 1024 * 1024,
    ) as websocket:
        rpc = RpcClient(websocket)

        await rpc.initialize()

        # Always enumerate available providers.
        discovered_providers = (
            await rpc.list_provider_ids()
        )

        if not discovered_providers:
            raise RuntimeError(
                "no providers discovered"
            )

        print(
            f"Discovered providers: "
            f"{len(discovered_providers)}"
        )

        for provider in discovered_providers:
            print(f"  - {provider}")

        # Optional provider filtering.
        if args.provider:
            requested = set(args.provider)

            unknown = (
                requested
                - set(discovered_providers)
            )

            if unknown:
                raise RuntimeError(
                    "unknown provider(s): "
                    + ", ".join(sorted(unknown))
                )

            providers = [
                provider
                for provider in discovered_providers
                if provider in requested
            ]

        else:
            providers = discovered_providers

        rows = []
        failed = False

        for provider in providers:
            print(
                f"\n[{provider}] "
                f"method={args.method}"
            )

            try:
                models, actual_method = (
                    await rpc.get_models(
                        provider_id=provider,
                        method=args.method,
                        codex_binary=args.codex_binary,
                    )
                )

            except Exception as exc:
                failed = True

                print(
                    f"  ERROR: {exc}",
                    file=sys.stderr,
                )

                continue

            count = 0

            for model in models:
                row = normalize_model(
                    model,
                    provider,
                )

                if row is None:
                    continue

                name = row["model"]

                print(f"  {name}")

                rows.append(row)
                count += 1

            print(
                f"  Found: {count} "
                f"(via {actual_method})"
            )

        if args.output:
            output_path = Path(args.output)

            output_path.write_text(
                json.dumps(
                    rows,
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )

            print(
                f"\nSaved: {output_path}"
            )

        print(
            f"Total: {len(rows)}"
        )

        if failed:
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate Codex model providers and "
            "retrieve models for each provider."
        )
    )

    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:4500",
        help=(
            "WebSocket endpoint of the "
            "existing App Server"
        ),
    )

    parser.add_argument(
        "--method",
        choices=[
            "normal",
            "debug",
            "auto",
        ],
        default="normal",
        help=(
            "Model retrieval method: "
            "normal=model/list RPC (default), "
            "debug=codex debug models, "
            "auto=normal with per-provider "
            "debug fallback"
        ),
    )

    parser.add_argument(
        "--provider",
        action="append",
        help=(
            "Provider ID to include (repeatable). "
            "Omit to enumerate all providers."
        ),
    )

    parser.add_argument(
        "--codex-binary",
        default="codex",
        help=(
            "Path to codex executable "
            "ON THE REMOTE SERVER"
        ),
    )

    parser.add_argument(
        "--output",
        default="all_models.json",
        help=(
            "Output JSON file path "
            "(default: all_models.json)"
        ),
    )

    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
