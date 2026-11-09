"""Local Qwen-Image adapter over the comfyui-qwen MCP bridge.

The bridge is a Streamable HTTP MCP server (default ``http://localhost:8189/mcp``)
that wraps a ComfyUI Qwen-Image workflow behind one ``generate_image`` tool. All
generation behavior (workflow patching, sampler defaults, output naming) stays
in the bridge; this adapter only maps the catalog selection onto that tool's
arguments and asks for the full PNG inline (``return_image="full"``), so the
caller receives raw bytes and never a host-local path or a bridge URL the
browser cannot reach from inside a container.

The catalog ``model`` is the bridge's model profile: ``turbo`` (8-step distill,
~1.5 min) or ``base`` (25 steps, several minutes).
"""

from __future__ import annotations

import base64
import binascii
from datetime import timedelta
import logging
from typing import Any

import httpx

from deeptutor.services.generation_http import GenerationProviderError, build_auth_headers
from deeptutor.services.imagegen.base import BaseImagegenAdapter
from deeptutor.services.imagegen.config import ImagegenConfig

logger = logging.getLogger(__name__)

_TOOL_NAME = "generate_image"
# Local diffusion is far slower than a hosted API: one base-profile 1024² image
# takes several minutes and a batch of four multiplies that.
_MIN_CALL_TIMEOUT_S = 900
_MAX_BATCH = 4


def _parse_size(size: str) -> tuple[int, int] | None:
    """``"1024x768"`` → ``(1024, 768)``; anything else → ``None`` (bridge default)."""
    width, sep, height = size.lower().replace("*", "x").partition("x")
    if not sep:
        return None
    try:
        parsed = int(width.strip()), int(height.strip())
    except ValueError:
        return None
    return parsed if all(value > 0 for value in parsed) else None


class ComfyUIQwenImagegenAdapter(BaseImagegenAdapter):
    """Call the bridge's ``generate_image`` MCP tool and return the PNG bytes."""

    async def generate(
        self, prompt: str, config: ImagegenConfig, *, n: int = 1
    ) -> list[tuple[bytes, str]]:
        if not config.base_url:
            raise GenerationProviderError("No MCP endpoint URL configured for image generation.")
        arguments = self._arguments(prompt, config, n=n)
        timeout = max(config.request_timeout, _MIN_CALL_TIMEOUT_S)
        logger.debug(
            "comfyui-qwen imagegen url=%s model=%s batch=%d",
            config.base_url,
            arguments.get("model"),
            arguments["batch_size"],
        )
        result = await self._call_tool(config, arguments, timeout=timeout)
        return self._images(result)

    @staticmethod
    def _arguments(prompt: str, config: ImagegenConfig, *, n: int) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "prompt": prompt,
            "batch_size": max(1, min(n, _MAX_BATCH)),
            "return_image": "full",
        }
        if config.model:
            arguments["model"] = config.model
        size = _parse_size(config.size) if config.size else None
        if size:
            arguments["width"], arguments["height"] = size
        return arguments

    @staticmethod
    async def _call_tool(config: ImagegenConfig, arguments: dict[str, Any], *, timeout: int) -> Any:
        """One MCP session: initialize, call the tool, close.

        Errors are converted only after every transport context has exited, so
        the caller sees a plain ``GenerationProviderError`` rather than anyio's
        task-group exception wrapper.
        """
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        headers = {
            **(build_auth_headers(config.auth_style, config.api_key) if config.api_key else {}),
            **(config.extra_headers or {}),
        }
        try:
            async with httpx.AsyncClient(
                headers=headers or None,
                timeout=httpx.Timeout(timeout, connect=10.0),
            ) as http_client:
                async with streamable_http_client(config.base_url, http_client=http_client) as (
                    read,
                    write,
                    _,
                ):
                    async with ClientSession(
                        read, write, read_timeout_seconds=timedelta(seconds=timeout)
                    ) as session:
                        await session.initialize()
                        return await session.call_tool(_TOOL_NAME, arguments=arguments)
        except Exception as exc:
            detail = exc
            # anyio wraps transport failures in an ExceptionGroup; surface the cause.
            while isinstance(detail, BaseExceptionGroup) and detail.exceptions:
                detail = detail.exceptions[0]
            raise GenerationProviderError(
                f"Local Qwen-Image MCP request to {config.base_url} failed: {detail or exc!r}. "
                "Check that the comfyui-qwen bridge and ComfyUI are running; from a "
                "container, use host.docker.internal instead of localhost."
            ) from exc

    @staticmethod
    def _images(result: Any) -> list[tuple[bytes, str]]:
        from mcp import types

        texts = [block.text for block in result.content if isinstance(block, types.TextContent)]
        if result.isError:
            detail = "\n".join(texts).strip() or "no detail provided"
            raise GenerationProviderError(f"Local Qwen-Image generation failed: {detail}")
        images: list[tuple[bytes, str]] = []
        for block in result.content:
            if not isinstance(block, types.ImageContent):
                continue
            try:
                data = base64.b64decode(block.data, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise GenerationProviderError(
                    "Local Qwen-Image MCP returned invalid base64 image data."
                ) from exc
            if data:
                images.append((data, block.mimeType or "image/png"))
        if not images:
            detail = "\n".join(texts).strip()[-500:] or "no image content"
            raise GenerationProviderError(f"Local Qwen-Image MCP returned no image: {detail}")
        return images


__all__ = ["ComfyUIQwenImagegenAdapter"]
