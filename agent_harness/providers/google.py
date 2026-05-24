"""Google Gemini adapter — ``GeminiModel`` + ``GoogleProvider``.

Targets the ``google-genai`` Python SDK (optional dependency). The SDK is
imported lazily so this module loads cleanly even without it installed;
instantiating :class:`GoogleProvider` without the SDK raises
:class:`NotSupportedError`.

Wire shape: Google AI's ``models.generate_content_stream`` API.
Translates canonical :class:`Message` blocks ↔ Gemini ``Content``/``Part``
shapes and emits canonical ``ModelEvent``s.

Example:
    >>> # GoogleProvider(api_key="…")  # doctest: +SKIP
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from agent_harness.core.errors import ModelError, NotSupportedError
from agent_harness.core.events import (
    MessageDelta,
    MessageEnd,
    MessageStart,
    ModelEnd,
    ModelStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallEnd,
    ToolCallStart,
)
from agent_harness.core.models import (
    Message,
    ModelCapabilities,
    ModelSettings,
    ProviderEvent,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    Usage,
)

GEMINI_3_5_FLASH = "gemini-3.5-flash"
"""Default Gemini model identifier used by Wave-2."""

_CAPS_GEMINI_3_5_FLASH = ModelCapabilities(
    parallel_tool_calls=True,
    thinking=True,
    cache_control=True,
    vision=True,
    audio_input=True,
    audio_output=False,
    structured_output=True,
    context_window=1_000_000,
    max_output_tokens=64_000,
    supports_compaction=False,
)


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _require_sdk() -> Any:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - exercised via mocks
        raise NotSupportedError(
            "google-genai SDK is not installed; "
            "install with `pip install agent-harness[google]`",
            cause=exc,
        ) from exc
    return genai


# --- Provider ---------------------------------------------------------------


class GoogleProvider:
    """Auth + transport for the Google Gemini API.

    Example:
        >>> # GoogleProvider(api_key="…")  # doctest: +SKIP
    """

    name: str = "google"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        if client is not None:
            self._client = client
            return
        sdk = _require_sdk()
        key = api_key if api_key is not None else os.environ.get("GOOGLE_API_KEY")
        kwargs: dict[str, Any] = {}
        if key is not None:
            kwargs["api_key"] = key
        self._client = sdk.Client(**kwargs)

    @property
    def client(self) -> Any:
        """The underlying SDK client (used by the Model adapter)."""
        return self._client

    async def request(
        self,
        payload: dict[str, Any],
        *,
        stream: bool = False,
        timeout: float | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        """Issue a Gemini ``generate_content`` request."""
        del timeout  # honored at client construction; per-request override TODO
        if stream:
            iterator = await self._client.aio.models.generate_content_stream(**payload)
            async for chunk in iterator:
                # ProviderEvent allows extras at runtime (extra="allow").
                yield ProviderEvent(kind="raw_chunk", chunk=chunk)  # type: ignore[call-arg]
        else:
            response = await self._client.aio.models.generate_content(**payload)
            yield ProviderEvent(kind="response", response=response)  # type: ignore[call-arg]


# --- Model ------------------------------------------------------------------


class GeminiModel:
    """Gemini Generate-Content-API adapter.

    Example:
        >>> # GeminiModel(provider=p)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        provider: GoogleProvider,
        name: str = GEMINI_3_5_FLASH,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.name = name
        self.provider = provider
        self.capabilities = capabilities if capabilities is not None else _CAPS_GEMINI_3_5_FLASH

    # ----- message translation ---------------------------------------------

    @staticmethod
    def _block_to_part(
        block: Any,
        tool_names_by_call_id: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        if isinstance(block, TextBlock):
            return {"text": block.text}
        if isinstance(block, ToolCallBlock):
            return {"function_call": {"name": block.name, "args": block.arguments}}
        if isinstance(block, ToolResultBlock):
            content = block.content
            if isinstance(content, str):
                response_obj: dict[str, Any] = {"result": content}
            else:
                response_obj = {"result": content}
            # Gemini's function_response.name is the *tool* name, not the
            # call id. We reconstruct it by scanning prior assistant
            # ``ToolCallBlock``s for the matching ``id`` and falling back to
            # the call id if nothing was found (preserves prior behavior for
            # callers that happen to set them equal).
            name = (tool_names_by_call_id or {}).get(block.tool_call_id) or block.tool_call_id
            return {
                "function_response": {
                    "name": name,
                    "response": response_obj,
                }
            }
        if isinstance(block, ThinkingBlock):
            return {"text": block.text, "thought": True}
        return None

    @classmethod
    def _messages_to_wire(cls, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        # Build a call_id -> tool_name map from prior assistant ToolCallBlocks
        # so ToolResultBlocks (which only carry tool_call_id) can be wired
        # with the function name Gemini's function_response expects.
        tool_names_by_call_id: dict[str, str] = {}
        for msg in messages:
            for b in msg.content:
                if isinstance(b, ToolCallBlock):
                    tool_names_by_call_id[b.id] = b.name
        system: str | None = None
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                text = msg.text
                system = text if system is None else f"{system}\n\n{text}"
                continue
            role = "user" if msg.role in {"user", "tool"} else "model"
            parts: list[dict[str, Any]] = []
            for b in msg.content:
                part = cls._block_to_part(b, tool_names_by_call_id)
                if part is not None:
                    parts.append(part)
            if parts:
                contents.append({"role": role, "parts": parts})
        return system, contents

    @staticmethod
    def _tools_to_wire(tools: list[Any]) -> list[dict[str, Any]]:
        decls: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                # If already a function declaration, accept as-is.
                if "function_declarations" in t:
                    return [t]
                decls.append(t)
                continue
            name = getattr(t, "name", None)
            if name is None:
                continue
            decls.append(
                {
                    "name": name,
                    "description": getattr(t, "description", "") or "",
                    "parameters": getattr(t, "input_schema", None)
                    or getattr(t, "schema", {"type": "object", "properties": {}}),
                }
            )
        if not decls:
            return []
        return [{"function_declarations": decls}]

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[Any],
        settings: ModelSettings,
    ) -> dict[str, Any]:
        system, contents = self._messages_to_wire(messages)
        config: dict[str, Any] = {}
        if system is not None:
            config["system_instruction"] = system
        if settings.temperature is not None:
            config["temperature"] = settings.temperature
        if settings.top_p is not None:
            config["top_p"] = settings.top_p
        if settings.max_tokens is not None:
            config["max_output_tokens"] = settings.max_tokens
        if settings.seed is not None:
            config["seed"] = settings.seed
        wire_tools = self._tools_to_wire(tools)
        if wire_tools:
            config["tools"] = wire_tools
        if self.capabilities.thinking and settings.thinking_budget is not None:
            config["thinking_config"] = {
                "include_thoughts": True,
                "thinking_budget": settings.thinking_budget,
            }
        # Provider-specific carry-through (safety_settings, etc.) in extra.
        for k, v in settings.extra.items():
            config[k] = v
        payload: dict[str, Any] = {
            "model": self.name,
            "contents": contents,
        }
        if config:
            payload["config"] = config
        return payload

    # ----- request streaming ------------------------------------------------

    async def request(
        self,
        messages: list[Message],
        tools: list[Any],
        settings: ModelSettings,
    ) -> AsyncIterator[Any]:
        """Stream a model response as canonical ``ModelEvent``s."""
        payload = self._build_payload(messages, tools, settings)
        yield ModelStart(model_name=self.name)
        client = self.provider.client

        message_id = "gemini-msg"
        text_acc = ""
        thinking_acc = ""
        in_thinking = False
        tool_calls: list[ToolCallBlock] = []
        usage = Usage()
        started = False

        try:
            iterator = await client.aio.models.generate_content_stream(**payload)
            async for chunk in iterator:
                if not started:
                    yield MessageStart(message_id=message_id)
                    started = True
                # Each chunk is a GenerateContentResponse with candidates[0].content.parts
                cands = getattr(chunk, "candidates", None) or []
                for cand in cands:
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", None) or []
                    for part in parts:
                        is_thought = bool(getattr(part, "thought", False))
                        text = getattr(part, "text", None)
                        fcall = getattr(part, "function_call", None)
                        if text and is_thought:
                            if not in_thinking:
                                in_thinking = True
                                yield ThinkingStart(message_id=message_id)
                            thinking_acc += text
                            yield ThinkingDelta(
                                message_id=message_id,
                                delta=text,
                                partial=thinking_acc,
                            )
                        elif text:
                            if in_thinking:
                                yield ThinkingEnd(message_id=message_id)
                                in_thinking = False
                            text_acc += text
                            partial = Message(
                                role="assistant",
                                content=[TextBlock(text=text_acc)],
                                timestamp=_now(),
                            )
                            yield MessageDelta(message_id=message_id, delta=text, partial=partial)
                        elif fcall is not None:
                            if in_thinking:
                                yield ThinkingEnd(message_id=message_id)
                                in_thinking = False
                            name = getattr(fcall, "name", "") or ""
                            args = getattr(fcall, "args", None) or {}
                            # google-genai represents id as fcall.id (optional).
                            call_id = getattr(fcall, "id", None) or f"call-{len(tool_calls)}"
                            args_dict = dict(args) if not isinstance(args, dict) else args
                            yield ToolCallStart(tool_call_id=call_id, tool_name=name)
                            yield ToolCallEnd(
                                tool_call_id=call_id,
                                tool_name=name,
                                arguments=args_dict,
                            )
                            tool_calls.append(
                                ToolCallBlock(id=call_id, name=name, arguments=args_dict)
                            )
                meta = getattr(chunk, "usage_metadata", None)
                if meta is not None:
                    usage = _usage_from_meta(meta) or usage
        except NotSupportedError:
            raise
        except Exception as exc:
            raise ModelError(
                f"Gemini stream failed: {exc}",
                cause=exc,
                context={"model": self.name},
            ) from exc

        if not started:
            yield MessageStart(message_id=message_id)
        if in_thinking:
            yield ThinkingEnd(message_id=message_id)
        final = _build_final_message(text_acc, thinking_acc, tool_calls)
        yield MessageEnd(message_id=message_id, final=final, usage=usage)
        yield ModelEnd(message_id=message_id, usage=usage)

    async def compact_messages(self, msgs: list[Message]) -> list[Message]:
        """Gemini does not expose a standalone compaction endpoint."""
        raise NotSupportedError("Gemini does not support server-side compaction")


# --- helpers ----------------------------------------------------------------


def _build_final_message(
    text: str,
    thinking: str,
    tool_calls: list[ToolCallBlock],
) -> Message:
    blocks: list[Any] = []
    if thinking:
        blocks.append(ThinkingBlock(text=thinking))
    if text:
        blocks.append(TextBlock(text=text))
    blocks.extend(tool_calls)
    return Message(role="assistant", content=blocks, timestamp=_now())


def _usage_from_meta(meta: Any) -> Usage | None:
    if meta is None:
        return None
    return Usage(
        input_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(meta, "candidates_token_count", 0) or 0),
        cache_read_tokens=int(getattr(meta, "cached_content_token_count", 0) or 0),
        cache_write_tokens=0,
    )
