"""Anthropic Messages adapter — ``AnthropicMessagesModel`` + ``AnthropicProvider``.

Targets the ``anthropic`` Python SDK (optional dependency). The SDK is
imported lazily so this module loads cleanly even without it installed;
instantiating :class:`AnthropicProvider` without the SDK raises
:class:`NotSupportedError`.

Wire shape: Anthropic's ``messages.stream`` API. Translates canonical
:class:`Message` blocks ↔ Anthropic content; emits canonical ``ModelEvent``s
(``ModelStart`` / ``MessageStart`` / ``MessageDelta`` / ``MessageEnd`` /
``ToolCallStart`` / ``ToolCallDelta`` / ``ToolCallEnd`` /
``ThinkingStart`` / ``ThinkingDelta`` / ``ThinkingEnd`` / ``ModelEnd``).

Example:
    >>> # The adapter is constructed against a Provider that owns auth.
    >>> # AnthropicProvider(api_key="sk-…")  # doctest: +SKIP
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from agent_harness.core.credentials import (
    Credential,
    CredentialResolver,
    api_key_from_credential,
    resolve_credential,
)
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
    ToolCallDelta,
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

OPUS_4_7 = "claude-opus-4-7"
"""Default Anthropic model identifier used by Wave-2."""

_CAPS_OPUS_4_7 = ModelCapabilities(
    parallel_tool_calls=True,
    thinking=True,
    cache_control=True,
    vision=True,
    audio_input=False,
    audio_output=False,
    structured_output=True,
    context_window=1_000_000,
    max_output_tokens=64_000,
    supports_compaction=False,
)


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _require_sdk() -> Any:
    """Lazy-import the ``anthropic`` SDK; raise NotSupportedError if absent."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - exercised via mocks
        raise NotSupportedError(
            "anthropic SDK is not installed; install with `pip install agent-harness[anthropic]`",
            cause=exc,
        ) from exc
    return anthropic


# --- Provider ---------------------------------------------------------------


class AnthropicProvider:
    """Auth + transport for the Anthropic API.

    Owns the API key, base URL, and the underlying ``AsyncAnthropic`` HTTP
    client. Has no opinion on the wire format (that's the Model's job).

    Example:
        >>> # AnthropicProvider(api_key="sk-…")  # doctest: +SKIP
    """

    name: str = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        credential: Credential | None = None,
        credential_resolver: CredentialResolver | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        if client is not None:
            # Pre-built client (tests / bring-your-own transport): the key is
            # already baked in, so credential resolution is skipped entirely.
            self._client = client
            return
        self._client = self._build_client(
            api_key=api_key,
            credential=credential,
            credential_resolver=credential_resolver,
        )

    def _build_client(
        self,
        *,
        api_key: str | None = None,
        credential: Credential | None = None,
        credential_resolver: CredentialResolver | None = None,
    ) -> Any:
        """Construct the ``AsyncAnthropic`` client from a resolved key.

        The key comes from an explicit ``api_key`` or, failing that, the
        resolved :data:`Credential`. There is no ``ANTHROPIC_API_KEY``
        fallback: with none of those, :func:`resolve_credential` raises
        :class:`NoCredentialError`.
        """
        sdk = _require_sdk()
        key = self._resolve_key(api_key, credential, credential_resolver)
        kwargs: dict[str, Any] = {"max_retries": self._max_retries, "api_key": key}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        return sdk.AsyncAnthropic(**kwargs)

    def _resolve_key(
        self,
        api_key: str | None,
        credential: Credential | None,
        credential_resolver: CredentialResolver | None,
    ) -> str:
        if api_key is not None:
            return api_key
        cred = resolve_credential(credential=credential, credential_resolver=credential_resolver)
        return api_key_from_credential(cred, expected_provider=self.name)

    def use_credential(self, credential: Credential) -> None:
        """Rebuild the client to authenticate with ``credential`` (the
        per-run credential seam an :class:`Agent` drives)."""
        self._client = self._build_client(credential=credential)

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
        """Issue an Anthropic ``messages.create`` request.

        The model adapter normally drives the SDK directly via
        :attr:`client`; this method exists for protocol parity.
        """
        del timeout  # honored at client construction; per-request override TODO
        if stream:
            async with self._client.messages.stream(**payload) as stream_ctx:
                async for chunk in stream_ctx:
                    # ProviderEvent allows extras at runtime (extra="allow");
                    # the ignore quiets mypy's declared-field check.
                    yield ProviderEvent(kind="raw_chunk", chunk=chunk)  # type: ignore[call-arg]
        else:
            response = await self._client.messages.create(**payload)
            yield ProviderEvent(kind="response", response=response)  # type: ignore[call-arg]


# --- Model ------------------------------------------------------------------


class AnthropicMessagesModel:
    """Anthropic Messages-API adapter.

    Example:
        >>> # caps = ModelCapabilities(context_window=1_000_000, ...)  # doctest: +SKIP
        >>> # AnthropicMessagesModel(provider=p)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        provider: AnthropicProvider,
        name: str = OPUS_4_7,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.name = name
        self.provider = provider
        self.capabilities = capabilities if capabilities is not None else _CAPS_OPUS_4_7

    # ----- message translation ---------------------------------------------

    @staticmethod
    def _block_to_wire(block: Any) -> dict[str, Any] | None:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}
        if isinstance(block, ToolCallBlock):
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.arguments,
            }
        if isinstance(block, ToolResultBlock):
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_call_id,
                "content": block.content,
            }
        if isinstance(block, ThinkingBlock):
            return {"type": "thinking", "thinking": block.text}
        return None  # ImageBlock support intentionally omitted in v1

    @classmethod
    def _messages_to_wire(cls, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        system: str | None = None
        wire: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                text = msg.text
                system = text if system is None else f"{system}\n\n{text}"
                continue
            # Anthropic does not have a separate "tool" role: tool_results
            # ride on a *user* message.
            role = "user" if msg.role == "tool" else msg.role
            blocks: list[dict[str, Any]] = []
            for b in msg.content:
                wire_b = cls._block_to_wire(b)
                if wire_b is not None:
                    blocks.append(wire_b)
            if not blocks:
                continue
            entry: dict[str, Any] = {"role": role, "content": blocks}
            if msg.metadata:
                # Carry-through provider-specific hints (e.g. cache_control)
                # via Message.metadata — never leak to the public surface.
                cache = msg.metadata.get("cache_control")
                if cache is not None and blocks:
                    blocks[-1] = {**blocks[-1], "cache_control": cache}
            wire.append(entry)
        return system, wire

    @staticmethod
    def _tools_to_wire(tools: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                out.append(t)
                continue
            # Best-effort: support objects with name/description/schema attrs.
            name = getattr(t, "name", None)
            if name is None:
                continue
            out.append(
                {
                    "name": name,
                    "description": getattr(t, "description", "") or "",
                    "input_schema": getattr(t, "input_schema", None)
                    or getattr(t, "schema", {"type": "object", "properties": {}}),
                }
            )
        return out

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[Any],
        settings: ModelSettings,
    ) -> dict[str, Any]:
        system, wire_msgs = self._messages_to_wire(messages)
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": wire_msgs,
            "max_tokens": settings.max_tokens or self.capabilities.max_output_tokens or 4096,
        }
        if system is not None:
            payload["system"] = system
        if settings.temperature is not None:
            payload["temperature"] = settings.temperature
        if settings.top_p is not None:
            payload["top_p"] = settings.top_p
        wire_tools = self._tools_to_wire(tools)
        if wire_tools:
            payload["tools"] = wire_tools
        if (
            settings.parallel_tool_calls is False
            and self.capabilities.parallel_tool_calls
            and wire_tools
        ):
            payload["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        if self.capabilities.thinking and settings.thinking_budget is not None:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": settings.thinking_budget,
            }
        # Provider-specific carry-through.
        for k, v in settings.extra.items():
            payload[k] = v
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

        message_id: str = ""
        text_acc = ""
        thinking_acc = ""
        tool_args_acc: dict[int, str] = {}
        tool_meta: dict[int, dict[str, str]] = {}
        usage = Usage()
        finalised = False

        try:
            async with client.messages.stream(**payload) as stream:
                async for ev in stream:
                    ev_type = getattr(ev, "type", None)
                    if ev_type == "message_start":
                        message_id = getattr(getattr(ev, "message", None), "id", "") or ""
                        yield MessageStart(message_id=message_id)
                    elif ev_type == "content_block_start":
                        block = getattr(ev, "content_block", None)
                        idx = getattr(ev, "index", 0)
                        btype = getattr(block, "type", None)
                        if btype == "tool_use":
                            tcid = getattr(block, "id", "") or ""
                            tname = getattr(block, "name", "") or ""
                            tool_args_acc[idx] = ""
                            tool_meta[idx] = {"id": tcid, "name": tname}
                            yield ToolCallStart(tool_call_id=tcid, tool_name=tname)
                        elif btype == "thinking":
                            yield ThinkingStart(message_id=message_id)
                    elif ev_type == "content_block_delta":
                        delta = getattr(ev, "delta", None)
                        dtype = getattr(delta, "type", None)
                        idx = getattr(ev, "index", 0)
                        if dtype == "text_delta":
                            chunk = getattr(delta, "text", "") or ""
                            text_acc += chunk
                            partial = Message(
                                role="assistant",
                                content=[TextBlock(text=text_acc)],
                                timestamp=_now(),
                            )
                            yield MessageDelta(message_id=message_id, delta=chunk, partial=partial)
                        elif dtype == "input_json_delta":
                            piece = getattr(delta, "partial_json", "") or ""
                            tool_args_acc[idx] = tool_args_acc.get(idx, "") + piece
                            meta = tool_meta.get(idx, {"id": "", "name": ""})
                            yield ToolCallDelta(tool_call_id=meta["id"], arguments_delta=piece)
                        elif dtype == "thinking_delta":
                            piece = getattr(delta, "thinking", "") or ""
                            thinking_acc += piece
                            yield ThinkingDelta(
                                message_id=message_id, delta=piece, partial=thinking_acc
                            )
                    elif ev_type == "content_block_stop":
                        idx = getattr(ev, "index", 0)
                        if idx in tool_args_acc:
                            meta = tool_meta.get(idx, {"id": "", "name": ""})
                            args = _parse_json_args(tool_args_acc.get(idx, ""))
                            yield ToolCallEnd(
                                tool_call_id=meta["id"],
                                tool_name=meta["name"],
                                arguments=args,
                            )
                        elif thinking_acc:
                            yield ThinkingEnd(message_id=message_id)
                    elif ev_type == "message_delta":
                        u = getattr(getattr(ev, "usage", None), "output_tokens", None)
                        if u is not None:
                            usage = usage + Usage(output_tokens=int(u))
                    elif ev_type == "message_stop":
                        final = _build_final_message(
                            text_acc, thinking_acc, tool_meta, tool_args_acc
                        )
                        # Pull usage from final message if present.
                        msg_obj = getattr(ev, "message", None) or getattr(
                            stream, "current_message", None
                        )
                        usage = _usage_from(msg_obj) or usage
                        yield MessageEnd(message_id=message_id, final=final, usage=usage)
                        yield ModelEnd(message_id=message_id, usage=usage)
                        finalised = True
        except NotSupportedError:
            raise
        except Exception as exc:
            raise ModelError(
                f"Anthropic stream failed: {exc}",
                cause=exc,
                context={"model": self.name},
            ) from exc

        if not finalised:
            # Stream closed without a message_stop — synthesise a graceful end
            # so downstream consumers don't hang.
            final = _build_final_message(text_acc, thinking_acc, tool_meta, tool_args_acc)
            yield MessageEnd(message_id=message_id, final=final, usage=usage)
            yield ModelEnd(message_id=message_id, usage=usage)

    async def compact_messages(self, msgs: list[Message]) -> list[Message]:
        """Anthropic does not currently expose server-side compaction."""
        raise NotSupportedError("Anthropic does not support server-side compaction")


# --- helpers ----------------------------------------------------------------


def _parse_json_args(raw: str) -> dict[str, Any]:
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"_value": parsed}


def _build_final_message(
    text: str,
    thinking: str,
    tool_meta: dict[int, dict[str, str]],
    tool_args: dict[int, str],
) -> Message:
    blocks: list[Any] = []
    if thinking:
        blocks.append(ThinkingBlock(text=thinking))
    if text:
        blocks.append(TextBlock(text=text))
    for idx, meta in sorted(tool_meta.items()):
        args = _parse_json_args(tool_args.get(idx, ""))
        blocks.append(ToolCallBlock(id=meta["id"], name=meta["name"], arguments=args))
    return Message(role="assistant", content=blocks, timestamp=_now())


def _usage_from(obj: Any) -> Usage | None:
    if obj is None:
        return None
    u = getattr(obj, "usage", None)
    if u is None:
        return None
    return Usage(
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
    )
