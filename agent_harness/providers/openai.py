"""OpenAI Responses adapter — ``OpenAIResponsesModel`` + ``OpenAIProvider``.

Targets the ``openai`` Python SDK (optional dependency). The SDK is
imported lazily so this module loads cleanly even without it installed;
instantiating :class:`OpenAIProvider` without the SDK raises
:class:`NotSupportedError`.

Wire shape: OpenAI's Responses API (``client.responses.stream``).
Translates canonical :class:`Message` blocks ↔ Responses input items and
emits canonical ``ModelEvent``s.

Provider-specific knobs (``previous_response_id``, ``reasoning.effort``,
etc.) are NOT part of the Model surface — callers pass them in
``ModelSettings.extra`` or ``Message.metadata`` (per MD2).

Example:
    >>> # OpenAIProvider(api_key="sk-…")  # doctest: +SKIP
"""

from __future__ import annotations

import json
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

GPT_5_5 = "gpt-5.5"
"""Default OpenAI model identifier used by Wave-2."""

_CAPS_GPT_5_5 = ModelCapabilities(
    parallel_tool_calls=True,
    thinking=True,
    cache_control=True,
    vision=True,
    audio_input=False,
    audio_output=False,
    structured_output=True,
    context_window=400_000,
    max_output_tokens=128_000,
    supports_compaction=False,
)


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _require_sdk() -> Any:
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - exercised via mocks
        raise NotSupportedError(
            "openai SDK is not installed; install with `pip install agent-harness[openai]`",
            cause=exc,
        ) from exc
    return openai


# --- Provider ---------------------------------------------------------------


class OpenAIProvider:
    """Auth + transport for the OpenAI API.

    Example:
        >>> # OpenAIProvider(api_key="sk-…")  # doctest: +SKIP
    """

    name: str = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        organization: str | None = None,
    ) -> None:
        self.base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        if client is not None:
            self._client = client
            return
        sdk = _require_sdk()
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        kwargs: dict[str, Any] = {"max_retries": max_retries}
        if key is not None:
            kwargs["api_key"] = key
        if base_url is not None:
            kwargs["base_url"] = base_url
        if timeout is not None:
            kwargs["timeout"] = timeout
        if organization is not None:
            kwargs["organization"] = organization
        self._client = sdk.AsyncOpenAI(**kwargs)

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
        """Issue an OpenAI Responses ``responses.create`` request."""
        del timeout  # honored at client construction; per-request override TODO
        if stream:
            async with self._client.responses.stream(**payload) as ctx:
                async for chunk in ctx:
                    # ProviderEvent allows extras at runtime (extra="allow").
                    yield ProviderEvent(kind="raw_chunk", chunk=chunk)  # type: ignore[call-arg]
        else:
            response = await self._client.responses.create(**payload)
            yield ProviderEvent(kind="response", response=response)  # type: ignore[call-arg]


# --- Model ------------------------------------------------------------------


class OpenAIResponsesModel:
    """OpenAI Responses-API adapter.

    Example:
        >>> # OpenAIResponsesModel(provider=p)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        provider: OpenAIProvider,
        name: str = GPT_5_5,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.name = name
        self.provider = provider
        self.capabilities = capabilities if capabilities is not None else _CAPS_GPT_5_5

    # ----- message translation ---------------------------------------------
    #
    # Note: there is no per-block ``_block_to_wire`` helper. The Responses API
    # interleaves tool calls / tool outputs as *top-level* items in the
    # ``input`` array rather than embedding them in message content, so the
    # translation lives entirely in ``_messages_to_wire`` below.

    @classmethod
    def _messages_to_wire(cls, messages: list[Message]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for msg in messages:
            # Tool results: top-level function_call_output items.
            for b in msg.content:
                if isinstance(b, ToolResultBlock):
                    out = b.content if isinstance(b.content, str) else str(b.content)
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": b.tool_call_id,
                            "output": out,
                        }
                    )
            # Assistant tool calls: top-level function_call items.
            for b in msg.content:
                if isinstance(b, ToolCallBlock):
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": b.id,
                            "name": b.name,
                            "arguments": json.dumps(b.arguments),
                        }
                    )
            # Text content: a "message" item carrying input_text / output_text.
            text_blocks = [b for b in msg.content if isinstance(b, TextBlock)]
            if text_blocks:
                role = msg.role if msg.role in {"system", "user", "assistant"} else "user"
                content_type = "output_text" if role == "assistant" else "input_text"
                items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": [{"type": content_type, "text": b.text} for b in text_blocks],
                    }
                )
        return items

    @staticmethod
    def _tools_to_wire(tools: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                out.append(t)
                continue
            name = getattr(t, "name", None)
            if name is None:
                continue
            out.append(
                {
                    "type": "function",
                    "name": name,
                    "description": getattr(t, "description", "") or "",
                    "parameters": getattr(t, "input_schema", None)
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
        payload: dict[str, Any] = {
            "model": self.name,
            "input": self._messages_to_wire(messages),
        }
        if settings.max_tokens is not None:
            payload["max_output_tokens"] = settings.max_tokens
        if settings.temperature is not None:
            payload["temperature"] = settings.temperature
        if settings.top_p is not None:
            payload["top_p"] = settings.top_p
        wire_tools = self._tools_to_wire(tools)
        if wire_tools:
            payload["tools"] = wire_tools
        if settings.parallel_tool_calls is not None and self.capabilities.parallel_tool_calls:
            payload["parallel_tool_calls"] = settings.parallel_tool_calls
        if self.capabilities.thinking and settings.thinking_budget is not None:
            # Map budget → reasoning effort buckets (low/medium/high).
            payload["reasoning"] = {"effort": _budget_to_effort(settings.thinking_budget)}
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

        message_id = ""
        text_acc = ""
        thinking_acc = ""
        in_thinking = False
        # OpenAI Responses function-call items carry two IDs:
        #   item.id      — the output-item id; what delta/done events reference
        #                  via ``item_id``.
        #   item.call_id — the tool-call id we surface publicly (e.g.
        #                  ``call_abc123``); also the id the model expects on
        #                  the following ``function_call_output`` turn.
        # Internal accumulators are keyed by item.id (so delta lookups land);
        # tool_public_ids maps item.id -> the public call_id we emit.
        tool_args: dict[str, str] = {}
        tool_names: dict[str, str] = {}
        tool_public_ids: dict[str, str] = {}
        usage = Usage()
        finalised = False

        try:
            async with client.responses.stream(**payload) as stream:
                async for ev in stream:
                    ev_type = getattr(ev, "type", "") or ""
                    if ev_type == "response.created":
                        resp = getattr(ev, "response", None)
                        message_id = getattr(resp, "id", "") or ""
                        yield MessageStart(message_id=message_id)
                    elif ev_type == "response.output_text.delta":
                        chunk = getattr(ev, "delta", "") or ""
                        text_acc += chunk
                        partial = Message(
                            role="assistant",
                            content=[TextBlock(text=text_acc)],
                            timestamp=_now(),
                        )
                        yield MessageDelta(message_id=message_id, delta=chunk, partial=partial)
                    elif ev_type == "response.reasoning_summary_text.delta":
                        if not in_thinking:
                            in_thinking = True
                            yield ThinkingStart(message_id=message_id)
                        chunk = getattr(ev, "delta", "") or ""
                        thinking_acc += chunk
                        yield ThinkingDelta(
                            message_id=message_id, delta=chunk, partial=thinking_acc
                        )
                    elif ev_type == "response.reasoning_summary_text.done":
                        if in_thinking:
                            yield ThinkingEnd(message_id=message_id)
                            in_thinking = False
                    elif ev_type == "response.output_item.added":
                        item = getattr(ev, "item", None)
                        itype = getattr(item, "type", None)
                        if itype == "function_call":
                            # Key accumulators by item.id so delta events find
                            # the right slot; remember item.call_id for public
                            # emission.
                            item_id = getattr(item, "id", "") or ""
                            public_call_id = getattr(item, "call_id", "") or item_id
                            name = getattr(item, "name", "") or ""
                            tool_names[item_id] = name
                            tool_args.setdefault(item_id, "")
                            tool_public_ids[item_id] = public_call_id
                            yield ToolCallStart(tool_call_id=public_call_id, tool_name=name)
                    elif ev_type == "response.function_call_arguments.delta":
                        item_id = getattr(ev, "item_id", "") or getattr(ev, "call_id", "") or ""
                        chunk = getattr(ev, "delta", "") or ""
                        tool_args[item_id] = tool_args.get(item_id, "") + chunk
                        yield ToolCallDelta(
                            tool_call_id=tool_public_ids.get(item_id, item_id),
                            arguments_delta=chunk,
                        )
                    elif ev_type == "response.function_call_arguments.done":
                        item_id = getattr(ev, "item_id", "") or getattr(ev, "call_id", "") or ""
                        raw = getattr(ev, "arguments", None) or tool_args.get(item_id, "")
                        args = _parse_json_args(raw)
                        yield ToolCallEnd(
                            tool_call_id=tool_public_ids.get(item_id, item_id),
                            tool_name=tool_names.get(item_id, ""),
                            arguments=args,
                        )
                    elif ev_type == "response.completed":
                        resp = getattr(ev, "response", None)
                        usage = _usage_from(resp) or usage
                        final = _build_final_message(
                            text_acc, thinking_acc, tool_names, tool_args, tool_public_ids
                        )
                        yield MessageEnd(message_id=message_id, final=final, usage=usage)
                        yield ModelEnd(message_id=message_id, usage=usage)
                        finalised = True
        except NotSupportedError:
            raise
        except Exception as exc:
            raise ModelError(
                f"OpenAI stream failed: {exc}",
                cause=exc,
                context={"model": self.name},
            ) from exc

        if not finalised:
            final = _build_final_message(
                text_acc, thinking_acc, tool_names, tool_args, tool_public_ids
            )
            yield MessageEnd(message_id=message_id, final=final, usage=usage)
            yield ModelEnd(message_id=message_id, usage=usage)

    async def compact_messages(self, msgs: list[Message]) -> list[Message]:
        """OpenAI Responses does not expose a standalone compaction endpoint."""
        raise NotSupportedError("OpenAI Responses does not support server-side compaction")


# --- helpers ----------------------------------------------------------------


def _budget_to_effort(budget: int) -> str:
    if budget <= 0:
        return "minimal"
    if budget < 4_000:
        return "low"
    if budget < 16_000:
        return "medium"
    return "high"


def _parse_json_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"_value": parsed}


def _build_final_message(
    text: str,
    thinking: str,
    tool_names: dict[str, str],
    tool_args: dict[str, str],
    tool_public_ids: dict[str, str] | None = None,
) -> Message:
    blocks: list[Any] = []
    if thinking:
        blocks.append(ThinkingBlock(text=thinking))
    if text:
        blocks.append(TextBlock(text=text))
    for item_id, name in tool_names.items():
        args = _parse_json_args(tool_args.get(item_id, ""))
        # ToolCallBlock.id must be the *public* call_id (what the next turn's
        # function_call_output references), not the internal item.id.
        public_id = (tool_public_ids or {}).get(item_id, item_id)
        blocks.append(ToolCallBlock(id=public_id, name=name, arguments=args))
    return Message(role="assistant", content=blocks, timestamp=_now())


def _usage_from(resp: Any) -> Usage | None:
    if resp is None:
        return None
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    return Usage(
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        cache_read_tokens=int(
            getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        ),
        cache_write_tokens=0,
    )
