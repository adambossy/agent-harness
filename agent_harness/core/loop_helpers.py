"""Internal helpers used by the four loop nodes (Wave 4).

Pulled out of :mod:`.loop` so each file stays under the 500-LOC core target
(S12). Nothing here is part of the public surface — callers should not
import these directly.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable
from typing import Any, cast

from .events import ToolExecEnd, ToolExecStart
from .models import Message, TextBlock
from .run_context import (
    RunContext,
    RunResult,
    reset_current_run_ctx,
    set_current_run_ctx,
    snapshot_from_ctx,
)
from .tools import Tool, ToolCall, ToolResult

# --- Hook integration ------------------------------------------------------


async def fire_hook(
    ctx: RunContext[Any], event: str, payload: dict[str, Any]
) -> dict[str, Any] | None:
    """Fire a hook if the agent has a registry; return modified payload or None.

    On ``deny``, returns ``None`` and the caller should abort. On ``modify``,
    returns the new payload. On ``allow`` / ``ignore`` / no-hook, returns the
    original payload unchanged.
    """
    agent = ctx.agent
    if agent is None or agent.hooks is None:
        return payload
    response = await agent.hooks.fire(cast(Any, event), payload)
    if response.action == "deny":
        return None
    if response.action == "modify" and response.modified_payload is not None:
        return response.modified_payload
    return payload


# --- Snapshot persistence --------------------------------------------------


async def persist_snapshot(ctx: RunContext[Any], *, current_node: str) -> None:
    """Persist a node-boundary snapshot to ``Session`` if configured (LP4)."""
    snap = snapshot_from_ctx(ctx, current_node=current_node)
    agent = ctx.agent
    if agent is not None and agent.session is not None:
        await agent.session.add_run_state(snap)


# --- Misc resolution helpers ----------------------------------------------


def resolve_instructions(ctx: RunContext[Any]) -> str:
    """Resolve the agent's ``instructions`` (string or callable)."""
    agent = ctx.agent
    if agent is None:
        return ""
    instr = agent.instructions
    if callable(instr):
        return str(instr(ctx))
    return str(instr or "")


async def maybe_coerce_output(ctx: RunContext[Any], message: Message) -> tuple[Any, str | None]:
    """Coerce ``message`` into ``agent.output_type``; return ``(value, error)``.

    Default ``str`` returns text directly. Other types are tried via
    ``output_validator`` → ``output_type(text)`` → ``output_type.model_validate_json``.
    """
    agent = ctx.agent
    if agent is None:
        return message.text, None
    text = message.text
    out_type = agent.output_type
    validator = agent.output_validator
    if validator is not None:
        try:
            raw = validator(text)
            if isinstance(raw, Awaitable):
                raw = await raw
            return raw, None
        except Exception as exc:
            return None, f"output_validator raised: {exc}"
    if out_type is str:
        return text, None
    try:
        return out_type(text), None
    except Exception:
        pass
    try:
        validate = getattr(out_type, "model_validate_json", None)
        if callable(validate):
            return validate(text), None
    except Exception as exc:
        return None, f"output validation failed: {exc}"
    return None, f"could not coerce {text!r} to {out_type!r}"


# --- Tool gathering --------------------------------------------------------


async def gather_tools(ctx: RunContext[Any]) -> list[Tool]:
    """Collect tools from every configured toolset."""
    agent = ctx.agent
    if agent is None:
        return []
    out: list[Tool] = []
    for ts in agent.toolsets:
        out.extend(await ts.list_tools(ctx))
    return out


async def find_toolset(ctx: RunContext[Any], tool_name: str) -> Any | None:
    """Locate the toolset that owns ``tool_name``."""
    agent = ctx.agent
    if agent is None:
        return None
    for ts in agent.toolsets:
        for t in await ts.list_tools(ctx):
            if t.name == tool_name:
                return ts
    return None


async def find_tool(ctx: RunContext[Any], tool_name: str) -> Tool | None:
    """Locate a tool by name across every configured toolset."""
    agent = ctx.agent
    if agent is None:
        return None
    for ts in agent.toolsets:
        for t in await ts.list_tools(ctx):
            if t.name == tool_name:
                return t
    return None


# --- Tool-policy resolution -----------------------------------------------


def needs_approval(t: Tool, rc: RunContext[Any]) -> bool:
    """Resolve ``ToolPolicy.needs_approval`` honoring permission-mode bypasses."""
    if rc.permission_mode in ("bypass", "dont_ask"):
        return False
    if rc.permission_mode == "accept_edits" and (
        t.policy.is_destructive or not t.policy.is_read_only
    ):
        return False
    needs = t.policy.needs_approval
    if callable(needs):
        try:
            return bool(needs(rc))
        except TypeError:
            return bool(needs())
    return bool(needs)


def is_concurrency_safe(t: Tool, rc: RunContext[Any]) -> bool:
    """Resolve ``ToolPolicy.is_concurrency_safe`` (or read-only as a fallback)."""
    if t.policy.is_read_only:
        return True
    safe = t.policy.is_concurrency_safe
    if callable(safe):
        try:
            return bool(safe(rc))
        except TypeError:
            return bool(safe())
    return bool(safe)


def content_to_text(result: ToolResult) -> str:
    """Render a :class:`ToolResult` to a tool-result-block-friendly string."""
    if result.error:
        return result.error
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "".join(parts)


# --- Tool execution --------------------------------------------------------


async def execute_tool(rc: RunContext[Any], call: ToolCall, t: Tool) -> ToolResult:
    """Dispatch a single tool call through the toolset that owns it.

    Honors ``policy.timeout_seconds``, ``policy.failure_error_function``, and
    fires the ``PreToolUse`` / ``PostToolUse`` hooks. Returns a normal
    :class:`ToolResult`; errors become an errored result rather than
    propagating, so a sibling parallel call doesn't get cancelled by
    :class:`TaskGroup` semantics.
    """
    pre_payload: dict[str, Any] = {
        "tool_name": call.name,
        "tool_call_id": call.id,
        "arguments": dict(call.arguments),
    }
    pre = await fire_hook(rc, "PreToolUse", pre_payload)
    if pre is None:
        result = ToolResult(
            content=[TextBlock(text="denied by PreToolUse hook")],
            error="denied by PreToolUse hook",
        )
        await rc.event_bus.publish(
            ToolExecStart(tool_call_id=call.id, tool_name=call.name, arguments=dict(call.arguments))
        )
        await rc.event_bus.publish(
            ToolExecEnd(tool_call_id=call.id, result=result, error=result.error)
        )
        return result
    args = pre.get("arguments", call.arguments) if isinstance(pre, dict) else call.arguments
    if not isinstance(args, dict):
        args = call.arguments

    await rc.event_bus.publish(
        ToolExecStart(tool_call_id=call.id, tool_name=call.name, arguments=dict(args))
    )
    ts = await find_toolset(rc, call.name)
    timeout = t.policy.timeout_seconds
    # NestedInterruption (raised by ``Agent.as_tool``'s wrapper when the
    # child has pending approvals) must propagate up to ``ToolDispatch``
    # so the parent can roll those approvals into its own pending list
    # (AT3). Local import to avoid pulling subagents into module-init time.
    from .subagents import NestedInterruption

    token = set_current_run_ctx(rc)
    try:
        try:
            if ts is None:
                raise RuntimeError(f"no toolset owns tool {call.name!r}")
            coro = ts.call_tool(rc, ToolCall(id=call.id, name=call.name, arguments=args))
            result = await (
                asyncio.wait_for(coro, timeout=timeout) if timeout is not None else coro
            )
        except NestedInterruption:
            # Surface a ToolExecEnd marker for subscribers' symmetry, then
            # re-raise so ``ToolDispatch`` can roll the child's pending
            # approvals into the parent's interruption.
            marker = ToolResult(
                content=[TextBlock(text="<paused: subagent awaiting approval>")],
                error="<paused>",
            )
            await rc.event_bus.publish(
                ToolExecEnd(tool_call_id=call.id, result=marker, error=marker.error)
            )
            raise
        except TimeoutError:
            msg = f"tool {call.name!r} timed out after {timeout}s"
            result = ToolResult(content=[TextBlock(text=msg)], error=msg)
        except Exception as exc:
            formatter = t.policy.failure_error_function
            msg = formatter(exc) if formatter is not None else f"{type(exc).__name__}: {exc}"
            result = ToolResult(content=[TextBlock(text=msg)], error=msg)
    finally:
        reset_current_run_ctx(token)
    await rc.event_bus.publish(ToolExecEnd(tool_call_id=call.id, result=result, error=result.error))
    post_payload = {
        "tool_name": call.name,
        "tool_call_id": call.id,
        "error": result.error,
    }
    event_name = "PostToolUseFailure" if result.error else "PostToolUse"
    with contextlib.suppress(Exception):
        await fire_hook(rc, event_name, post_payload)
    return result


# --- Terminal-result helpers ----------------------------------------------


def processor_name(proc: Any) -> str:
    """Best-effort label for a history processor for event reporting."""
    return getattr(proc, "name", None) or type(proc).__name__ or "processor"


def last_assistant(messages: list[Message]) -> Message | None:
    """Return the last assistant message, or ``None`` if none exists."""
    for m in reversed(messages):
        if m.role == "assistant":
            return m
    return None


def terminal_result(rc: RunContext[Any]) -> RunResult[Any]:
    """Build a terminal :class:`RunResult` from the current context."""
    last = last_assistant(rc.messages)
    output: Any = rc.output if rc.output is not None else (last.text if last is not None else None)
    return RunResult(
        output=output,
        messages=list(rc.messages),
        pending_approvals=list(rc.pending_approvals),
        run_state=snapshot_from_ctx(rc, current_node="End"),
        usage=rc.usage.model_copy(),
    )


__all__ = [
    "content_to_text",
    "execute_tool",
    "find_tool",
    "find_toolset",
    "fire_hook",
    "gather_tools",
    "is_concurrency_safe",
    "last_assistant",
    "maybe_coerce_output",
    "needs_approval",
    "persist_snapshot",
    "processor_name",
    "resolve_instructions",
    "terminal_result",
]
