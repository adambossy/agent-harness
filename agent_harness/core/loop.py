"""The 4-node typed graph that drives a single turn (Wave 4 / LP1-LP8).

Nodes:

* :class:`PrepareTurn` — load history, append the user prompt, snapshot.
* :class:`ModelRequest` — gather tools, stream the model, validate output.
* :class:`ToolDispatch` — approval gate + parallel-safe batching.
* :class:`DecideNext` — terminate or run history processors and loop.

Each node's ``run`` return-type annotation is the edge set the pydantic-graph
engine reads at graph-build time (LP2). Loop helpers live in
:mod:`.loop_helpers` to keep this file under the per-file LOC budget.

Imports are Protocols-only (LP5): no concrete provider / sandbox / session.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .events import (
    ApprovalRequested,
    CompactionEnd,
    CompactionStart,
    Error,
    MessageDelta,
    MessageEnd,
    MessageStart,
    ModelEnd,
    ModelRetryRequest,
    ModelStart,
    NodeEnter,
    NodeExit,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolExecEnd,
    ToolExecStart,
)
from .graph import End, GraphCtx, Node
from .history import apply_processor
from .loop_helpers import (
    content_to_text,
    execute_tool,
    find_tool,
    fire_hook,
    gather_tools,
    is_concurrency_safe,
    last_assistant,
    maybe_coerce_output,
    needs_approval,
    persist_snapshot,
    processor_name,
    resolve_instructions,
    terminal_result,
)
from .models import Message, TextBlock, ToolResultBlock, Usage
from .run_context import (
    RunContext,
    RunResult,
    is_tool_allowed_in_mode,
    make_interruption,
)
from .run_state import ApprovalRequest
from .tools import Tool, ToolCall, ToolResult

# ---------------------------------------------------------------------------
# PrepareTurn
# ---------------------------------------------------------------------------


@dataclass
class PrepareTurn(Node[RunContext[Any], None, RunResult[Any]]):
    """Load history, append the prompt, snapshot."""

    async def run(
        self,
        ctx: GraphCtx[RunContext[Any], None],
    ) -> ModelRequest | End[RunResult[Any]]:
        rc = ctx.state
        await rc.event_bus.publish(NodeEnter(node="PrepareTurn", turn=rc.turn))

        agent = rc.agent
        if agent is None:
            await persist_snapshot(rc, current_node="PrepareTurn")
            await rc.event_bus.publish(NodeExit(node="PrepareTurn", next=None))
            return End(terminal_result(rc))

        # Load history from session on fresh runs.
        if agent.session is not None and rc.turn == 0 and not rc.messages:
            session_msgs = await agent.session.get_messages()
            rc.messages.extend(session_msgs)

        # Apply instructions overlay as a leading system message if absent.
        instr = resolve_instructions(rc)
        if instr and not any(m.role == "system" for m in rc.messages):
            rc.messages.insert(
                0,
                Message(
                    role="system",
                    content=[TextBlock(text=instr)],
                    timestamp=datetime.now(UTC),
                ),
            )

        # Append the user prompt (fresh runs only).
        if rc.prompt_text and rc.turn == 0:
            payload = {"prompt": rc.prompt_text, "agent": agent.name}
            payload = await fire_hook(rc, "UserPromptSubmit", payload) or payload
            text_value = payload.get("prompt", rc.prompt_text)
            text = text_value if isinstance(text_value, str) else rc.prompt_text
            user_msg = Message(
                role="user",
                content=[TextBlock(text=text)],
                timestamp=datetime.now(UTC),
            )
            rc.messages.append(user_msg)
            if agent.session is not None:
                await agent.session.add_messages([user_msg])

        if rc.turn == 0:
            await fire_hook(rc, "SessionStart", {"agent": agent.name, "run_id": rc.run_id})

        await persist_snapshot(rc, current_node="PrepareTurn")
        await rc.event_bus.publish(NodeExit(node="PrepareTurn", next="ModelRequest"))
        return ModelRequest()


# ---------------------------------------------------------------------------
# ModelRequest
# ---------------------------------------------------------------------------


@dataclass
class ModelRequest(Node[RunContext[Any], None, RunResult[Any]]):
    """Stream a model response; validate; pick the next node."""

    async def run(
        self,
        ctx: GraphCtx[RunContext[Any], None],
    ) -> ToolDispatch | ModelRequest | DecideNext | End[RunResult[Any]]:
        rc = ctx.state
        rc.turn += 1
        await rc.event_bus.publish(NodeEnter(node="ModelRequest", turn=rc.turn))

        agent = rc.agent
        if agent is None:
            await rc.event_bus.publish(NodeExit(node="ModelRequest", next=None))
            return End(terminal_result(rc))

        tools = await gather_tools(rc)
        visible_tools = [t for t in tools if is_tool_allowed_in_mode(t, rc.permission_mode)]
        settings = agent.model_settings

        final_message: Message | None = None
        final_usage: Usage = Usage()
        try:
            # ``Model.request`` is declared ``async def -> AsyncIterator`` in
            # the Protocol; concrete implementations are async-generator
            # functions whose call returns the iterator directly. Mypy can't
            # bridge the two shapes, so we cast to ``Any``.
            stream: Any = agent.model.request(rc.messages, visible_tools, settings)
            async for ev in stream:
                await rc.event_bus.publish(ev)
                if isinstance(ev, MessageEnd):
                    final_message = ev.final
                    final_usage = ev.usage
                elif isinstance(
                    ev,
                    ModelStart
                    | MessageStart
                    | MessageDelta
                    | ToolCallStart
                    | ToolCallDelta
                    | ToolCallEnd
                    | ModelEnd,
                ):
                    pass
        except Exception as exc:
            await rc.event_bus.publish(
                Error(message=f"model request failed: {exc}", cause=type(exc), recoverable=False)
            )
            raise

        if final_message is None:
            await rc.event_bus.publish(
                Error(message="model emitted no final message", recoverable=False)
            )
            await rc.event_bus.publish(NodeExit(node="ModelRequest", next=None))
            return End(terminal_result(rc))

        rc.usage = rc.usage + final_usage
        rc.messages.append(final_message)
        if agent.session is not None:
            await agent.session.add_messages([final_message])

        tool_calls = list(final_message.tool_calls)

        # Output validation only when no further tool calls.
        if not tool_calls:
            value, err = await maybe_coerce_output(rc, final_message)
            if err is not None and agent.output_type is not str:
                await rc.event_bus.publish(ModelRetryRequest(reason=err))
                rc.messages.append(
                    Message(
                        role="user",
                        content=[TextBlock(text=f"Please retry; {err}")],
                        timestamp=datetime.now(UTC),
                    )
                )
                await persist_snapshot(rc, current_node="ModelRequest")
                await rc.event_bus.publish(NodeExit(node="ModelRequest", next="ModelRequest"))
                return ModelRequest()
            rc.output = value

        await persist_snapshot(rc, current_node="ModelRequest")

        if tool_calls:
            rc.pending_tool_calls = [
                ToolCall(id=tc.id, name=tc.name, arguments=dict(tc.arguments)) for tc in tool_calls
            ]
            await rc.event_bus.publish(NodeExit(node="ModelRequest", next="ToolDispatch"))
            return ToolDispatch()
        await rc.event_bus.publish(NodeExit(node="ModelRequest", next="DecideNext"))
        return DecideNext()


# ---------------------------------------------------------------------------
# ToolDispatch
# ---------------------------------------------------------------------------


@dataclass
class ToolDispatch(Node[RunContext[Any], None, RunResult[Any]]):
    """Approval gate + concurrent dispatch of pending tool calls."""

    async def run(
        self,
        ctx: GraphCtx[RunContext[Any], None],
    ) -> DecideNext | End[RunResult[Any]]:
        rc = ctx.state
        await rc.event_bus.publish(NodeEnter(node="ToolDispatch", turn=rc.turn))

        agent = rc.agent
        if agent is None or not rc.pending_tool_calls:
            await persist_snapshot(rc, current_node="ToolDispatch")
            await rc.event_bus.publish(NodeExit(node="ToolDispatch", next="DecideNext"))
            return DecideNext()

        calls = list(rc.pending_tool_calls)
        rc.pending_tool_calls = []

        # Resolve tools for each call.
        resolved: list[tuple[ToolCall, Tool | None]] = [
            (call, await find_tool(rc, call.name)) for call in calls
        ]

        # Permission-mode + missing-tool gate.
        mode_errors: list[tuple[ToolCall, str]] = []
        ok: list[tuple[ToolCall, Tool]] = []
        for call, t in resolved:
            if t is None:
                mode_errors.append((call, f"no tool named {call.name!r}"))
                continue
            if not is_tool_allowed_in_mode(t, rc.permission_mode):
                mode_errors.append(
                    (call, f"tool {call.name!r} not allowed in {rc.permission_mode!r} mode")
                )
                continue
            ok.append((call, t))

        # Approval gate.
        approvals_pending: list[ApprovalRequest] = []
        approved: list[tuple[ToolCall, Tool]] = []
        denied: list[tuple[ToolCall, str]] = []
        for call, t in ok:
            if not needs_approval(t, rc):
                approved.append((call, t))
                continue
            decision = rc.resolved_approvals.get(call.id)
            if decision is None:
                approvals_pending.append(
                    ApprovalRequest(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        arguments=dict(call.arguments),
                        requested_at=datetime.now(UTC),
                    )
                )
                continue
            if decision.approve:
                approved.append((call, t))
            else:
                denied.append((call, decision.rationale or "denied by user"))

        if approvals_pending:
            rc.pending_approvals = approvals_pending
            rc.pending_tool_calls = calls
            await rc.event_bus.publish(ApprovalRequested(requests=approvals_pending))
            interruption = make_interruption(rc)
            await persist_snapshot(rc, current_node="ToolDispatch")
            await rc.event_bus.publish(NodeExit(node="ToolDispatch", next=None, interrupted=True))
            return End(
                RunResult(
                    output=None,
                    messages=list(rc.messages),
                    pending_approvals=interruption.pending_approvals,
                    run_state=interruption.run_state,
                    usage=rc.usage.model_copy(),
                )
            )

        # Execute approved + record mode-rejections and denials as errors.
        results: dict[str, ToolResult] = {}
        for call, reason in mode_errors + denied:
            results[call.id] = ToolResult(content=[TextBlock(text=reason)], error=reason)
            await rc.event_bus.publish(
                ToolExecStart(
                    tool_call_id=call.id, tool_name=call.name, arguments=dict(call.arguments)
                )
            )
            await rc.event_bus.publish(
                ToolExecEnd(tool_call_id=call.id, result=results[call.id], error=reason)
            )

        parallel = [(c, t) for (c, t) in approved if is_concurrency_safe(t, rc)]
        serial = [(c, t) for (c, t) in approved if not is_concurrency_safe(t, rc)]

        if parallel:
            async with asyncio.TaskGroup() as tg:
                tasks: list[tuple[ToolCall, asyncio.Task[ToolResult]]] = [
                    (call, tg.create_task(execute_tool(rc, call, t))) for call, t in parallel
                ]
            for call, task in tasks:
                results[call.id] = task.result()

        for call, t in serial:
            results[call.id] = await execute_tool(rc, call, t)

        result_blocks = [
            ToolResultBlock(tool_call_id=call.id, content=content_to_text(results[call.id]))
            for call in calls
            if call.id in results
        ]
        if result_blocks:
            tool_msg = Message(
                role="tool",
                content=list(result_blocks),
                timestamp=datetime.now(UTC),
            )
            rc.messages.append(tool_msg)
            if agent.session is not None:
                await agent.session.add_messages([tool_msg])

        rc.pending_approvals = []
        rc.resolved_approvals = {}

        await persist_snapshot(rc, current_node="ToolDispatch")
        await rc.event_bus.publish(NodeExit(node="ToolDispatch", next="DecideNext"))
        return DecideNext()


# ---------------------------------------------------------------------------
# DecideNext
# ---------------------------------------------------------------------------


@dataclass
class DecideNext(Node[RunContext[Any], None, RunResult[Any]]):
    """Run history processors; terminate or loop back."""

    async def run(
        self,
        ctx: GraphCtx[RunContext[Any], None],
    ) -> ModelRequest | End[RunResult[Any]]:
        rc = ctx.state
        await rc.event_bus.publish(NodeEnter(node="DecideNext", turn=rc.turn))

        agent = rc.agent
        if agent is None:
            await persist_snapshot(rc, current_node="DecideNext")
            await rc.event_bus.publish(NodeExit(node="DecideNext", next=None))
            return End(terminal_result(rc))

        last = last_assistant(rc.messages)
        finished = last is not None and not last.has_tool_call() and rc.output is not None
        if (
            not finished
            and last is not None
            and not last.has_tool_call()
            and agent.output_type is str
        ):
            rc.output = last.text
            finished = True

        if finished:
            await persist_snapshot(rc, current_node="DecideNext")
            await fire_hook(rc, "Stop", {"agent": agent.name})
            await fire_hook(rc, "SessionEnd", {"agent": agent.name, "run_id": rc.run_id})
            await rc.event_bus.publish(NodeExit(node="DecideNext", next=None))
            return End(terminal_result(rc))

        for proc in agent.history_processors:
            name = processor_name(proc)
            before = len(rc.messages)
            await rc.event_bus.publish(CompactionStart(processor_name=name, messages_before=before))
            payload = await fire_hook(rc, "PreCompact", {"processor": name, "messages": before})
            if payload is None:
                await rc.event_bus.publish(
                    CompactionEnd(processor_name=name, messages_after=before, usage_added=Usage())
                )
                continue
            try:
                rc.messages = await apply_processor(proc, rc.messages, rc)
            except Exception as exc:
                await rc.event_bus.publish(
                    Error(
                        message=f"history processor {name!r} failed: {exc}",
                        cause=type(exc),
                        recoverable=True,
                    )
                )
            after = len(rc.messages)
            await rc.event_bus.publish(
                CompactionEnd(processor_name=name, messages_after=after, usage_added=Usage())
            )
            await fire_hook(rc, "PostCompact", {"processor": name, "messages": after})

        await persist_snapshot(rc, current_node="DecideNext")
        await rc.event_bus.publish(NodeExit(node="DecideNext", next="ModelRequest"))
        return ModelRequest()


__all__ = [
    "DecideNext",
    "ModelRequest",
    "PrepareTurn",
    "ToolDispatch",
]
