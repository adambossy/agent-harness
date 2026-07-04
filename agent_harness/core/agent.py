"""The ``Agent`` orchestrator (Wave 4 / Wave 5).

The public face of the harness — users call ``Agent.run``, ``Agent.iter``,
``Agent.stream``, ``Agent.as_tool``, and ``Agent.resume``. Every method
delegates to the 4-node typed graph in :mod:`.loop`; the agent itself
holds no mutable state (AG9).

Imports are Protocols-only (LP5 / S11): no concrete ``Provider`` /
``Sandbox`` / ``Session`` types ever appear here.

Example:
    >>> Agent.__name__
    'Agent'
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Generic, TypeVar, cast

from .errors import ConfigError
from .events import (
    AgentEnd,
    AgentStart,
    EventBus,
    InMemoryEventBus,
    RunEnd,
    RunStart,
    SubagentStart,
    SubagentStop,
)
from .graph import GraphRunResult, build_graph, iter_graph, run_graph
from .history import HistoryProcessor
from .hooks import HookRegistry
from .memory import LongTermMemory, Session
from .models import Message, Model, ModelSettings, TextBlock, UsagePricer
from .run_context import (
    RunContext,
    RunResult,
    get_current_run_ctx,
    is_tool_allowed_in_mode,
    make_interruption,
    snapshot_from_ctx,
)
from .run_state import Approval, RunStateSnapshot
from .sandbox import Sandbox
from .tools import Tool, ToolPolicy, ToolResult
from .toolsets import Toolset

Deps = TypeVar("Deps")
Out = TypeVar("Out")


class Agent(Generic[Deps, Out]):
    """Composable agent — public entry point.

    Constructed from a small set of named parts (AG8). Methods:

    - :meth:`run` — execute to completion or interruption (AG1).
    - :meth:`iter` — yield each node before execution (AG2).
    - :meth:`stream` — yield bus events (AG3).
    - :meth:`as_tool` — wrap as a callable tool (AG4 / S9).
    - :meth:`resume` — re-enter from a snapshot (AG6 / S10).

    Example:
        >>> from tests.fakes import FakeModel, FakeTurn
        >>> agent = Agent(
        ...     name="demo",
        ...     model=FakeModel(script=[FakeTurn(text="ok")]),
        ...     toolsets=[],
        ... )
        >>> agent.name
        'demo'
    """

    name: str
    instructions: str | Callable[[RunContext[Deps]], str]
    model: Model
    toolsets: list[Toolset]
    session: Session | None
    persist_session: bool
    sandbox: Sandbox | None
    long_term_memory: LongTermMemory | None
    history_processors: list[HistoryProcessor]
    output_type: type[Out] | type[str]
    deps_type: type[Deps] | None
    hooks: HookRegistry | None
    model_settings: ModelSettings
    output_validator: Callable[[Any], Awaitable[Out] | Out] | None
    usage_pricer: UsagePricer | None

    def __init__(
        self,
        *,
        name: str,
        model: Model,
        toolsets: list[Toolset] | None = None,
        instructions: str | Callable[[RunContext[Deps]], str] = "",
        session: Session | None = None,
        persist_session: bool = True,
        sandbox: Sandbox | None = None,
        long_term_memory: LongTermMemory | None = None,
        history_processors: list[HistoryProcessor] | None = None,
        output_type: type[Out] | type[str] = str,
        deps_type: type[Deps] | None = None,
        hooks: HookRegistry | None = None,
        model_settings: ModelSettings | None = None,
        output_validator: Callable[[Any], Awaitable[Out] | Out] | None = None,
        usage_pricer: UsagePricer | None = None,
    ) -> None:
        if not name:
            raise ConfigError("Agent.name must be non-empty", context={"name": name})
        self.name = name
        self.instructions = instructions
        self.model = model
        self.toolsets = list(toolsets) if toolsets is not None else []
        self.session = session
        # When False, the run loop still *reads* prior history from ``session``
        # but does not *write* messages or run-state back to it — i.e. no
        # persistence to the session store. Lets a caller own persistence
        # elsewhere without double-writing.
        self.persist_session = persist_session
        self.sandbox = sandbox
        self.long_term_memory = long_term_memory
        self.history_processors = list(history_processors) if history_processors is not None else []
        self.output_type = output_type
        self.deps_type = deps_type
        self.hooks = hooks
        self.model_settings = model_settings or ModelSettings()
        self.output_validator = output_validator
        # Host-supplied token→cost hook. When set, the loop publishes a
        # ``ModelUsage`` event after each model response (opt-in: no pricer,
        # no cost events — tokens still ride ``MessageEnd``).
        self.usage_pricer = usage_pricer

    # ----- run -------------------------------------------------------------

    async def run(
        self,
        prompt: str | list[Message],
        *,
        deps: Deps | None = None,
        run_state: RunStateSnapshot | None = None,
        approvals: list[Approval] | None = None,
        event_bus: EventBus | None = None,
    ) -> RunResult[Out]:
        """Execute to completion or paused-on-approval (AG1).

        Example:
            >>> import asyncio
            >>> from tests.fakes import FakeModel, FakeTurn
            >>> a = Agent(name="d", model=FakeModel(script=[FakeTurn(text="hi")]), toolsets=[])
            >>> asyncio.run(a.run("hello")).output
            'hi'
        """
        from .loop import DecideNext, ModelRequest, PrepareTurn, ToolDispatch

        ctx, bus_owned = self._build_context(prompt, deps, run_state, approvals, event_bus)
        graph = build_graph(
            [PrepareTurn, ModelRequest, ToolDispatch, DecideNext],
            name=f"{self.name}-loop",
        )
        start_node = _start_node_for(run_state)
        t0 = time.monotonic()
        await ctx.event_bus.publish(
            RunStart(run_id=ctx.run_id, agent_name=self.name, prompt=ctx.prompt_text)
        )
        await ctx.event_bus.publish(AgentStart(agent_name=self.name))
        graph_result: GraphRunResult[Any, Any] = await run_graph(graph, start_node, state=ctx)
        result = cast(RunResult[Out], graph_result.output)
        await ctx.event_bus.publish(AgentEnd(agent_name=self.name))
        duration_ms = int((time.monotonic() - t0) * 1000)
        await ctx.event_bus.publish(
            RunEnd(
                run_id=ctx.run_id,
                result=result,
                usage=result.usage,
                duration_ms=duration_ms,
            )
        )
        if bus_owned:
            await ctx.event_bus.close()
        return result

    async def iter(
        self,
        prompt: str | list[Message],
        *,
        deps: Deps | None = None,
        run_state: RunStateSnapshot | None = None,
        approvals: list[Approval] | None = None,
        event_bus: EventBus | None = None,
    ) -> AsyncIterator[Any]:
        """Yield each node (or :class:`End`) as the loop executes (AG2)."""
        from .loop import DecideNext, ModelRequest, PrepareTurn, ToolDispatch

        ctx, bus_owned = self._build_context(prompt, deps, run_state, approvals, event_bus)
        graph = build_graph(
            [PrepareTurn, ModelRequest, ToolDispatch, DecideNext],
            name=f"{self.name}-loop",
        )
        start_node = _start_node_for(run_state)
        await ctx.event_bus.publish(
            RunStart(run_id=ctx.run_id, agent_name=self.name, prompt=ctx.prompt_text)
        )
        await ctx.event_bus.publish(AgentStart(agent_name=self.name))
        async with iter_graph(graph, start_node, state=ctx) as run:
            async for step in run:
                yield step
        await ctx.event_bus.publish(AgentEnd(agent_name=self.name))
        if bus_owned:
            await ctx.event_bus.close()

    async def stream(
        self,
        prompt: str | list[Message],
        *,
        deps: Deps | None = None,
        run_state: RunStateSnapshot | None = None,
        approvals: list[Approval] | None = None,
        event_bus: EventBus | None = None,
    ) -> AsyncIterator[Any]:
        """Yield each :class:`Event` published during the run (AG3)."""
        bus_owned = event_bus is None
        bus = event_bus if event_bus is not None else InMemoryEventBus()
        subscription = bus.subscribe()

        async def _drive() -> RunResult[Out]:
            try:
                return await self.run(
                    prompt,
                    deps=deps,
                    run_state=run_state,
                    approvals=approvals,
                    event_bus=bus,
                )
            finally:
                if bus_owned:
                    with contextlib.suppress(Exception):
                        await bus.close()

        task = asyncio.create_task(_drive())
        try:
            async for ev in subscription:
                yield ev
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            else:
                exc = task.exception()
                if exc is not None:
                    raise exc

    # ----- as_tool / resume ------------------------------------------------

    def as_tool(
        self,
        *,
        name: str,
        description: str,
        deps_map: Callable[[dict[str, Any]], Deps] | None = None,
    ) -> Tool:
        """Wrap this agent as a :class:`Tool` for another agent (AG4 / S9).

        Wave 5 hardens nested-approval propagation (decision #4):

        * Nested events are republished on the **parent's** :class:`EventBus`
          (AT2) — bookended by :class:`SubagentStart` / :class:`SubagentStop`
          and with the child's own ``RunStart`` / ``RunEnd`` / ``AgentStart``
          / ``AgentEnd`` filtered (the parent's loop already emits its own).
        * Nested ``pending_approvals`` raise
          :exc:`agent_harness.core.subagents.NestedInterruption`, which the
          parent's :class:`ToolDispatch` catches and rolls into the parent's
          own ``pending_approvals`` (AT3).

        Example:
            >>> from tests.fakes import FakeModel, FakeTurn
            >>> child = Agent(name="c", model=FakeModel(script=[FakeTurn(text="hi")]), toolsets=[])
            >>> child.as_tool(name="run_child", description="child").name
            'run_child'
        """
        agent = self

        async def _invoke(**kwargs: Any) -> ToolResult:
            # Local imports keep the agent↔subagents cycle out of module init.
            from .subagents import NestedInterruption, RepublishingBus

            mapped: Deps | None = deps_map(kwargs) if deps_map is not None else None
            prompt = kwargs.get("prompt", "")

            parent_ctx = get_current_run_ctx()
            parent_bus: EventBus | None = parent_ctx.event_bus if parent_ctx else None
            parent_name = parent_ctx.agent.name if parent_ctx and parent_ctx.agent else ""
            tool_call_id = kwargs.get("_tool_call_id", "") or ""

            if parent_bus is not None:
                await parent_bus.publish(
                    SubagentStart(
                        parent_agent_name=parent_name,
                        child_agent_name=agent.name,
                        tool_call_id=tool_call_id,
                    )
                )

            child_bus: EventBus = (
                cast(EventBus, RepublishingBus(parent_bus))
                if parent_bus is not None
                else InMemoryEventBus()
            )
            try:
                result = await agent.run(prompt, deps=mapped, event_bus=child_bus)
            finally:
                if parent_bus is not None:
                    await parent_bus.publish(
                        SubagentStop(
                            parent_agent_name=parent_name,
                            child_agent_name=agent.name,
                            tool_call_id=tool_call_id,
                        )
                    )
                if isinstance(child_bus, InMemoryEventBus):
                    await child_bus.close()

            # Roll the child's usage into the parent's (AT4).
            if parent_ctx is not None:
                parent_ctx.usage = parent_ctx.usage + result.usage

            if result.pending_approvals:
                raise NestedInterruption(
                    child_agent_name=agent.name,
                    tool_call_id=tool_call_id,
                    pending_approvals=result.pending_approvals,
                    child_run_state=result.run_state,
                )
            text = str(result.output) if result.output is not None else ""
            return ToolResult(content=[TextBlock(text=text)])

        return Tool(
            name=name,
            description=description,
            schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Prompt for the subagent."}
                },
                "required": ["prompt"],
            },
            policy=ToolPolicy(is_concurrency_safe=False),
            fn=_invoke,
        )

    @classmethod
    async def resume(
        cls,
        snapshot: RunStateSnapshot,
        *,
        agent: Agent[Any, Any],
        approvals: list[Approval] | None = None,
        event_bus: EventBus | None = None,
    ) -> RunResult[Any]:
        """Resume a paused run from ``snapshot`` (AG6 / S10).

        The framework auto-persisted ``snapshot`` via
        :meth:`Session.add_run_state` before returning the original
        :class:`Interruption` (decision #13). The caller passes the same
        :class:`Agent` instance (config-only resume) plus user decisions.

        Example:
            >>> Agent.resume.__name__
            'resume'
        """
        return await agent.run(
            prompt=snapshot.messages[-1].text if snapshot.messages else "",
            run_state=snapshot,
            approvals=approvals,
            event_bus=event_bus,
        )

    # ----- internal --------------------------------------------------------

    def _build_context(
        self,
        prompt: str | list[Message],
        deps: Deps | None,
        run_state: RunStateSnapshot | None,
        approvals: list[Approval] | None,
        event_bus: EventBus | None,
    ) -> tuple[RunContext[Deps], bool]:
        """Construct the :class:`RunContext` for this run.

        Returns ``(ctx, bus_owned)``: ``bus_owned`` is True when the agent
        allocated the bus (and is therefore responsible for closing it).
        """
        bus_owned = event_bus is None
        bus = event_bus if event_bus is not None else InMemoryEventBus()

        if run_state is not None:
            run_id = run_state.run_id
            messages = list(run_state.messages)
            usage = run_state.usage.model_copy()
            turn = run_state.turn
            permission_mode = run_state.permission_mode
            pre_plan_mode = run_state.pre_plan_mode
            prompt_text = ""
            pending = list(run_state.pending_tool_calls)
            pending_approvals = list(run_state.pending_approvals)
        else:
            run_id = f"run_{uuid.uuid4().hex[:12]}"
            messages = []
            from .models import Usage as _Usage

            usage = _Usage()
            turn = 0
            permission_mode = "default"
            pre_plan_mode = None
            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                messages = list(prompt)
                prompt_text = messages[-1].text if messages else ""
            pending = []
            pending_approvals = []

        resolved: dict[str, Approval] = {a.tool_call_id: a for a in approvals or []}

        ctx = RunContext[Deps](
            run_id=run_id,
            agent=self,
            deps=deps,
            messages=messages,
            usage=usage,
            sandbox=self.sandbox,
            long_term_memory=self.long_term_memory,
            event_bus=bus,
            turn=turn,
            pending_tool_calls=pending,
            pending_approvals=pending_approvals,
            permission_mode=permission_mode,
            pre_plan_mode=pre_plan_mode,
            prompt_text=prompt_text,
            output=None,
            resolved_approvals=resolved,
        )
        return ctx, bus_owned


def _start_node_for(run_state: RunStateSnapshot | None) -> Any:
    """Pick the entry node — fresh starts at :class:`PrepareTurn`; resumed
    runs re-enter at the snapshot's recorded ``current_node``."""
    from .loop import DecideNext, ModelRequest, PrepareTurn, ToolDispatch

    if run_state is None:
        return PrepareTurn()
    table: dict[str, Any] = {
        "PrepareTurn": PrepareTurn(),
        "ModelRequest": ModelRequest(),
        "ToolDispatch": ToolDispatch(),
        "DecideNext": DecideNext(),
    }
    return table.get(run_state.current_node, PrepareTurn())


__all__ = [
    "Agent",
    "RunContext",
    "RunResult",
    "is_tool_allowed_in_mode",
    "make_interruption",
    "snapshot_from_ctx",
]
