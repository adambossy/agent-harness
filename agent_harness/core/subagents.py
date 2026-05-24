"""Subagent support — declarative defs, worktree isolation, ``as_tool`` glue.

Wave 5. Decision #4 (open-questions.md) ships only the ``as_tool`` seam; no
``Handoff`` / control transfer. The Wave-5-unique pieces live here:

* :class:`AgentDefinition` — MD-frontmatter record (SD1 / SD2).
* :class:`SubagentRegistry` — loads project + user definitions (SD1).
* :func:`build_agent_tool` — materializes a definition into a :class:`Tool`
  (AT1).
* :exc:`NestedInterruption` — sentinel raised by ``as_tool`` when the child
  returns ``pending_approvals``; caught by ``ToolDispatch`` (AT3).
* :func:`republish_event_for_parent` — forwards child events onto the
  parent's bus (AT2).
* :func:`create_worktree` / :func:`remove_worktree` — git-worktree helpers
  for ``isolation: worktree`` (IS1 / IS2).

Example:
    >>> SubagentRegistry.load(project_root=None, user_root=None).list_agents()
    []
"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from .errors import ConfigError, ToolError
from .events import (
    AgentEnd,
    AgentStart,
    Event,
    EventBus,
    RunEnd,
    RunStart,
)
from .models import TextBlock
from .run_state import ApprovalRequest, RunStateSnapshot
from .tools import Tool, ToolPolicy, ToolResult

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .agent import Agent


# --- Constants -------------------------------------------------------------

PROJECT_AGENTS_DIRNAME: str = ".agent/agents"
"""Project-scoped subagent definitions root."""

USER_AGENTS_HOME_SUBPATH: str = ".agent_harness/agents"
"""User-scoped subagent definitions root."""

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<front>.*?)\n---\s*(?:\n(?P<body>.*))?\Z",
    re.DOTALL,
)
_NAME_RE = re.compile(r"\A[a-zA-Z0-9_][a-zA-Z0-9_\-]*\Z")

IsolationMode = Literal["none", "worktree", "remote"]
"""How a subagent's filesystem is isolated from the parent (IS1)."""


# --- NestedInterruption sentinel ------------------------------------------


class NestedInterruption(Exception):  # noqa: N818 — control-flow sentinel, not an error
    """Raised by the ``as_tool`` wrapper when the child has pending approvals.

    Caught by the parent's :class:`ToolDispatch`, which rolls the child's
    ``pending_approvals`` into the parent's ``pending_approvals`` (AT3).

    Example:
        >>> NestedInterruption(
        ...     child_agent_name="c",
        ...     tool_call_id="t",
        ...     pending_approvals=[],
        ...     child_run_state=None,
        ... ).child_agent_name
        'c'
    """

    def __init__(
        self,
        *,
        child_agent_name: str,
        tool_call_id: str,
        pending_approvals: list[ApprovalRequest],
        child_run_state: RunStateSnapshot | None,
    ) -> None:
        super().__init__(
            f"subagent {child_agent_name!r} paused awaiting "
            f"{len(pending_approvals)} approval(s)"
        )
        self.child_agent_name = child_agent_name
        self.tool_call_id = tool_call_id
        self.pending_approvals = list(pending_approvals)
        self.child_run_state = child_run_state


# --- AgentDefinition + parsing --------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Declarative subagent loaded from MD-with-frontmatter (SD1 / SD2).

    Example:
        >>> AgentDefinition(name="d", description="x", body_path=Path("/x.md")).name
        'd'
    """

    name: str
    description: str = ""
    model: str | None = None
    effort: str | None = None
    permission_mode: str | None = None
    mcp_servers: tuple[str, ...] = ()
    required_mcp_servers: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    max_turns: int | None = None
    skills: tuple[str, ...] = ()
    memory: str | None = None
    background: bool = False
    isolation: IsolationMode = "none"
    initial_prompt: str = ""
    body_path: Path = field(default_factory=lambda: Path("/dev/null"))
    source: str = "project"


def _coerce_str_tuple(value: Any, key: str, path: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ConfigError(
            f"subagent file {path}: `{key}` must be a list of strings",
            context={"path": str(path), "key": key, "got": repr(value)},
        )
    return tuple(value)


def _bad(path: Path, msg: str, **ctx: Any) -> ConfigError:
    return ConfigError(f"subagent file {path}: {msg}", context={"path": str(path), **ctx})


def _parse_definition_file(path: Path, *, source: str) -> AgentDefinition:
    """Parse a single ``<root>/<name>.md`` into an :class:`AgentDefinition`."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _bad(path, f"failed to read: {exc}") from exc
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise _bad(path, "missing a YAML frontmatter block")
    try:
        loaded = yaml.safe_load(match.group("front"))
    except yaml.YAMLError as exc:
        raise _bad(path, f"malformed YAML frontmatter: {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise _bad(path, f"frontmatter must be a mapping; got {type(loaded).__name__}")
    data: dict[str, Any] = loaded

    name = data.get("name", path.stem)
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise _bad(path, f"`name` must match {_NAME_RE.pattern}", got=repr(name))
    description = data.get("description", "")
    if description is not None and not isinstance(description, str):
        raise _bad(path, "`description` must be a string")
    isolation = data.get("isolation", "none")
    if isolation not in ("none", "worktree", "remote"):
        raise _bad(
            path,
            "`isolation` must be one of 'none' | 'worktree' | 'remote'",
            got=repr(isolation),
        )
    max_turns = data.get("max_turns")
    if max_turns is not None and not isinstance(max_turns, int):
        raise _bad(path, "`max_turns` must be an int", got=repr(max_turns))
    background = data.get("background", False)
    if not isinstance(background, bool):
        raise _bad(path, "`background` must be a bool", got=repr(background))

    return AgentDefinition(
        name=name,
        description=(description or "").strip(),
        model=data.get("model"),
        effort=data.get("effort"),
        permission_mode=data.get("permission_mode"),
        mcp_servers=_coerce_str_tuple(data.get("mcp_servers"), "mcp_servers", path),
        required_mcp_servers=_coerce_str_tuple(
            data.get("required_mcp_servers"), "required_mcp_servers", path
        ),
        tools=_coerce_str_tuple(data.get("tools"), "tools", path),
        disallowed_tools=_coerce_str_tuple(data.get("disallowed_tools"), "disallowed_tools", path),
        max_turns=max_turns,
        skills=_coerce_str_tuple(data.get("skills"), "skills", path),
        memory=data.get("memory"),
        background=background,
        isolation=isolation,
        initial_prompt=str(data.get("initial_prompt") or ""),
        body_path=path,
        source=source,
    )


# --- SubagentRegistry ------------------------------------------------------


@dataclass(slots=True)
class SubagentRegistry:
    """Discovered subagent definitions indexed by name (SD1 / SD4).

    Project entries shadow user entries with the same name.

    Example:
        >>> SubagentRegistry().list_agents()
        []
    """

    agents: dict[str, AgentDefinition] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        *,
        project_root: Path | None = None,
        user_root: Path | None = None,
    ) -> SubagentRegistry:
        """Walk both roots and build the registry (non-recursive discovery)."""
        reg = cls()
        if user_root is not None:
            reg._ingest(user_root / USER_AGENTS_HOME_SUBPATH, source="user")
        if project_root is not None:
            reg._ingest(project_root / PROJECT_AGENTS_DIRNAME, source="project")
        return reg

    def _ingest(self, root: Path, *, source: str) -> None:
        if not root.is_dir():
            return
        for entry in sorted(root.iterdir()):
            if entry.is_file() and entry.suffix == ".md":
                definition = _parse_definition_file(entry, source=source)
                self.agents[definition.name] = definition

    def list_agents(self) -> list[AgentDefinition]:
        """All loaded definitions, name-sorted."""
        return sorted(self.agents.values(), key=lambda d: d.name)

    def get(self, name: str) -> AgentDefinition:
        """Look up by name; raise :class:`ToolError` if unknown."""
        try:
            return self.agents[name]
        except KeyError as exc:
            raise ToolError(
                f"no subagent named {name!r}",
                context={"requested": name, "available": sorted(self.agents)},
            ) from exc


# --- Event republishing ----------------------------------------------------


async def republish_event_for_parent(parent_bus: EventBus, event: Event) -> None:
    """Forward a child event onto the parent's bus, stripping run bookends.

    ``RunStart`` / ``RunEnd`` / ``AgentStart`` / ``AgentEnd`` are dropped
    because the parent's loop emits its own bookends and the nested run is
    bracketed by :class:`SubagentStart` / :class:`SubagentStop` (AT2).
    """
    if isinstance(event, RunStart | RunEnd | AgentStart | AgentEnd):
        return
    await parent_bus.publish(event)


class _EmptyAsyncIterator:
    """Empty stream for :class:`RepublishingBus.subscribe` (no events ever)."""

    def __aiter__(self) -> _EmptyAsyncIterator:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class RepublishingBus:
    """``EventBus``-shaped forwarder used by :meth:`Agent.as_tool` (AT2).

    Wraps the parent's :class:`EventBus` and republishes the child's events
    via :func:`republish_event_for_parent` (which filters run/agent
    bookends). ``subscribe`` returns an empty stream — callers should
    subscribe to the parent bus directly.

    Example:
        >>> RepublishingBus(None).maxsize
        1
    """

    maxsize: int = 1

    def __init__(self, parent: EventBus | None) -> None:
        self._parent: EventBus | None = parent

    async def publish(self, event: Any) -> None:
        if self._parent is None:
            return
        await republish_event_for_parent(self._parent, event)

    def subscribe(self, *, from_event: int | None = None) -> AsyncIterator[Any]:
        del from_event
        return _EmptyAsyncIterator()

    async def close(self) -> None:
        return


# --- Worktree isolation ----------------------------------------------------


@dataclass(slots=True)
class WorktreeHandle:
    """Bookkeeping for a worktree created by :func:`create_worktree`.

    Example:
        >>> WorktreeHandle(
        ...     path=Path("/tmp"),
        ...     branch="",
        ...     parent_repo=Path("/tmp"),
        ...     active=False,
        ... ).active
        False
    """

    path: Path
    branch: str
    parent_repo: Path
    active: bool


def create_worktree(parent_repo: str | Path, *, name_hint: str = "subagent") -> WorktreeHandle:
    """Create a fresh git worktree off the parent's HEAD.

    Returns an *inactive* handle (``path == parent_repo``) when git is
    unavailable or ``parent_repo`` isn't a git repo — callers treat that as
    "share parent's cwd" so the higher-level contract still holds.
    """
    parent_path = Path(parent_repo).resolve()
    inactive = WorktreeHandle(path=parent_path, branch="", parent_repo=parent_path, active=False)
    if shutil.which("git") is None or not parent_path.is_dir():
        return inactive
    probe = subprocess.run(
        ["git", "-C", str(parent_path), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return inactive

    unique = f"{name_hint}-{uuid.uuid4().hex[:8]}"
    worktree_path = parent_path / ".agent_harness" / "worktrees" / unique
    branch = f"agent-harness/{unique}"
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "git",
            "-C",
            str(parent_path),
            "worktree",
            "add",
            "-b",
            branch,
            str(worktree_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return inactive
    return WorktreeHandle(path=worktree_path, branch=branch, parent_repo=parent_path, active=True)


def remove_worktree(handle: WorktreeHandle, *, force: bool = True) -> bool:
    """Tear down a worktree created via :func:`create_worktree`."""
    if not handle.active:
        return False
    args = ["git", "-C", str(handle.parent_repo), "worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(handle.path))
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return result.returncode == 0


# --- build_agent_tool ------------------------------------------------------


def build_agent_tool(definition: AgentDefinition, parent_agent: Agent[Any, Any]) -> Tool:
    """Materialize an :class:`AgentDefinition` into a callable :class:`Tool`.

    The parent's model invokes the tool by name; the wrapper spawns a child
    :class:`Agent` (inheriting the parent's :class:`Model` — TODO(v0.0.2):
    respect ``definition.model``) and dispatches via :meth:`Agent.as_tool`
    so nested-event republishing and nested-approval propagation are reused.

    Example:
        >>> build_agent_tool.__name__
        'build_agent_tool'
    """
    from .agent import Agent  # local import — avoids the agent↔subagents cycle

    if definition.required_mcp_servers and not parent_agent.toolsets:
        # SD3: fail-fast when a hard dependency is missing.
        raise ConfigError(
            f"subagent {definition.name!r} requires MCP servers "
            f"{list(definition.required_mcp_servers)} but the parent agent "
            "declares no toolsets",
            context={"subagent": definition.name},
        )

    # TODO(v0.0.2): fork-subagent cache reuse (AT5); honor definition.model.
    child: Agent[Any, Any] = Agent(
        name=definition.name,
        model=parent_agent.model,
        toolsets=list(parent_agent.toolsets),
        instructions=definition.initial_prompt,
        sandbox=parent_agent.sandbox,
        long_term_memory=parent_agent.long_term_memory,
        history_processors=list(parent_agent.history_processors),
        hooks=parent_agent.hooks,
        model_settings=parent_agent.model_settings,
    )
    base = child.as_tool(name=definition.name, description=definition.description)
    return Tool(
        name=base.name,
        description=base.description,
        schema=base.schema,
        policy=ToolPolicy(is_concurrency_safe=False, always_load=True),
        fn=base.fn,
    )


# --- Misc helpers ----------------------------------------------------------


def make_paused_tool_result(child_agent_name: str, pending_count: int) -> ToolResult:
    """Build the informational :class:`ToolResult` for a paused subagent.

    Example:
        >>> make_paused_tool_result("c", 2).metadata["pending_approvals"]
        2
    """
    return ToolResult(
        content=[TextBlock(text=f"subagent {child_agent_name!r} paused awaiting approval")],
        error=f"subagent {child_agent_name!r} paused awaiting approval",
        metadata={"pending_approvals": pending_count, "child_agent_name": child_agent_name},
    )


def iter_definition_paths(roots: Iterable[Path]) -> list[Path]:
    """Yield every ``<root>/<name>.md`` reachable from ``roots``.

    Example:
        >>> iter_definition_paths([])
        []
    """
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if entry.is_file() and entry.suffix == ".md":
                out.append(entry)
    return out


__all__ = [
    "AgentDefinition",
    "IsolationMode",
    "NestedInterruption",
    "PROJECT_AGENTS_DIRNAME",
    "SubagentRegistry",
    "USER_AGENTS_HOME_SUBPATH",
    "WorktreeHandle",
    "build_agent_tool",
    "create_worktree",
    "iter_definition_paths",
    "make_paused_tool_result",
    "remove_worktree",
    "republish_event_for_parent",
]
