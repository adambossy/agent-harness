"""Unit tests for :mod:`agent_harness.core.subagents` (Wave 5).

Covers:

* :class:`AgentDefinition` construction.
* :class:`SubagentRegistry.load` — file discovery + frontmatter parsing
  (MD1-MD4 / SD1-SD4).
* :func:`build_agent_tool` — definition → :class:`Tool` materialization.
* :func:`create_worktree` / :func:`remove_worktree` — git-worktree isolation
  hooks (IS1 / IS2).
* :class:`NestedInterruption` — sentinel for nested approval propagation.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from agent_harness.core.agent import Agent
from agent_harness.core.errors import ConfigError, ToolError
from agent_harness.core.models import Model
from agent_harness.core.run_state import ApprovalRequest
from agent_harness.core.subagents import (
    AgentDefinition,
    NestedInterruption,
    RepublishingBus,
    SubagentRegistry,
    WorktreeHandle,
    build_agent_tool,
    create_worktree,
    iter_definition_paths,
    make_paused_tool_result,
    remove_worktree,
    republish_event_for_parent,
)
from tests.fakes import FakeModel, FakeTurn

_GIT = shutil.which("git")


def _model(*turns: FakeTurn) -> Model:
    """Build a FakeModel cast to ``Model`` for mypy."""
    return cast(Model, FakeModel(script=list(turns)))


# --- AgentDefinition -------------------------------------------------------


def test_agent_definition_defaults() -> None:
    d = AgentDefinition(name="x", description="d", body_path=Path("/x.md"))
    assert d.name == "x"
    assert d.isolation == "none"
    assert d.background is False
    assert d.mcp_servers == ()


def test_agent_definition_isolation_literal() -> None:
    """``isolation`` is a closed-set Literal — only the 3 modes allowed."""
    d = AgentDefinition(name="x", description="d", isolation="worktree", body_path=Path("/x.md"))
    assert d.isolation == "worktree"


# --- SubagentRegistry ------------------------------------------------------


def _write_def(root: Path, name: str, body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_registry_loads_project_definitions(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    _write_def(
        project / ".agent" / "agents",
        "summarizer",
        "---\nname: summarizer\ndescription: Summarize text\n---\nbody",
    )
    reg = SubagentRegistry.load(project_root=project, user_root=None)
    assert [d.name for d in reg.list_agents()] == ["summarizer"]
    assert reg.get("summarizer").description == "Summarize text"


def test_registry_loads_user_definitions(tmp_path: Path) -> None:
    user = tmp_path / "user"
    _write_def(
        user / ".agent_harness" / "agents",
        "scribe",
        "---\nname: scribe\ndescription: Write notes\n---\nbody",
    )
    reg = SubagentRegistry.load(project_root=None, user_root=user)
    assert reg.get("scribe").source == "user"


def test_registry_project_shadows_user(tmp_path: Path) -> None:
    user = tmp_path / "user"
    project = tmp_path / "proj"
    _write_def(
        user / ".agent_harness" / "agents",
        "dup",
        "---\nname: dup\ndescription: from user\n---\nbody",
    )
    _write_def(
        project / ".agent" / "agents",
        "dup",
        "---\nname: dup\ndescription: from project\n---\nbody",
    )
    reg = SubagentRegistry.load(project_root=project, user_root=user)
    assert reg.get("dup").description == "from project"
    assert reg.get("dup").source == "project"


def test_registry_get_unknown_raises_toolerror(tmp_path: Path) -> None:
    reg = SubagentRegistry.load(project_root=tmp_path, user_root=None)
    with pytest.raises(ToolError):
        reg.get("never_defined")


def test_registry_skips_non_md_entries(tmp_path: Path) -> None:
    root = tmp_path / ".agent" / "agents"
    root.mkdir(parents=True)
    (root / "ignore.txt").write_text("not yaml")
    _write_def(root, "real", "---\nname: real\ndescription: ok\n---\nbody")
    reg = SubagentRegistry.load(project_root=tmp_path, user_root=None)
    assert [d.name for d in reg.list_agents()] == ["real"]


def test_registry_rejects_bad_frontmatter(tmp_path: Path) -> None:
    path = _write_def(
        tmp_path / ".agent" / "agents",
        "broken",
        "no frontmatter at all\nthis won't parse",
    )
    with pytest.raises(ConfigError) as exc:
        SubagentRegistry.load(project_root=tmp_path, user_root=None)
    assert "missing a YAML frontmatter block" in str(exc.value)
    assert exc.value.context["path"] == str(path)


def test_registry_rejects_bad_isolation(tmp_path: Path) -> None:
    _write_def(
        tmp_path / ".agent" / "agents",
        "iso",
        "---\nname: iso\ndescription: bad\nisolation: cloud\n---\nbody",
    )
    with pytest.raises(ConfigError) as exc:
        SubagentRegistry.load(project_root=tmp_path, user_root=None)
    assert "isolation" in str(exc.value)


def test_registry_parses_full_field_set(tmp_path: Path) -> None:
    body = (
        "---\n"
        "name: reviewer\n"
        "description: Code reviewer\n"
        "model: opus-4.7\n"
        "permission_mode: plan\n"
        "mcp_servers: [github, jira]\n"
        "required_mcp_servers: [github]\n"
        "tools: [Read, Grep]\n"
        "disallowed_tools: [Write, Bash]\n"
        "max_turns: 30\n"
        "skills: [security-review]\n"
        "memory: project\n"
        "background: true\n"
        "isolation: worktree\n"
        "initial_prompt: |\n"
        "  You are a reviewer.\n"
        "---\nbody\n"
    )
    _write_def(tmp_path / ".agent" / "agents", "reviewer", body)
    d = SubagentRegistry.load(project_root=tmp_path, user_root=None).get("reviewer")
    assert d.model == "opus-4.7"
    assert d.permission_mode == "plan"
    assert d.mcp_servers == ("github", "jira")
    assert d.required_mcp_servers == ("github",)
    assert d.tools == ("Read", "Grep")
    assert d.disallowed_tools == ("Write", "Bash")
    assert d.max_turns == 30
    assert d.skills == ("security-review",)
    assert d.memory == "project"
    assert d.background is True
    assert d.isolation == "worktree"
    assert "reviewer" in d.initial_prompt.lower()


def test_iter_definition_paths(tmp_path: Path) -> None:
    root = tmp_path / "agents"
    root.mkdir()
    (root / "a.md").write_text("ok")
    (root / "b.md").write_text("ok")
    (root / "c.txt").write_text("not md")
    out = iter_definition_paths([root, tmp_path / "missing"])
    assert sorted(p.name for p in out) == ["a.md", "b.md"]


# --- build_agent_tool ------------------------------------------------------


def test_build_agent_tool_materializes_child() -> None:
    parent: Agent[Any, Any] = Agent(
        name="parent", model=_model(FakeTurn(text="parent")), toolsets=[]
    )
    d = AgentDefinition(name="child", description="A child agent.", body_path=Path("/c.md"))
    tool = build_agent_tool(d, parent)
    assert tool.name == "child"
    assert tool.description == "A child agent."
    assert tool.policy.always_load is True


def test_build_agent_tool_rejects_unmet_mcp_requirements() -> None:
    parent: Agent[Any, Any] = Agent(name="parent", model=_model(FakeTurn(text="x")), toolsets=[])
    d = AgentDefinition(
        name="reqs",
        description="needs github",
        required_mcp_servers=("github",),
        body_path=Path("/c.md"),
    )
    with pytest.raises(ConfigError) as exc:
        build_agent_tool(d, parent)
    assert "github" in str(exc.value)


# --- NestedInterruption ----------------------------------------------------


def test_nested_interruption_payload() -> None:
    from datetime import UTC, datetime

    req = ApprovalRequest(
        tool_call_id="c1",
        tool_name="rm",
        arguments={"path": "/tmp"},
        requested_at=datetime.now(UTC),
    )
    exc = NestedInterruption(
        child_agent_name="child",
        tool_call_id="parent-call",
        pending_approvals=[req],
        child_run_state=None,
    )
    assert exc.child_agent_name == "child"
    assert exc.tool_call_id == "parent-call"
    assert exc.pending_approvals == [req]
    assert "child" in str(exc)


# --- republish_event_for_parent / RepublishingBus -------------------------


@pytest.mark.asyncio
async def test_republish_filters_run_and_agent_bookends() -> None:
    from datetime import UTC, datetime

    from agent_harness.core.events import (
        AgentEnd,
        AgentStart,
        InMemoryEventBus,
        MessageDelta,
        RunEnd,
        RunStart,
    )
    from agent_harness.core.models import Message, TextBlock, Usage

    parent = InMemoryEventBus()
    sub = parent.subscribe()

    msg = Message(
        role="assistant",
        content=[TextBlock(text="hi")],
        timestamp=datetime.now(UTC),
    )

    # These four are filtered out.
    await republish_event_for_parent(parent, RunStart(run_id="r", agent_name="c", prompt="p"))
    await republish_event_for_parent(parent, AgentStart(agent_name="c"))
    await republish_event_for_parent(parent, AgentEnd(agent_name="c"))
    await republish_event_for_parent(
        parent, RunEnd(run_id="r", result=None, usage=Usage(), duration_ms=1)
    )
    # This one passes through.
    await republish_event_for_parent(parent, MessageDelta(message_id="m", delta="hi", partial=msg))
    await parent.close()
    received: list[Any] = [ev async for ev in sub]
    assert len(received) == 1
    assert isinstance(received[0], MessageDelta)


@pytest.mark.asyncio
async def test_republishing_bus_delegates_to_parent() -> None:
    from datetime import UTC, datetime

    from agent_harness.core.events import (
        AgentStart,
        InMemoryEventBus,
        MessageDelta,
    )
    from agent_harness.core.models import Message, TextBlock

    parent = InMemoryEventBus()
    sub = parent.subscribe()
    bus = RepublishingBus(parent)
    msg = Message(role="assistant", content=[TextBlock(text="x")], timestamp=datetime.now(UTC))
    await bus.publish(AgentStart(agent_name="c"))  # filtered
    await bus.publish(MessageDelta(message_id="m", delta="x", partial=msg))  # passed
    await parent.close()
    seen = [ev async for ev in sub]
    assert len(seen) == 1
    assert isinstance(seen[0], MessageDelta)


@pytest.mark.asyncio
async def test_republishing_bus_with_none_parent_drops_silently() -> None:
    bus = RepublishingBus(None)
    from datetime import UTC, datetime

    from agent_harness.core.events import AgentStart, MessageDelta
    from agent_harness.core.models import Message, TextBlock

    msg = Message(role="assistant", content=[TextBlock(text="x")], timestamp=datetime.now(UTC))
    # No parent — both should be silently dropped without raising.
    await bus.publish(AgentStart(agent_name="c"))
    await bus.publish(MessageDelta(message_id="m", delta="x", partial=msg))
    await bus.close()


# --- make_paused_tool_result ----------------------------------------------


def test_make_paused_tool_result_carries_metadata() -> None:
    r = make_paused_tool_result("child", 3)
    assert r.error is not None
    assert r.metadata == {"pending_approvals": 3, "child_agent_name": "child"}


# --- create_worktree / remove_worktree -----------------------------------


def test_create_worktree_inactive_when_not_git(tmp_path: Path) -> None:
    handle = create_worktree(tmp_path)
    assert isinstance(handle, WorktreeHandle)
    assert handle.active is False
    # Inactive handles report the parent path so callers fall back to "share cwd".
    assert handle.path == tmp_path.resolve()


def test_remove_worktree_inactive_returns_false(tmp_path: Path) -> None:
    handle = WorktreeHandle(path=tmp_path, branch="", parent_repo=tmp_path, active=False)
    assert remove_worktree(handle) is False


@pytest.mark.skipif(_GIT is None, reason="git binary unavailable")
def test_create_worktree_inside_git_repo(tmp_path: Path) -> None:
    # Init a real repo with an initial commit so worktree-add can branch.
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "README.md").write_text("hi", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"], check=True)

    handle = create_worktree(tmp_path, name_hint="testsub")
    try:
        assert handle.active is True
        assert handle.path != tmp_path.resolve()  # IS1: different cwd
        assert handle.path.is_dir()
        # The worktree's README inherits from the parent.
        assert (handle.path / "README.md").read_text(encoding="utf-8") == "hi"
    finally:
        ok = remove_worktree(handle)
        assert ok is True
        assert not handle.path.exists()
