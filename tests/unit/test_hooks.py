"""Unit tests for ``agent_harness.core.hooks`` — config + registry + dispatch."""

from __future__ import annotations

import asyncio
from typing import Any, get_args

import pytest

from agent_harness.core.errors import ConfigError
from agent_harness.core.hooks import (
    DEFAULT_HOOK_TIMEOUT_SECONDS,
    HOOK_EVENT_NAMES,
    HookConfig,
    HookEvent,
    HookExecutor,
    HookRegistry,
    HookResponse,
    HttpExecutor,
    InProcessExecutor,
    SubprocessExecutor,
)

# ---------------------------------------------------------------------------
# Closed-set taxonomy (HK1).
# ---------------------------------------------------------------------------


def test_hook_event_is_closed_set_of_19() -> None:
    """The closed-set taxonomy must list exactly the named lifecycle points."""

    expected = {
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "UserPromptSubmit",
        "SessionStart",
        "SessionEnd",
        "Stop",
        "PreCompact",
        "PostCompact",
        "SubagentStart",
        "SubagentStop",
        "PermissionRequest",
        "Elicitation",
        "CwdChanged",
        "FileChanged",
        "WorktreeCreate",
        "WorktreeRemove",
        "InstructionsLoaded",
        "ConfigChange",
    }
    assert set(get_args(HookEvent)) == expected
    assert expected == HOOK_EVENT_NAMES


def test_default_timeout_is_30s() -> None:
    """HK11 — 30 second default budget."""

    assert DEFAULT_HOOK_TIMEOUT_SECONDS == 30.0


# ---------------------------------------------------------------------------
# Response shape.
# ---------------------------------------------------------------------------


def test_hook_response_default_action_is_allow() -> None:
    assert HookResponse().action == "allow"


def test_hook_response_rejects_extra_keys() -> None:
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        HookResponse.model_validate({"action": "allow", "bogus": 1})


def test_hook_response_actions_are_closed() -> None:
    for action in ("allow", "deny", "modify", "ignore"):
        assert HookResponse.model_validate({"action": action}).action == action
    with pytest.raises(Exception):  # noqa: B017
        HookResponse.model_validate({"action": "explode"})


# ---------------------------------------------------------------------------
# HookConfig validation.
# ---------------------------------------------------------------------------


def test_subprocess_config_requires_command() -> None:
    with pytest.raises(ConfigError, match="subprocess hook requires"):
        HookConfig(event="PreToolUse", executor="subprocess")


def test_http_config_requires_url() -> None:
    with pytest.raises(ConfigError, match="http hook requires url"):
        HookConfig(event="PreToolUse", executor="http")


def test_in_process_config_requires_callable_ref_with_colon() -> None:
    with pytest.raises(ConfigError, match="module:func"):
        HookConfig(event="PreToolUse", executor="in_process", callable_ref="nope")
    with pytest.raises(ConfigError, match="module:func"):
        HookConfig(event="PreToolUse", executor="in_process")


def test_in_process_config_accepts_module_function_ref() -> None:
    cfg = HookConfig(
        event="PreToolUse",
        executor="in_process",
        callable_ref="some.module:func",
    )
    assert cfg.callable_ref == "some.module:func"


def test_config_rejects_non_positive_timeout() -> None:
    with pytest.raises(ConfigError, match="timeout_seconds must be positive"):
        HookConfig(
            event="PreToolUse",
            executor="subprocess",
            command=["true"],
            timeout_seconds=0,
        )


def test_config_default_blocking_is_true() -> None:
    cfg = HookConfig(event="PreToolUse", executor="subprocess", command=["true"])
    assert cfg.blocking is True
    # ``source`` is loader-owned (HK7): a hook constructed in user code or
    # parsed from JSON has no source until the loader assigns one.
    assert cfg.source is None


def test_config_rejects_user_supplied_source() -> None:
    """``source`` is loader-owned (HK7) — user code can't set it directly."""

    with pytest.raises(ConfigError, match="loader-owned"):
        HookConfig(
            event="PreToolUse",
            executor="subprocess",
            command=["true"],
            source="user",
        )


def test_with_source_stamps_a_loader_resolved_tier() -> None:
    """The sanctioned path: build the base config then stamp the tier."""

    base = HookConfig(event="Stop", executor="subprocess", command=["true"])
    assert base.source is None
    stamped = HookConfig.with_source(base, source="project")
    assert stamped.source == "project"
    # The base config remains unchanged (with_source returns a new instance).
    assert base.source is None


# ---------------------------------------------------------------------------
# Registry + dispatch with a fake executor.
# ---------------------------------------------------------------------------


class _FakeExecutor:
    """Records calls; returns scripted responses."""

    def __init__(self, responses: list[HookResponse | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[HookConfig, dict[str, Any], float | None]] = []

    async def execute(
        self,
        config: HookConfig,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> HookResponse:
        self.calls.append((config, dict(payload), timeout))
        if not self._responses:
            return HookResponse(action="allow")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _cfg(
    *,
    event: HookEvent = "PreToolUse",
    matcher: dict[str, Any] | None = None,
    blocking: bool = True,
    source: str = "user",
    timeout: float = DEFAULT_HOOK_TIMEOUT_SECONDS,
) -> HookConfig:
    base = HookConfig(
        event=event,
        matcher=matcher,
        executor="in_process",
        callable_ref="x:y",
        blocking=blocking,
        timeout_seconds=timeout,
    )
    return HookConfig.with_source(base, source=source)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fire_empty_registry_returns_ignore() -> None:
    reg = HookRegistry([])
    res = await reg.fire("PreToolUse", {})
    assert res.action == "ignore"
    # Reason distinguishes "no hooks configured" from "hooks ran and abstained".
    assert res.reason == "no hooks configured"


@pytest.mark.asyncio
async def test_fire_all_abstained_returns_ignore_with_different_reason() -> None:
    """When hooks ARE configured but all return ``ignore``, the aggregate
    response carries a distinct reason from the empty-registry case."""

    fake = _FakeExecutor([HookResponse(action="ignore", reason="abstained")])
    reg = HookRegistry([_cfg()], executors={"in_process": fake})
    res = await reg.fire("PreToolUse", {"tool_name": "bash"})
    assert res.action == "ignore"
    assert res.reason == "all hooks abstained"


@pytest.mark.asyncio
async def test_fire_unknown_event_raises_config_error() -> None:
    reg = HookRegistry([])
    with pytest.raises(ConfigError, match="unknown HookEvent"):
        await reg.fire("NotAnEvent", {})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fire_invokes_matching_hook_and_returns_response() -> None:
    fake = _FakeExecutor([HookResponse(action="allow", model_context="ctx")])
    reg = HookRegistry([_cfg()], executors={"in_process": fake})
    res = await reg.fire("PreToolUse", {"tool_name": "bash"})
    assert res.action == "allow"
    assert res.model_context == "ctx"
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_matcher_filters_hooks() -> None:
    fake = _FakeExecutor([HookResponse(action="deny", reason="no")])
    reg = HookRegistry(
        [_cfg(matcher={"tool_name": "bash"})],
        executors={"in_process": fake},
    )
    res = await reg.fire("PreToolUse", {"tool_name": "read"})
    assert res.action == "ignore"
    assert fake.calls == []


@pytest.mark.asyncio
async def test_matcher_list_membership() -> None:
    fake = _FakeExecutor([HookResponse(action="deny", reason="blocked")])
    reg = HookRegistry(
        [_cfg(matcher={"tool_name": ["read", "edit"]})],
        executors={"in_process": fake},
    )
    res = await reg.fire("PreToolUse", {"tool_name": "edit"})
    assert res.action == "deny"


@pytest.mark.asyncio
async def test_first_deny_wins_and_short_circuits() -> None:
    """HK7: when a hook denies, no later hook runs."""
    first = _FakeExecutor([HookResponse(action="deny", reason="nope")])
    second = _FakeExecutor([HookResponse(action="allow")])
    reg = HookRegistry(
        [
            HookConfig.with_source(
                HookConfig(
                    event="PreToolUse",
                    executor="in_process",
                    callable_ref="a:b",
                ),
                source="user",
            ),
            HookConfig.with_source(
                HookConfig(
                    event="PreToolUse",
                    executor="subprocess",
                    command=["true"],
                ),
                source="agent",
            ),
        ],
        executors={"in_process": first, "subprocess": second},
    )
    res = await reg.fire("PreToolUse", {})
    assert res.action == "deny"
    assert res.reason == "nope"
    assert second.calls == [], "second hook must not run after deny"


@pytest.mark.asyncio
async def test_modify_composes_left_to_right() -> None:
    """HK7: ``modify`` hooks compose; later hooks see the earlier hook's payload."""
    first = _FakeExecutor(
        [HookResponse(action="modify", modified_payload={"tool_name": "bash", "extra": 1})]
    )
    second = _FakeExecutor(
        [HookResponse(action="modify", modified_payload={"tool_name": "bash", "extra": 2})]
    )
    reg = HookRegistry(
        [
            HookConfig.with_source(
                HookConfig(
                    event="PreToolUse",
                    executor="in_process",
                    callable_ref="a:b",
                ),
                source="user",
            ),
            HookConfig.with_source(
                HookConfig(
                    event="PreToolUse",
                    executor="subprocess",
                    command=["true"],
                ),
                source="agent",
            ),
        ],
        executors={"in_process": first, "subprocess": second},
    )
    res = await reg.fire("PreToolUse", {"tool_name": "bash"})
    assert res.action == "modify"
    assert res.modified_payload == {"tool_name": "bash", "extra": 2}
    # The second hook saw the first hook's modified payload.
    assert second.calls[0][1]["extra"] == 1


@pytest.mark.asyncio
async def test_hooks_sorted_by_source_tier() -> None:
    """HK7: resolution order policy → user → project → local → skill → agent."""
    captured: list[str] = []

    class _Recorder:
        def __init__(self, name: str) -> None:
            self._name = name

        async def execute(
            self,
            config: HookConfig,
            payload: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> HookResponse:
            captured.append(self._name)
            return HookResponse(action="allow")

    cfgs = [
        HookConfig.with_source(
            HookConfig(event="Stop", executor="subprocess", command=["true"]), source="agent"
        ),
        HookConfig.with_source(
            HookConfig(event="Stop", executor="subprocess", command=["true"]), source="policy"
        ),
        HookConfig.with_source(
            HookConfig(event="Stop", executor="subprocess", command=["true"]), source="project"
        ),
    ]
    reg = HookRegistry(cfgs, executors={"subprocess": _Recorder("only")})
    # Examine ordering through `hooks_for` (deterministic, no I/O).
    order = [c.source for c in reg.hooks_for("Stop")]
    assert order == ["policy", "project", "agent"]


@pytest.mark.asyncio
async def test_non_blocking_hook_is_fire_and_forget() -> None:
    """HK6: non-blocking hooks don't contribute to the aggregate response."""
    started = asyncio.Event()
    finished = asyncio.Event()

    class _Slow:
        async def execute(
            self,
            config: HookConfig,
            payload: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> HookResponse:
            started.set()
            await asyncio.sleep(0.01)
            finished.set()
            return HookResponse(action="deny", reason="late")

    reg = HookRegistry(
        [_cfg(blocking=False)],
        executors={"in_process": _Slow()},
    )
    res = await reg.fire("PreToolUse", {})
    assert res.action == "ignore"
    # The hook is scheduled; eventually finishes, but did not block fire().
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.wait_for(finished.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_hook_exception_becomes_ignore_and_reports() -> None:
    """HK4: a hook exception is converted to ``ignore`` and surfaced via error_sink."""
    reports: list[str] = []
    reg = HookRegistry(
        [_cfg()],
        executors={"in_process": _FakeExecutor([RuntimeError("boom")])},
        error_sink=reports.append,
    )
    res = await reg.fire("PreToolUse", {})
    assert res.action == "ignore"
    assert reports and "boom" in reports[0]


@pytest.mark.asyncio
async def test_async_error_sink_is_awaited() -> None:
    """HK4: error_sink may be async."""
    reports: list[str] = []

    async def sink(msg: str) -> None:
        reports.append(msg)

    reg = HookRegistry(
        [_cfg()],
        executors={"in_process": _FakeExecutor([RuntimeError("crashed")])},
        error_sink=sink,
    )
    await reg.fire("PreToolUse", {})
    assert reports and "crashed" in reports[0]


@pytest.mark.asyncio
async def test_missing_executor_is_skipped_and_reported() -> None:
    """A hook config naming an unregistered executor is skipped, not crashed."""
    reports: list[str] = []
    cfg = HookConfig.with_source(
        HookConfig(
            event="PreToolUse",
            executor="subprocess",
            command=["true"],
        ),
        source="user",
    )
    reg = HookRegistry([cfg], executors={}, error_sink=reports.append)
    res = await reg.fire("PreToolUse", {})
    assert res.action == "ignore"
    assert reports and "no executor" in reports[0]


@pytest.mark.asyncio
async def test_hook_for_other_event_does_not_fire() -> None:
    fake = _FakeExecutor([HookResponse(action="deny")])
    reg = HookRegistry(
        [_cfg(event="PreToolUse")],
        executors={"in_process": fake},
    )
    res = await reg.fire("PostToolUse", {})
    assert res.action == "ignore"
    assert fake.calls == []


@pytest.mark.asyncio
async def test_default_executors_satisfy_protocol() -> None:
    """All three built-in executors satisfy :class:`HookExecutor`."""
    for cls in (SubprocessExecutor, HttpExecutor, InProcessExecutor):
        assert isinstance(cls(), HookExecutor)


@pytest.mark.asyncio
async def test_registry_constructs_default_executors_when_none_provided() -> None:
    """Constructor should fall back to all three built-ins when ``executors=None``."""
    reg = HookRegistry([])
    # Public surface: hooks_for is empty but firing a known event must still resolve.
    res = await reg.fire("ConfigChange", {})
    assert res.action == "ignore"


@pytest.mark.asyncio
async def test_aclose_waits_for_pending_background_tasks() -> None:
    """``aclose`` cancels and awaits in-flight non-blocking hook tasks so
    they don't leak when the registry is torn down."""
    started = asyncio.Event()

    class _Slow:
        async def execute(
            self,
            config: HookConfig,
            payload: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> HookResponse:
            started.set()
            await asyncio.sleep(60)  # long enough that aclose must cancel.
            return HookResponse(action="allow")

    reg = HookRegistry(
        [_cfg(blocking=False)],
        executors={"in_process": _Slow()},
    )
    await reg.fire("PreToolUse", {})
    await asyncio.wait_for(started.wait(), timeout=1.0)
    # ``aclose`` should return quickly because we cancel the slow task.
    await asyncio.wait_for(reg.aclose(timeout=1.0), timeout=1.5)
    assert reg._background == set()


@pytest.mark.asyncio
async def test_aclose_is_idempotent_when_empty() -> None:
    """``aclose`` on a registry with no background tasks is a no-op."""
    reg = HookRegistry([])
    await reg.aclose()
    await reg.aclose()  # twice for good measure


@pytest.mark.asyncio
async def test_error_sink_exception_does_not_break_loop() -> None:
    """HK4: a misbehaving error sink must NOT propagate into the loop."""

    def bad_sink(_msg: str) -> None:
        raise RuntimeError("sink exploded")

    reg = HookRegistry(
        [_cfg()],
        executors={"in_process": _FakeExecutor([RuntimeError("crashed")])},
        error_sink=bad_sink,
    )
    # The hook raises, ``_report`` is called with the message, the sink
    # raises in turn — but ``fire`` must still return cleanly with ``ignore``.
    res = await reg.fire("PreToolUse", {})
    assert res.action == "ignore"
