"""Unit tests for the three built-in :class:`HookExecutor` implementations."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import httpx
import pytest

from agent_harness.core.hooks import (
    HookConfig,
    HookResponse,
    HttpExecutor,
    InProcessExecutor,
    SubprocessExecutor,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# SubprocessExecutor (HK8).
# ---------------------------------------------------------------------------


def _python() -> str:
    """Return a usable Python executable for subprocess tests."""
    return sys.executable


async def test_subprocess_executor_round_trips_json() -> None:
    """Stdin payload echoed back as a HookResponse via stdout (HK8)."""
    script = (
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "print(json.dumps({'action': 'modify', "
        "'modified_payload': {'tool_name': payload['tool_name'].upper()}}))\n"
    )
    cfg = HookConfig(
        event="PreToolUse",
        executor="subprocess",
        command=[_python(), "-c", script],
    )
    res = await SubprocessExecutor().execute(cfg, {"tool_name": "bash"})
    assert res.action == "modify"
    assert res.modified_payload == {"tool_name": "BASH"}


async def test_subprocess_executor_non_zero_exit_is_deny() -> None:
    """HK8: non-zero exit code = denial with stderr as the reason."""
    script = "import sys; sys.stderr.write('blocked: forbidden'); sys.exit(2)"
    cfg = HookConfig(
        event="PreToolUse",
        executor="subprocess",
        command=[_python(), "-c", script],
    )
    res = await SubprocessExecutor().execute(cfg, {})
    assert res.action == "deny"
    assert "blocked: forbidden" in (res.reason or "")


async def test_subprocess_executor_malformed_stdout_becomes_ignore() -> None:
    """HK4: bad JSON from hook → ``ignore`` (not a crash)."""
    cfg = HookConfig(
        event="PreToolUse",
        executor="subprocess",
        command=[_python(), "-c", "print('this is not json')"],
    )
    res = await SubprocessExecutor().execute(cfg, {})
    assert res.action == "ignore"
    assert res.reason and "malformed" in res.reason


async def test_subprocess_executor_empty_stdout_is_ignore() -> None:
    cfg = HookConfig(
        event="PreToolUse",
        executor="subprocess",
        command=[_python(), "-c", "pass"],
    )
    res = await SubprocessExecutor().execute(cfg, {})
    assert res.action == "ignore"


async def test_subprocess_executor_timeout_is_deny() -> None:
    """HK11: hook that exceeds the budget = deny."""
    cfg = HookConfig(
        event="PreToolUse",
        executor="subprocess",
        command=[_python(), "-c", "import time; time.sleep(2)"],
        timeout_seconds=0.2,
    )
    res = await SubprocessExecutor().execute(cfg, {})
    assert res.action == "deny"
    assert "timed out" in (res.reason or "")


async def test_subprocess_executor_missing_binary_is_ignore() -> None:
    """Failure to spawn the process surfaces as ``ignore`` (HK4)."""
    cfg = HookConfig(
        event="PreToolUse",
        executor="subprocess",
        command=["/__definitely_does_not_exist__/agent_hook"],
    )
    res = await SubprocessExecutor().execute(cfg, {})
    assert res.action == "ignore"
    assert res.reason and "failed to spawn" in res.reason


async def test_subprocess_executor_non_dict_json_is_ignore() -> None:
    """A JSON list/scalar from stdout is not a HookResponse."""
    cfg = HookConfig(
        event="PreToolUse",
        executor="subprocess",
        command=[_python(), "-c", "print('[1,2,3]')"],
    )
    res = await SubprocessExecutor().execute(cfg, {})
    assert res.action == "ignore"


# ---------------------------------------------------------------------------
# InProcessExecutor (HK10).
# ---------------------------------------------------------------------------


async def test_in_process_executor_calls_sync_callable() -> None:
    cfg = HookConfig(
        event="PreToolUse",
        executor="in_process",
        callable_ref=f"{__name__}:_hook_sync_allow",
    )
    res = await InProcessExecutor().execute(cfg, {"x": 1})
    assert res.action == "allow"
    assert res.reason == "sync ok"


async def test_in_process_executor_calls_async_callable() -> None:
    cfg = HookConfig(
        event="PreToolUse",
        executor="in_process",
        callable_ref=f"{__name__}:_hook_async_modify",
    )
    res = await InProcessExecutor().execute(cfg, {"tool_name": "bash"})
    assert res.action == "modify"
    assert res.modified_payload == {"tool_name": "BASH"}


async def test_in_process_executor_callable_can_return_dict() -> None:
    cfg = HookConfig(
        event="PreToolUse",
        executor="in_process",
        callable_ref=f"{__name__}:_hook_returns_dict",
    )
    res = await InProcessExecutor().execute(cfg, {})
    assert res.action == "deny"
    assert res.reason == "policy"


async def test_in_process_executor_none_means_allow() -> None:
    cfg = HookConfig(
        event="PreToolUse",
        executor="in_process",
        callable_ref=f"{__name__}:_hook_returns_none",
    )
    res = await InProcessExecutor().execute(cfg, {})
    assert res.action == "allow"


async def test_in_process_executor_exception_becomes_ignore() -> None:
    """HK4: exceptions in user code never crash the loop."""
    cfg = HookConfig(
        event="PreToolUse",
        executor="in_process",
        callable_ref=f"{__name__}:_hook_raises",
    )
    res = await InProcessExecutor().execute(cfg, {})
    assert res.action == "ignore"
    assert res.reason and "RuntimeError" in res.reason


async def test_in_process_executor_missing_callable_is_ignore() -> None:
    cfg = HookConfig(
        event="PreToolUse",
        executor="in_process",
        callable_ref=f"{__name__}:does_not_exist",
    )
    res = await InProcessExecutor().execute(cfg, {})
    assert res.action == "ignore"
    assert res.reason and "import failed" in res.reason


async def test_in_process_executor_missing_module_is_ignore() -> None:
    cfg = HookConfig(
        event="PreToolUse",
        executor="in_process",
        callable_ref="agent_harness._nonexistent_module:func",
    )
    res = await InProcessExecutor().execute(cfg, {})
    assert res.action == "ignore"


async def test_in_process_executor_unsupported_return_type_is_ignore() -> None:
    cfg = HookConfig(
        event="PreToolUse",
        executor="in_process",
        callable_ref=f"{__name__}:_hook_returns_int",
    )
    res = await InProcessExecutor().execute(cfg, {})
    assert res.action == "ignore"
    assert res.reason and "unsupported" in res.reason


async def test_in_process_executor_async_timeout_is_deny() -> None:
    """HK11: async in-process callable that exceeds the timeout is denied."""
    cfg = HookConfig(
        event="PreToolUse",
        executor="in_process",
        callable_ref=f"{__name__}:_hook_async_slow",
        timeout_seconds=0.05,
    )
    res = await InProcessExecutor().execute(cfg, {})
    assert res.action == "deny"
    assert "timed out" in (res.reason or "")


# ---------------------------------------------------------------------------
# HttpExecutor (HK9).
# ---------------------------------------------------------------------------


# A tiny in-process httpx mock — instead of a real server we use httpx.MockTransport.


async def test_http_executor_round_trips_json() -> None:
    received: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["body"] = json.loads(request.content)
        received["url"] = str(request.url)
        return httpx.Response(200, json={"action": "modify", "modified_payload": {"ok": True}})

    # Patch AsyncClient with a MockTransport.
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> Any:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(httpx, "AsyncClient", factory)
    try:
        cfg = HookConfig(
            event="PreToolUse",
            executor="http",
            url="https://hook.invalid/check",
        )
        res = await HttpExecutor().execute(cfg, {"tool": "bash"})
    finally:
        monkey.undo()

    assert res.action == "modify"
    assert res.modified_payload == {"ok": True}
    assert received["body"] == {"tool": "bash"}
    assert "hook.invalid" in received["url"]


async def test_http_executor_non_2xx_is_deny() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server died")

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> Any:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(httpx, "AsyncClient", factory)
    try:
        cfg = HookConfig(event="PreToolUse", executor="http", url="https://hook.invalid/x")
        res = await HttpExecutor().execute(cfg, {})
    finally:
        monkey.undo()

    assert res.action == "deny"
    assert "500" in (res.reason or "")


async def test_http_executor_non_json_body_is_ignore() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"<!doctype html>", headers={"content-type": "text/html"}
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> Any:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(httpx, "AsyncClient", factory)
    try:
        cfg = HookConfig(event="PreToolUse", executor="http", url="https://hook.invalid/x")
        res = await HttpExecutor().execute(cfg, {})
    finally:
        monkey.undo()

    assert res.action == "ignore"


async def test_http_executor_timeout_is_deny() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> Any:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(httpx, "AsyncClient", factory)
    try:
        cfg = HookConfig(
            event="PreToolUse",
            executor="http",
            url="https://hook.invalid/x",
            timeout_seconds=0.05,
        )
        res = await HttpExecutor().execute(cfg, {})
    finally:
        monkey.undo()

    assert res.action == "deny"
    assert "timed out" in (res.reason or "")


# ---------------------------------------------------------------------------
# In-process callables used as in_process fixtures.
# ---------------------------------------------------------------------------


def _hook_sync_allow(payload: dict[str, Any]) -> HookResponse:
    return HookResponse(action="allow", reason="sync ok")


async def _hook_async_modify(payload: dict[str, Any]) -> HookResponse:
    return HookResponse(
        action="modify",
        modified_payload={"tool_name": str(payload["tool_name"]).upper()},
    )


def _hook_returns_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {"action": "deny", "reason": "policy"}


def _hook_returns_none(payload: dict[str, Any]) -> None:
    return None


def _hook_raises(payload: dict[str, Any]) -> HookResponse:
    raise RuntimeError("boom")


def _hook_returns_int(payload: dict[str, Any]) -> int:
    return 42


async def _hook_async_slow(payload: dict[str, Any]) -> HookResponse:
    await asyncio.sleep(2)
    return HookResponse(action="allow")
