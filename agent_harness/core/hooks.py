"""Closed-set lifecycle hooks (HK1-HK13).

Users **intervene** at named lifecycle points by configuring data in
``settings.json`` / skill / agent frontmatter, not by registering callables
in user code. Three executor types ship out of the box:

* :class:`SubprocessExecutor` — JSON-on-stdin / JSON-on-stdout (HK8).
* :class:`HttpExecutor`       — POST JSON, parse JSON response (HK9).
* :class:`InProcessExecutor`  — import ``module:function`` (HK10).

:data:`HookEvent` is a **closed set** (HK1); adding a name is a versioned
API change. Misbehaving hooks (crash / timeout / malformed response) are
coerced into an ``"ignore"`` :class:`HookResponse` instead of crashing the
loop (HK4 / HK11).

Example:
    >>> import asyncio
    >>> asyncio.run(HookRegistry([]).fire("SessionStart", {})).action
    'ignore'
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, get_args, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ConfigError

# --- Closed-set event taxonomy (HK1) -----------------------------------------

HookEvent = Literal[
    # Tool lifecycle
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    # Turn lifecycle
    "UserPromptSubmit",
    "SessionStart",
    "SessionEnd",
    "Stop",
    # Compaction lifecycle
    "PreCompact",
    "PostCompact",
    # Subagent lifecycle
    "SubagentStart",
    "SubagentStop",
    # Approval / elicitation
    "PermissionRequest",
    "Elicitation",
    # Workspace lifecycle
    "CwdChanged",
    "FileChanged",
    "WorktreeCreate",
    "WorktreeRemove",
    # Configuration lifecycle
    "InstructionsLoaded",
    "ConfigChange",
]
"""Closed-set event names. Adding a member is a versioned API change."""

HOOK_EVENT_NAMES: frozenset[str] = frozenset(get_args(HookEvent))
"""Runtime mirror of :data:`HookEvent` for validation."""

DEFAULT_HOOK_TIMEOUT_SECONDS: float = 30.0
"""Default per-hook wall-clock budget (HK11)."""


# --- Hook response -----------------------------------------------------------


class HookResponse(BaseModel):
    """What a hook returns; loop interprets ``action`` per event semantics.

    Example: ``HookResponse(action="deny", reason="no rm").action == "deny"``.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["allow", "deny", "modify", "ignore"] = "allow"
    reason: str | None = None
    modified_payload: dict[str, Any] | None = None
    model_context: str | None = None


# --- Hook config -------------------------------------------------------------


class HookConfig(BaseModel):
    """Declarative hook config — loaded from settings / skill / frontmatter.

    Example: ``HookConfig(event="SessionStart", executor="subprocess", command=["echo"]).timeout_seconds == 30.0``.
    """

    model_config = ConfigDict(extra="forbid")

    event: HookEvent
    matcher: dict[str, Any] | None = None
    executor: Literal["subprocess", "http", "in_process"]
    command: list[str] | None = None
    url: str | None = None
    callable_ref: str | None = None
    timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS
    blocking: bool = True
    # ``source`` is *loader-owned* (HK7): frozen, defaults to ``None``, set
    # only via :meth:`with_source` by the Wave-3 loader. Direct construction
    # with a non-``None`` value raises :class:`ConfigError`.
    source: Literal["policy", "user", "project", "local", "skill", "agent"] | None = Field(
        default=None, frozen=True
    )

    @model_validator(mode="after")
    def _check(self) -> HookConfig:
        ctx = {"event": self.event}
        match self.executor:
            case "subprocess":
                if not self.command:
                    raise ConfigError("subprocess hook requires non-empty command", context=ctx)
            case "http":
                if not self.url:
                    raise ConfigError("http hook requires url", context=ctx)
            case "in_process":
                if not self.callable_ref or ":" not in self.callable_ref:
                    raise ConfigError(
                        "in_process hook requires 'module:func' callable_ref",
                        context={**ctx, "ref": self.callable_ref},
                    )
        if self.timeout_seconds <= 0:
            raise ConfigError(
                "timeout_seconds must be positive",
                context={"timeout_seconds": self.timeout_seconds},
            )
        if self.source is not None:
            raise ConfigError(
                "HookConfig.source is loader-owned; use HookConfig.with_source(...)",
                context={**ctx, "source": self.source},
            )
        return self

    @classmethod
    def with_source(
        cls,
        base: HookConfig,
        *,
        source: Literal["policy", "user", "project", "local", "skill", "agent"],
    ) -> HookConfig:
        """Return a copy of ``base`` with ``source`` stamped (HK7 loader path).

        The *only* sanctioned write path for ``HookConfig.source``. Uses
        ``model_construct`` so the loader-owned validator (which rejects
        user-supplied sources) is bypassed.

        Example:
            >>> base = HookConfig(event="Stop", executor="subprocess", command=["true"])
            >>> HookConfig.with_source(base, source="user").source
            'user'
        """

        data = base.model_dump()
        data["source"] = source
        return cls.model_construct(**data)


# --- Executor Protocol + three built-ins -------------------------------------


@runtime_checkable
class HookExecutor(Protocol):
    """A way to invoke a hook. Example: ``isinstance(SubprocessExecutor(), HookExecutor)``."""

    async def execute(
        self, config: HookConfig, payload: dict[str, Any], *, timeout: float | None = None
    ) -> HookResponse: ...


def _parse_response(raw: str | bytes | dict[str, Any]) -> HookResponse:
    """Parse a HookResponse; malformed input → ``ignore`` (HK4)."""
    if isinstance(raw, dict):
        data: dict[str, Any] = raw
    else:
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        text = text.strip()
        if not text:
            return HookResponse(action="ignore", reason="empty hook response")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return HookResponse(action="ignore", reason=f"malformed hook JSON: {exc}")
        if not isinstance(parsed, dict):
            return HookResponse(action="ignore", reason="hook response not a JSON object")
        data = parsed
    try:
        return HookResponse.model_validate(data)
    except Exception as exc:
        return HookResponse(action="ignore", reason=f"invalid HookResponse: {exc}")


class SubprocessExecutor:
    """Subprocess (HK8): JSON on stdin, JSON on stdout. Non-zero exit → ``deny``."""

    async def execute(
        self,
        config: HookConfig,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> HookResponse:
        assert config.command, "validated by HookConfig"
        budget = timeout if timeout is not None else config.timeout_seconds
        try:
            proc = await asyncio.create_subprocess_exec(
                *config.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            return HookResponse(action="ignore", reason=f"failed to spawn hook: {exc}")
        blob = json.dumps(payload).encode("utf-8")
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(blob), timeout=budget)
        except TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            return HookResponse(action="deny", reason=f"hook timed out after {budget}s")
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip() or "non-zero exit"
            return HookResponse(action="deny", reason=err)
        return _parse_response(stdout)


class HttpExecutor:
    """HTTP (HK9): POST JSON, parse JSON. Non-2xx → ``deny``. Uses ``httpx`` (lazy import)."""

    async def execute(
        self,
        config: HookConfig,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> HookResponse:
        assert config.url, "validated by HookConfig"
        budget = timeout if timeout is not None else config.timeout_seconds
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover — only without the extra
            return HookResponse(action="ignore", reason=f"httpx not installed: {exc}")
        try:
            async with httpx.AsyncClient(timeout=budget) as client:
                resp = await client.post(config.url, json=payload)
        except httpx.TimeoutException:
            return HookResponse(action="deny", reason=f"hook timed out after {budget}s")
        except httpx.HTTPError as exc:
            return HookResponse(action="ignore", reason=f"http hook error: {exc}")
        if not (200 <= resp.status_code < 300):
            return HookResponse(
                action="deny",
                reason=f"http hook returned {resp.status_code}: {resp.text[:200]}",
            )
        try:
            body = resp.json()
        except ValueError as exc:
            return HookResponse(action="ignore", reason=f"http hook bad json: {exc}")
        if not isinstance(body, dict):
            return HookResponse(action="ignore", reason="http hook body not a JSON object")
        return _parse_response(body)


class InProcessExecutor:
    """In-process (HK10): import ``module:function`` and call it (sync or async).

    Exceptions become ``ignore`` so a buggy hook never crashes the loop (HK4).
    """

    async def execute(
        self,
        config: HookConfig,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> HookResponse:
        assert config.callable_ref and ":" in config.callable_ref, "validated by HookConfig"
        budget = timeout if timeout is not None else config.timeout_seconds
        mod_name, fn_name = config.callable_ref.split(":", 1)
        try:
            func = getattr(importlib.import_module(mod_name), fn_name)
        except (ImportError, AttributeError) as exc:
            return HookResponse(action="ignore", reason=f"hook import failed: {exc}")
        try:
            result = func(payload)
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=budget)
        except TimeoutError:
            return HookResponse(action="deny", reason=f"hook timed out after {budget}s")
        except Exception as exc:
            return HookResponse(action="ignore", reason=f"hook raised {type(exc).__name__}: {exc}")
        if isinstance(result, HookResponse):
            return result
        if isinstance(result, dict):
            return _parse_response(result)
        if result is None:
            return HookResponse(action="allow")
        return HookResponse(
            action="ignore", reason=f"hook returned unsupported {type(result).__name__}"
        )


# --- Registry ----------------------------------------------------------------

# Resolution order (HK7): most general → most specific. Hooks without a
# loader-set ``source`` sort between policy and user (``_UNSOURCED_KEY = 1``)
# so direct callers and tests see deterministic ordering.
_SOURCE_ORDER: dict[str, int] = {
    "policy": 0, "user": 1, "project": 2, "local": 3, "skill": 4, "agent": 5,
}  # fmt: skip
_UNSOURCED_KEY: int = 1


def _resolution_key(c: HookConfig) -> int:
    return _UNSOURCED_KEY if c.source is None else _SOURCE_ORDER[c.source]


def _matches(matcher: dict[str, Any] | None, payload: dict[str, Any]) -> bool:
    """Shallow matcher: scalar = equality, list = membership; missing = mismatch."""
    if matcher is None:
        return True
    for key, expected in matcher.items():
        if key not in payload:
            return False
        actual = payload[key]
        if isinstance(expected, list) and actual not in expected:
            return False
        if not isinstance(expected, list) and actual != expected:
            return False
    return True


_DEFAULT_EXECUTORS: dict[str, Callable[[], HookExecutor]] = {
    "subprocess": SubprocessExecutor,
    "http": HttpExecutor,
    "in_process": InProcessExecutor,
}


class HookRegistry:
    """Loads hooks at construction (HK12) and fires them at lifecycle points.

    Example: ``asyncio.run(HookRegistry([]).fire("Stop", {})).action == "ignore"``.
    """

    def __init__(
        self,
        configs: list[HookConfig],
        *,
        executors: dict[str, HookExecutor] | None = None,
        error_sink: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> None:
        self._by_event: dict[str, list[HookConfig]] = {}
        for cfg in configs:
            self._by_event.setdefault(cfg.event, []).append(cfg)
        for bucket in self._by_event.values():
            bucket.sort(key=_resolution_key)
        if executors is None:
            executors = {name: factory() for name, factory in _DEFAULT_EXECUTORS.items()}
        self._executors: dict[str, HookExecutor] = executors
        self._error_sink = error_sink
        self._background: set[asyncio.Task[Any]] = set()

    def hooks_for(self, event: HookEvent) -> list[HookConfig]:
        """Return the resolved hook list for ``event`` (diagnostic / tests)."""
        return list(self._by_event.get(event, []))

    async def fire(self, event: HookEvent, payload: dict[str, Any]) -> HookResponse:
        """Fire matching hooks for ``event`` in resolution order (HK5-HK7).

        Blocking hooks are awaited; first ``deny`` wins; ``modify`` composes
        left-to-right. Non-blocking hooks spawn as fire-and-forget tasks (HK6).
        Misbehaving hooks become ``ignore`` (HK4).
        """
        if event not in HOOK_EVENT_NAMES:
            raise ConfigError(f"unknown HookEvent {event!r}", context={"event": event})
        hooks = self._by_event.get(event, [])
        if not hooks:
            return HookResponse(action="ignore", reason="no hooks configured")

        current = dict(payload)
        # Start with a sentinel reason; tightened below if every hook
        # either ignored or short-circuited.
        aggregate = HookResponse(action="ignore", reason="all hooks abstained")
        modified = False
        last_ctx: str | None = None

        for cfg in hooks:
            if not _matches(cfg.matcher, current):
                continue
            executor = self._executors.get(cfg.executor)
            if executor is None:
                await self._report(f"no executor registered for {cfg.executor!r}")
                continue
            if not cfg.blocking:
                self._spawn_background(executor, cfg, dict(current))
                continue
            response = await self._safe_execute(executor, cfg, current)
            if response.model_context:
                last_ctx = response.model_context
            match response.action:
                case "deny":
                    if last_ctx and not response.model_context:
                        response = response.model_copy(update={"model_context": last_ctx})
                    return response
                case "modify":
                    modified = True
                    if response.modified_payload is not None:
                        current = response.modified_payload
                    aggregate = response
                case "allow":
                    if not modified:
                        aggregate = response
                case "ignore":
                    pass
        if modified:
            aggregate = aggregate.model_copy(
                update={"action": "modify", "modified_payload": current}
            )
        if last_ctx and not aggregate.model_context:
            aggregate = aggregate.model_copy(update={"model_context": last_ctx})
        return aggregate

    async def _safe_execute(
        self,
        executor: HookExecutor,
        cfg: HookConfig,
        payload: dict[str, Any],
    ) -> HookResponse:
        try:
            return await asyncio.wait_for(
                executor.execute(cfg, payload, timeout=cfg.timeout_seconds),
                timeout=cfg.timeout_seconds + 1.0,
            )
        except TimeoutError:
            return HookResponse(
                action="deny", reason=f"hook timed out after {cfg.timeout_seconds}s"
            )
        except Exception as exc:
            await self._report(f"hook {cfg.event}/{cfg.executor} raised: {exc}")
            return HookResponse(action="ignore", reason=str(exc))

    def _spawn_background(
        self,
        executor: HookExecutor,
        cfg: HookConfig,
        payload: dict[str, Any],
    ) -> None:
        async def _runner() -> None:
            await self._safe_execute(executor, cfg, payload)

        task = asyncio.create_task(_runner())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def aclose(self, *, timeout: float = 5.0) -> None:
        """Cancel and await any in-flight non-blocking hook tasks (idempotent)."""
        if not self._background:
            return
        pending = list(self._background)
        for task in pending:
            if not task.done():
                task.cancel()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=timeout
            )
        self._background.clear()

    async def _report(self, message: str) -> None:
        if self._error_sink is None:
            return
        try:
            result = self._error_sink(message)
            if inspect.isawaitable(result):
                await result
        except Exception:
            # The error sink must never break the loop (HK4); swallow.
            pass


__all__ = [
    "DEFAULT_HOOK_TIMEOUT_SECONDS", "HOOK_EVENT_NAMES",
    "HookConfig", "HookEvent", "HookExecutor", "HookRegistry", "HookResponse",
    "HttpExecutor", "InProcessExecutor", "SubprocessExecutor",
]  # fmt: skip
