"""Modal-container :class:`Sandbox` (SB4).

Threat model: full container isolation per Modal's container substrate
(seccomp, namespaces, dedicated rootfs, opt-in egress). Each
``ModalSandbox`` is bound to one Modal *Sandbox* object (their primitive,
not ours); v0.0.1 leases one container per run / per session and tears it
down on close. Container image pinning, egress, and resource limits are
configured through :class:`SandboxConfig` plus Modal-specific kwargs.

The ``modal`` SDK is an *optional* dependency (extras: ``modal``). The
module imports cleanly without it; constructing :class:`ModalSandbox`
without the SDK raises :class:`NotSupportedError` immediately so the
failure surfaces at config time, not on the first call.

Cancellation contract (SB2 / SB3): the wrapper always applies
:func:`asyncio.wait_for`, which translates expiry to
:class:`SandboxTimeoutError`. Modal's underlying SDK calls *may* ignore
:class:`asyncio.CancelledError`; that is OK — the outer ``wait_for``
guarantees abort timing.

Example:
    >>> # Real usage requires `pip install agent-harness[modal]` and a Modal token.
    >>> # sb = ModalSandbox(root="/workspace", app_name="my-agent")  # doctest: +SKIP
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any

from agent_harness.core.errors import (
    NotSupportedError,
    SandboxError,
    SandboxTimeoutError,
)
from agent_harness.core.sandbox import (
    ExecResult,
    FileEntry,
    FileStat,
    SandboxConfig,
)


def _require_modal_sdk() -> Any:
    """Lazy-import the ``modal`` SDK; raise NotSupportedError if absent."""

    try:
        import modal
    except ImportError as exc:  # pragma: no cover - exercised via mocks
        raise NotSupportedError(
            "modal SDK is not installed; " "install with `pip install agent-harness[modal]`",
            cause=exc,
        ) from exc
    return modal


class ModalSandbox:
    """Modal-container-backed :class:`Sandbox` (SB4).

    Construction takes a Modal *App* (or its name) and an optional
    pre-built ``modal.Image``; the constructor lazy-spawns one Modal
    Sandbox object the first time a method is awaited (or eagerly via
    :meth:`open`).

    For tests, pass ``modal_module=<mock>`` to inject a stub in place of
    the real SDK; the stub must expose ``App``, ``Image``, ``Sandbox`` and
    a ``current_app_name``/``Image.debian_slim`` surface that is
    sufficient for the call sites the test exercises.

    Example:
        >>> # Construction is sync and lightweight; the container is
        >>> # created lazily.
        >>> # sb = ModalSandbox(root="/workspace", app_name="agent")  # doctest: +SKIP
    """

    name: str
    root: str
    config: SandboxConfig

    def __init__(
        self,
        *,
        root: str = "/workspace",
        config: SandboxConfig | None = None,
        app_name: str = "agent-harness-sandbox",
        image: Any | None = None,
        name: str = "modal",
        modal_module: Any | None = None,
    ) -> None:
        self._modal: Any = modal_module if modal_module is not None else _require_modal_sdk()
        self.root = root
        self.name = name
        self.config = config if config is not None else SandboxConfig()
        self._app_name: str = app_name
        self._image: Any | None = image
        self._sandbox: Any | None = None
        self._lock = asyncio.Lock()

    # --- Lifecycle --------------------------------------------------------

    async def open(self) -> None:
        """Eagerly create the underlying Modal Sandbox. Idempotent."""

        await self._ensure_sandbox()

    async def close(self) -> None:
        """Terminate the underlying Modal Sandbox if it was opened."""

        if self._sandbox is None:
            return
        terminate = getattr(self._sandbox, "terminate", None)
        if terminate is None:
            self._sandbox = None
            return
        try:
            # Dispatch the (possibly blocking) terminate via to_thread, then
            # await if it returned a coroutine.
            await self._to_thread_maybe(terminate)
        finally:
            self._sandbox = None

    async def _ensure_sandbox(self) -> Any:
        if self._sandbox is not None:
            return self._sandbox
        async with self._lock:
            # Re-check under the lock to defeat the obvious race — another
            # coroutine may have set ``_sandbox`` while we were waiting.
            if self._sandbox is None:
                modal = self._modal
                image = self._image or modal.Image.debian_slim()
                # SDK calls below may be blocking gRPC — run on a worker
                # thread, then await if they returned a coroutine. Keeps the
                # event loop responsive during the (one-shot) container spin.
                app = await self._to_thread_maybe(
                    modal.App.lookup,
                    self._app_name,
                    create_if_missing=True,
                )
                self._sandbox = await self._to_thread_maybe(
                    modal.Sandbox.create,
                    image=image,
                    app=app,
                    workdir=self.root,
                )
            return self._sandbox

    # --- Timeout helper ---------------------------------------------------

    @staticmethod
    async def _with_timeout[T](
        coro: Coroutine[object, object, T],
        timeout: float | None,
        op: str,
    ) -> T:
        if timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except TimeoutError as exc:
            raise SandboxTimeoutError(
                f"{op} exceeded timeout of {timeout}s",
                context={"op": op, "timeout": timeout},
                cause=exc,
            ) from exc

    @staticmethod
    async def _await_maybe(value: Any) -> Any:
        """Some Modal calls are sync, others async; this helper awaits
        when needed."""
        if asyncio.iscoroutine(value):
            return await value
        return value

    @staticmethod
    async def _to_thread_maybe(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Invoke a Modal SDK method on a worker thread, then ``await`` if it
        returned a coroutine.

        The Modal SDK ships both sync and async variants of the same name in
        different versions; some are thin gRPC-blocking wrappers, others
        return awaitables. Dispatching the call itself through
        :func:`asyncio.to_thread` keeps the event loop responsive in the
        blocking case; the ``await self._await_maybe(...)`` handles the
        async case. The combination is uniformly correct.
        """
        result = await asyncio.to_thread(fn, *args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    # --- File ops ---------------------------------------------------------

    async def read_file(self, path: str, *, timeout: float | None = None) -> str:
        async def _op() -> str:
            sb = await self._ensure_sandbox()
            data = await self._to_thread_maybe(sb.read_file, path)
            if isinstance(data, bytes):
                return data.decode("utf-8")
            return str(data)

        try:
            return await self._with_timeout(_op(), timeout, "read_file")
        except SandboxTimeoutError:
            raise
        except Exception as exc:
            raise SandboxError(f"modal read_file failed: {path}", cause=exc) from exc

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        timeout: float | None = None,
    ) -> None:
        async def _op() -> None:
            sb = await self._ensure_sandbox()
            await self._to_thread_maybe(sb.write_file, path, content.encode("utf-8"))

        try:
            await self._with_timeout(_op(), timeout, "write_file")
        except SandboxTimeoutError:
            raise
        except Exception as exc:
            raise SandboxError(f"modal write_file failed: {path}", cause=exc) from exc

    async def stat(self, path: str, *, timeout: float | None = None) -> FileStat:
        async def _op() -> FileStat:
            sb = await self._ensure_sandbox()
            raw = await self._to_thread_maybe(sb.stat, path)
            return FileStat(
                path=getattr(raw, "path", path),
                size=int(getattr(raw, "size", 0)),
                mtime=_coerce_mtime(getattr(raw, "mtime", None)),
                is_dir=bool(getattr(raw, "is_dir", False)),
            )

        try:
            return await self._with_timeout(_op(), timeout, "stat")
        except SandboxTimeoutError:
            raise
        except Exception as exc:
            raise SandboxError(f"modal stat failed: {path}", cause=exc) from exc

    async def readdir(
        self,
        path: str,
        *,
        timeout: float | None = None,
    ) -> list[FileEntry]:
        async def _op() -> list[FileEntry]:
            sb = await self._ensure_sandbox()
            raw = await self._to_thread_maybe(sb.listdir, path)
            return [
                FileEntry(
                    name=getattr(entry, "name", str(entry)),
                    is_dir=bool(getattr(entry, "is_dir", False)),
                )
                for entry in raw
            ]

        try:
            return await self._with_timeout(_op(), timeout, "readdir")
        except SandboxTimeoutError:
            raise
        except Exception as exc:
            raise SandboxError(f"modal readdir failed: {path}", cause=exc) from exc

    async def exists(self, path: str, *, timeout: float | None = None) -> bool:
        async def _op() -> bool:
            sb = await self._ensure_sandbox()
            return bool(await self._to_thread_maybe(sb.exists, path))

        try:
            return await self._with_timeout(_op(), timeout, "exists")
        except SandboxTimeoutError:
            raise
        except Exception as exc:
            raise SandboxError(f"modal exists failed: {path}", cause=exc) from exc

    async def mkdir(
        self,
        path: str,
        *,
        parents: bool = False,
        timeout: float | None = None,
    ) -> None:
        async def _op() -> None:
            sb = await self._ensure_sandbox()
            await self._to_thread_maybe(sb.mkdir, path, parents=parents)

        try:
            await self._with_timeout(_op(), timeout, "mkdir")
        except SandboxTimeoutError:
            raise
        except Exception as exc:
            raise SandboxError(f"modal mkdir failed: {path}", cause=exc) from exc

    async def rm(
        self,
        path: str,
        *,
        recursive: bool = False,
        timeout: float | None = None,
    ) -> None:
        async def _op() -> None:
            sb = await self._ensure_sandbox()
            await self._to_thread_maybe(sb.rm, path, recursive=recursive)

        try:
            await self._with_timeout(_op(), timeout, "rm")
        except SandboxTimeoutError:
            raise
        except Exception as exc:
            raise SandboxError(f"modal rm failed: {path}", cause=exc) from exc

    async def read_file_bytes(
        self,
        path: str,
        *,
        timeout: float | None = None,
    ) -> bytes:
        async def _op() -> bytes:
            sb = await self._ensure_sandbox()
            data = await self._to_thread_maybe(sb.read_file, path)
            if isinstance(data, bytes):
                return data
            return str(data).encode("utf-8")

        try:
            return await self._with_timeout(_op(), timeout, "read_file_bytes")
        except SandboxTimeoutError:
            raise
        except Exception as exc:
            raise SandboxError(f"modal read_file_bytes failed: {path}", cause=exc) from exc

    # --- Exec -------------------------------------------------------------

    async def exec(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        if not cmd:
            raise SandboxError("exec: cmd must be a non-empty argv list")

        async def _op() -> ExecResult:
            sb = await self._ensure_sandbox()
            # Modal's ``Sandbox.exec`` may be a sync gRPC-blocking call —
            # dispatch via to_thread, then await if it returned a coroutine.
            proc = await self._to_thread_maybe(
                sb.exec,
                *cmd,
                workdir=cwd if cwd is not None else self.root,
                env=env,
            )
            if stdin is not None and hasattr(proc, "stdin"):
                await self._to_thread_maybe(proc.stdin.write, stdin)
                close = getattr(proc.stdin, "drain_and_close", None) or getattr(
                    proc.stdin, "close", None
                )
                if close is not None:
                    await self._to_thread_maybe(close)
            exit_code = await self._to_thread_maybe(proc.wait)
            out_raw = await self._to_thread_maybe(proc.stdout.read)
            err_raw = await self._to_thread_maybe(proc.stderr.read)
            return ExecResult(
                exit_code=int(exit_code if exit_code is not None else -1),
                stdout=_to_text(out_raw),
                stderr=_to_text(err_raw),
                timed_out=False,
            )

        try:
            return await self._with_timeout(_op(), timeout, "exec")
        except SandboxTimeoutError:
            raise
        except Exception as exc:
            raise SandboxError(f"modal exec failed: {' '.join(cmd)}", cause=exc) from exc


def _coerce_mtime(value: Any) -> datetime:
    """Modal returns mtime in heterogeneous shapes (epoch, naive
    datetime). Normalize to a timezone-aware UTC datetime."""

    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC)
    return datetime.now(UTC)


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value) if value is not None else ""
