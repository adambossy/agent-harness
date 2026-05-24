"""In-process sandbox — host filesystem rooted at a directory.

Threat model: **none**. ``InProcessSandbox`` operates directly on the host
filesystem under a configured ``root``. It enforces a *path jail* (SB7) so
operations cannot escape ``root`` via ``..`` or absolute paths, but offers no
isolation against malicious processes, signal handling, resource exhaustion,
or network egress. Use it for tests, development, and single-user CLIs where
the user trusts the agent (SB8).

The implementation honors the 9-method :class:`Sandbox` Protocol from
``core/sandbox.py`` and the *timeout-primary* cancellation contract (SB2 /
SB3): every method wraps its work in :func:`asyncio.wait_for`. Blocking
filesystem calls are dispatched via :func:`asyncio.to_thread`; ``exec`` uses
:func:`asyncio.create_subprocess_exec` (no shell interpretation).

Example:
    >>> import asyncio, tempfile
    >>> from agent_harness.sandboxes.inprocess import InProcessSandbox
    >>> async def _demo() -> str:
    ...     with tempfile.TemporaryDirectory() as tmp:
    ...         sb = InProcessSandbox(root=tmp)
    ...         await sb.write_file("hello.txt", "hi")
    ...         return await sb.read_file("hello.txt")
    >>> asyncio.run(_demo())
    'hi'
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path

from agent_harness.core.errors import SandboxError, SandboxTimeoutError
from agent_harness.core.sandbox import (
    ExecResult,
    FileEntry,
    FileStat,
    SandboxConfig,
)


class InProcessSandbox:
    """In-process :class:`Sandbox` rooted at a host directory (SB4).

    Path arguments are interpreted relative to :attr:`root`. Absolute paths
    are accepted iff they resolve under :attr:`root`; any path that escapes
    the root (via ``..``, symlink, or absolute path outside the tree) is
    rejected with :class:`SandboxError` (SB7).

    The class is safe to construct without ``modal`` / ``httpx`` installed
    (it has no optional dependencies). Construction creates the root if it
    does not already exist; pass ``create_root=False`` to require an
    existing directory.

    Example:
        >>> import asyncio, tempfile
        >>> async def _ok() -> bool:
        ...     with tempfile.TemporaryDirectory() as tmp:
        ...         sb = InProcessSandbox(root=tmp)
        ...         return await sb.exists(".")
        >>> asyncio.run(_ok())
        True
    """

    name: str
    root: str
    config: SandboxConfig

    def __init__(
        self,
        *,
        root: str | Path,
        config: SandboxConfig | None = None,
        name: str = "in-process",
        create_root: bool = True,
    ) -> None:
        resolved = Path(root).expanduser().resolve()
        if create_root:
            resolved.mkdir(parents=True, exist_ok=True)
        elif not resolved.is_dir():
            raise SandboxError(
                f"root does not exist or is not a directory: {resolved}",
                context={"root": str(resolved)},
            )
        self._root_path: Path = resolved
        self.root = str(resolved)
        self.name = name
        self.config = config if config is not None else SandboxConfig()

    # --- Path jail (SB7) --------------------------------------------------

    def _resolve_inside_root(self, path: str) -> Path:
        """Resolve ``path`` (relative or absolute) and confirm it sits
        under :attr:`root`. Raises :class:`SandboxError` if it escapes."""

        candidate = Path(path)
        joined = candidate if candidate.is_absolute() else self._root_path / candidate
        # ``resolve(strict=False)`` does not require the path to exist but
        # still normalizes ``..`` and follows symlinks where present.
        resolved = joined.resolve(strict=False)
        if resolved != self._root_path and not resolved.is_relative_to(self._root_path):
            raise SandboxError(
                f"path escapes sandbox root: {path!r}",
                context={"path": path, "root": self.root},
            )
        return resolved

    # --- Timeout helper ---------------------------------------------------

    @staticmethod
    async def _with_timeout[T](
        coro: Coroutine[object, object, T],
        timeout: float | None,
        op: str,
    ) -> T:
        """Enforce the outer deadline (SB3). ``coro`` is an awaitable
        producing a ``T``; this helper centralizes the
        :func:`asyncio.wait_for` wrap and translates :class:`TimeoutError`
        into :class:`SandboxTimeoutError`."""

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

    # --- File ops ---------------------------------------------------------

    async def read_file(self, path: str, *, timeout: float | None = None) -> str:
        target = self._resolve_inside_root(path)

        def _read() -> str:
            try:
                return target.read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                raise SandboxError(
                    f"file not found: {path}",
                    context={"path": path},
                    cause=exc,
                ) from exc
            except OSError as exc:
                raise SandboxError(
                    f"read failed: {path}: {exc}",
                    context={"path": path},
                    cause=exc,
                ) from exc

        return await self._with_timeout(asyncio.to_thread(_read), timeout, "read_file")

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        timeout: float | None = None,
    ) -> None:
        target = self._resolve_inside_root(path)

        def _write() -> None:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            except OSError as exc:
                raise SandboxError(
                    f"write failed: {path}: {exc}",
                    context={"path": path},
                    cause=exc,
                ) from exc

        await self._with_timeout(asyncio.to_thread(_write), timeout, "write_file")

    async def stat(self, path: str, *, timeout: float | None = None) -> FileStat:
        target = self._resolve_inside_root(path)

        def _stat() -> FileStat:
            try:
                st = target.stat()
            except FileNotFoundError as exc:
                raise SandboxError(
                    f"path not found: {path}",
                    context={"path": path},
                    cause=exc,
                ) from exc
            return FileStat(
                path=str(target),
                size=st.st_size,
                mtime=datetime.fromtimestamp(st.st_mtime, tz=UTC),
                is_dir=target.is_dir(),
            )

        return await self._with_timeout(asyncio.to_thread(_stat), timeout, "stat")

    async def readdir(
        self,
        path: str,
        *,
        timeout: float | None = None,
    ) -> list[FileEntry]:
        target = self._resolve_inside_root(path)

        def _readdir() -> list[FileEntry]:
            if not target.exists():
                raise SandboxError(
                    f"directory not found: {path}",
                    context={"path": path},
                )
            if not target.is_dir():
                raise SandboxError(
                    f"not a directory: {path}",
                    context={"path": path},
                )
            entries = sorted(target.iterdir(), key=lambda p: p.name)
            return [FileEntry(name=p.name, is_dir=p.is_dir()) for p in entries]

        return await self._with_timeout(asyncio.to_thread(_readdir), timeout, "readdir")

    async def exists(self, path: str, *, timeout: float | None = None) -> bool:
        target = self._resolve_inside_root(path)

        def _exists() -> bool:
            return target.exists()

        return await self._with_timeout(asyncio.to_thread(_exists), timeout, "exists")

    async def mkdir(
        self,
        path: str,
        *,
        parents: bool = False,
        timeout: float | None = None,
    ) -> None:
        target = self._resolve_inside_root(path)

        def _mkdir() -> None:
            try:
                target.mkdir(parents=parents, exist_ok=parents)
            except FileExistsError as exc:
                raise SandboxError(
                    f"already exists: {path}",
                    context={"path": path},
                    cause=exc,
                ) from exc
            except OSError as exc:
                raise SandboxError(
                    f"mkdir failed: {path}: {exc}",
                    context={"path": path},
                    cause=exc,
                ) from exc

        await self._with_timeout(asyncio.to_thread(_mkdir), timeout, "mkdir")

    async def rm(
        self,
        path: str,
        *,
        recursive: bool = False,
        timeout: float | None = None,
    ) -> None:
        target = self._resolve_inside_root(path)
        if target == self._root_path:
            raise SandboxError(
                "cannot remove sandbox root",
                context={"path": path, "root": self.root},
            )

        def _rm() -> None:
            try:
                if target.is_dir() and not target.is_symlink():
                    if recursive:
                        shutil.rmtree(target)
                    else:
                        target.rmdir()
                else:
                    target.unlink()
            except FileNotFoundError as exc:
                raise SandboxError(
                    f"path not found: {path}",
                    context={"path": path},
                    cause=exc,
                ) from exc
            except OSError as exc:
                raise SandboxError(
                    f"rm failed: {path}: {exc}",
                    context={"path": path},
                    cause=exc,
                ) from exc

        await self._with_timeout(asyncio.to_thread(_rm), timeout, "rm")

    async def read_file_bytes(
        self,
        path: str,
        *,
        timeout: float | None = None,
    ) -> bytes:
        target = self._resolve_inside_root(path)

        def _read_bytes() -> bytes:
            try:
                return target.read_bytes()
            except FileNotFoundError as exc:
                raise SandboxError(
                    f"file not found: {path}",
                    context={"path": path},
                    cause=exc,
                ) from exc
            except OSError as exc:
                raise SandboxError(
                    f"read failed: {path}: {exc}",
                    context={"path": path},
                    cause=exc,
                ) from exc

        return await self._with_timeout(asyncio.to_thread(_read_bytes), timeout, "read_file_bytes")

    # --- Exec -------------------------------------------------------------

    def _filter_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        """Apply network-config filtering when relevant. v0.0.1 defers
        real seccomp/iptables enforcement to container backends; the
        in-process path only does best-effort env filtering."""

        if env is None:
            # Inherit the parent process's environment by default.
            return None
        return dict(env)

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
        cwd_path = self._resolve_inside_root(cwd) if cwd is not None else self._root_path
        if not cwd_path.is_dir():
            raise SandboxError(
                f"cwd is not a directory: {cwd}",
                context={"cwd": cwd},
            )

        async def _spawn_and_wait() -> ExecResult:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd_path),
                env=self._filter_env(env) if env is not None else os.environ.copy(),
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdin_bytes = stdin.encode("utf-8") if stdin is not None else None
            try:
                out, err = await proc.communicate(stdin_bytes)
            except asyncio.CancelledError:
                # On cancellation (typically wait_for timeout), kill the
                # process so it doesn't dangle. The harness's outer
                # ``wait_for`` will translate the timeout to
                # ``SandboxTimeoutError``.
                proc.kill()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(proc.wait())
                raise
            return ExecResult(
                exit_code=proc.returncode if proc.returncode is not None else -1,
                stdout=out.decode("utf-8", errors="replace"),
                stderr=err.decode("utf-8", errors="replace"),
                timed_out=False,
            )

        return await self._with_timeout(_spawn_and_wait(), timeout, "exec")
