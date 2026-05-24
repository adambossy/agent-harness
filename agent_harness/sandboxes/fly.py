"""Fly.io machine :class:`Sandbox` (SB4).

Threat model: full machine isolation per Fly.io's Firecracker-microVM
substrate. Each :class:`FlySandbox` is bound to one Fly Machine; v0.0.1
leases one machine per run / per session and tears it down on close.
Geographic placement (``region``), image pinning, and resource limits
(``cpu_kind``, ``memory_mb``) are config-time choices; egress is
controlled by the Fly app's network configuration plus
:class:`SandboxConfig.net`.

The ``httpx`` HTTP client is an *optional* dependency (extras: ``fly``).
The module imports cleanly without it; constructing :class:`FlySandbox`
without ``httpx`` raises :class:`NotSupportedError` immediately.

Communication model: Fly Machines run a small ``flyctl``-compatible HTTP
API on ``http://_api.internal:4280``; for local testing we hit the public
Machines API at ``https://api.machines.dev/v1``. File ops are implemented
on top of ``machines/{id}/exec`` shell escapes (Fly exposes no native FS
API). Reads decode UTF-8; binary reads use base64.

Cancellation contract (SB2 / SB3): the wrapper always applies
:func:`asyncio.wait_for`. ``httpx`` honors :class:`asyncio.CancelledError`
between requests but not mid-request; the outer ``wait_for`` enforces the
deadline either way.

Example:
    >>> # Real usage requires `pip install agent-harness[fly]` and FLY_API_TOKEN.
    >>> # sb = FlySandbox(app="my-agent", region="iad")  # doctest: +SKIP
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import shlex
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


def _require_httpx() -> Any:
    """Lazy-import ``httpx``; raise NotSupportedError if absent."""

    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - exercised via mocks
        raise NotSupportedError(
            "httpx is not installed; " "install with `pip install agent-harness[fly]`",
            cause=exc,
        ) from exc
    return httpx


DEFAULT_API_BASE = "https://api.machines.dev/v1"


class FlySandbox:
    """Fly.io machine-backed :class:`Sandbox` (SB4).

    Construction takes a Fly app name + an API token; the constructor
    lazy-spawns one Fly Machine the first time a method is awaited (or
    eagerly via :meth:`open`). All filesystem operations route through
    ``machines/{id}/exec`` shell escapes — Fly does not expose a native
    file API.

    For tests, pass ``http_client=<mock>`` to inject an ``httpx.AsyncClient``
    surrogate; the stub must expose async ``post`` / ``delete`` / ``aclose``
    methods that return ``httpx.Response``-shaped objects.

    Example:
        >>> # Construction is sync and lightweight; the machine is created lazily.
        >>> # sb = FlySandbox(app="agent", region="iad", api_token="…")  # doctest: +SKIP
    """

    name: str
    root: str
    config: SandboxConfig

    def __init__(
        self,
        *,
        app: str,
        api_token: str,
        root: str = "/workspace",
        config: SandboxConfig | None = None,
        region: str = "iad",
        image: str = "ghcr.io/fly-apps/agent-harness-sandbox:latest",
        cpu_kind: str = "shared",
        cpus: int = 1,
        memory_mb: int = 512,
        api_base: str = DEFAULT_API_BASE,
        name: str = "fly",
        http_client: Any | None = None,
    ) -> None:
        self._httpx: Any = _require_httpx() if http_client is None else None
        self._client: Any | None = http_client
        self._owns_client: bool = http_client is None
        self.root = root
        self.name = name
        self.config = config if config is not None else SandboxConfig()
        self._app = app
        self._api_token = api_token
        self._region = region
        self._image = image
        self._cpu_kind = cpu_kind
        self._cpus = cpus
        self._memory_mb = memory_mb
        self._api_base = api_base.rstrip("/")
        self._machine_id: str | None = None
        self._lock = asyncio.Lock()

    # --- Lifecycle --------------------------------------------------------

    async def open(self) -> None:
        """Eagerly create the underlying Fly Machine. Idempotent."""

        await self._ensure_machine()

    async def close(self) -> None:
        """Destroy the underlying Fly Machine if it was opened."""

        if self._machine_id is not None and self._client is not None:
            # Best-effort teardown; deletion failures shouldn't mask an
            # already-failing caller path.
            with contextlib.suppress(SandboxError):
                await self._request(
                    "DELETE",
                    f"/apps/{self._app}/machines/{self._machine_id}?force=true",
                )
            self._machine_id = None
        if self._owns_client and self._client is not None:
            aclose = getattr(self._client, "aclose", None)
            if aclose is not None:
                await aclose()
            self._client = None

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        self._client = self._httpx.AsyncClient(
            base_url=self._api_base,
            headers={
                "Authorization": f"Bearer {self._api_token}",
                "Content-Type": "application/json",
            },
            timeout=None,
        )
        return self._client

    async def _ensure_machine(self) -> str:
        if self._machine_id is not None:
            return self._machine_id
        async with self._lock:
            # Re-check under the lock to defeat the obvious race; another
            # coroutine may have set ``_machine_id`` while we awaited the
            # lock. The ``type: ignore`` exists because mypy cannot model
            # concurrent state mutation across the await point.
            if self._machine_id is not None:
                return self._machine_id  # type: ignore[unreachable]
            payload = {
                "region": self._region,
                "config": {
                    "image": self._image,
                    "guest": {
                        "cpu_kind": self._cpu_kind,
                        "cpus": self._cpus,
                        "memory_mb": self._memory_mb,
                    },
                    "init": {"cmd": ["sleep", "infinity"]},
                },
            }
            body = await self._request(
                "POST",
                f"/apps/{self._app}/machines",
                json_body=payload,
            )
            machine_id = body.get("id") if isinstance(body, dict) else None
            if not isinstance(machine_id, str):
                raise SandboxError(
                    "fly create-machine returned no machine id",
                    context={"response": body},
                )
            self._machine_id = machine_id
        return self._machine_id

    # --- HTTP plumbing ----------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        client = await self._ensure_client()
        if method == "POST":
            response = await client.post(path, json=json_body)
        elif method == "DELETE":
            response = await client.delete(path)
        elif method == "GET":
            response = await client.get(path)
        else:  # pragma: no cover - guard
            raise SandboxError(f"unsupported HTTP method: {method}")

        status = getattr(response, "status_code", 0)
        if not (200 <= status < 300):
            text = getattr(response, "text", str(response))
            raise SandboxError(
                f"fly api {method} {path} returned {status}",
                context={"status": status, "body": text},
            )
        try:
            return response.json()
        except (ValueError, json.JSONDecodeError):
            return {}

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

    # --- Exec primitive (file ops are built on top) ----------------------

    async def _exec_raw(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
    ) -> ExecResult:
        machine_id = await self._ensure_machine()
        payload: dict[str, Any] = {"command": cmd}
        if cwd is not None:
            payload["workdir"] = cwd
        if env:
            payload["env"] = env
        if stdin is not None:
            payload["stdin"] = stdin
        body = await self._request(
            "POST",
            f"/apps/{self._app}/machines/{machine_id}/exec",
            json_body=payload,
        )
        if not isinstance(body, dict):
            raise SandboxError("fly exec returned non-object body", context={"body": body})
        return ExecResult(
            exit_code=int(body.get("exit_code", -1)),
            stdout=str(body.get("stdout", "")),
            stderr=str(body.get("stderr", "")),
            timed_out=False,
        )

    # --- File ops (implemented over exec) --------------------------------

    async def read_file(self, path: str, *, timeout: float | None = None) -> str:
        async def _op() -> str:
            result = await self._exec_raw(["cat", _absolutize(path, self.root)])
            if result.exit_code != 0:
                raise SandboxError(
                    f"read_file failed: {path}",
                    context={"path": path, "stderr": result.stderr},
                )
            return result.stdout

        return await self._with_timeout(_op(), timeout, "read_file")

    async def read_file_bytes(
        self,
        path: str,
        *,
        timeout: float | None = None,
    ) -> bytes:
        async def _op() -> bytes:
            result = await self._exec_raw(["base64", "-w0", _absolutize(path, self.root)])
            if result.exit_code != 0:
                raise SandboxError(
                    f"read_file_bytes failed: {path}",
                    context={"path": path, "stderr": result.stderr},
                )
            try:
                return base64.b64decode(result.stdout.strip())
            except (ValueError, binascii.Error) as exc:
                raise SandboxError(
                    f"read_file_bytes: invalid base64 from {path}",
                    cause=exc,
                ) from exc

        return await self._with_timeout(_op(), timeout, "read_file_bytes")

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        timeout: float | None = None,
    ) -> None:
        async def _op() -> None:
            target = _absolutize(path, self.root)
            # ``tee`` is portable and supports stdin; we shell-quote the
            # path to avoid injection.
            cmd = [
                "sh",
                "-c",
                f"mkdir -p $(dirname {shlex.quote(target)}) && tee {shlex.quote(target)} > /dev/null",
            ]
            result = await self._exec_raw(cmd, stdin=content)
            if result.exit_code != 0:
                raise SandboxError(
                    f"write_file failed: {path}",
                    context={"path": path, "stderr": result.stderr},
                )

        await self._with_timeout(_op(), timeout, "write_file")

    async def stat(self, path: str, *, timeout: float | None = None) -> FileStat:
        async def _op() -> FileStat:
            target = _absolutize(path, self.root)
            # ``stat -c "%s %Y %F"`` portable on GNU coreutils (matches the
            # default image). Fall back to ``-f`` flags on BSD if needed.
            result = await self._exec_raw(
                ["stat", "-c", "%s %Y %F", target],
            )
            if result.exit_code != 0:
                raise SandboxError(
                    f"stat failed: {path}",
                    context={"path": path, "stderr": result.stderr},
                )
            parts = result.stdout.strip().split(maxsplit=2)
            if len(parts) != 3:
                raise SandboxError(
                    f"stat: unexpected output for {path}",
                    context={"stdout": result.stdout},
                )
            size_s, mtime_s, kind = parts
            return FileStat(
                path=target,
                size=int(size_s),
                mtime=datetime.fromtimestamp(int(mtime_s), tz=UTC),
                is_dir="directory" in kind,
            )

        return await self._with_timeout(_op(), timeout, "stat")

    async def readdir(
        self,
        path: str,
        *,
        timeout: float | None = None,
    ) -> list[FileEntry]:
        async def _op() -> list[FileEntry]:
            target = _absolutize(path, self.root)
            # ``ls -A1p`` lists hidden entries except ``.`` / ``..``, one
            # per line, with a trailing ``/`` for directories.
            result = await self._exec_raw(["ls", "-A1p", target])
            if result.exit_code != 0:
                raise SandboxError(
                    f"readdir failed: {path}",
                    context={"path": path, "stderr": result.stderr},
                )
            entries: list[FileEntry] = []
            for raw in result.stdout.splitlines():
                line = raw.rstrip("\r")
                if not line:
                    continue
                if line.endswith("/"):
                    entries.append(FileEntry(name=line[:-1], is_dir=True))
                else:
                    entries.append(FileEntry(name=line, is_dir=False))
            return entries

        return await self._with_timeout(_op(), timeout, "readdir")

    async def exists(self, path: str, *, timeout: float | None = None) -> bool:
        async def _op() -> bool:
            target = _absolutize(path, self.root)
            result = await self._exec_raw(["test", "-e", target])
            return result.exit_code == 0

        return await self._with_timeout(_op(), timeout, "exists")

    async def mkdir(
        self,
        path: str,
        *,
        parents: bool = False,
        timeout: float | None = None,
    ) -> None:
        async def _op() -> None:
            target = _absolutize(path, self.root)
            cmd = ["mkdir", "-p", target] if parents else ["mkdir", target]
            result = await self._exec_raw(cmd)
            if result.exit_code != 0:
                raise SandboxError(
                    f"mkdir failed: {path}",
                    context={"path": path, "stderr": result.stderr},
                )

        await self._with_timeout(_op(), timeout, "mkdir")

    async def rm(
        self,
        path: str,
        *,
        recursive: bool = False,
        timeout: float | None = None,
    ) -> None:
        async def _op() -> None:
            target = _absolutize(path, self.root)
            if target.rstrip("/") == self.root.rstrip("/"):
                raise SandboxError(
                    "cannot remove sandbox root",
                    context={"path": path, "root": self.root},
                )
            cmd = ["rm", "-rf", target] if recursive else ["rm", target]
            result = await self._exec_raw(cmd)
            if result.exit_code != 0:
                raise SandboxError(
                    f"rm failed: {path}",
                    context={"path": path, "stderr": result.stderr},
                )

        await self._with_timeout(_op(), timeout, "rm")

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
            return await self._exec_raw(cmd, cwd=cwd, env=env, stdin=stdin)

        return await self._with_timeout(_op(), timeout, "exec")


def _absolutize(path: str, root: str) -> str:
    """Resolve ``path`` against ``root`` if it isn't already absolute.
    Lightweight string join (Fly's machines see only their own rootfs;
    real path-jail enforcement is the machine's job)."""

    if path.startswith("/"):
        return path
    if not path or path == ".":
        return root
    return f"{root.rstrip('/')}/{path}"
