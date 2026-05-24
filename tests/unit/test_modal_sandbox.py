"""Unit tests for :class:`agent_harness.sandboxes.modal.ModalSandbox`.

The ``modal`` SDK is an optional dependency and is *not* installed in CI.
These tests inject a hand-rolled stub via the ``modal_module=`` kwarg,
which is the supported test-injection surface (see the constructor's
docstring). Two paths get explicit coverage:

1. **SDK missing → ``NotSupportedError``** at construction (lazy-import
   contract documented in the module docstring).
2. **9-method surface against the stub**, plus the timeout-primary
   cancellation contract (SB2 / SB3).
"""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import UTC, datetime
from typing import Any

import pytest

from agent_harness.core.errors import (
    NotSupportedError,
    SandboxError,
    SandboxTimeoutError,
)
from agent_harness.core.sandbox import (
    ExecResult,
    FileEntry,
    FileStat,
    Sandbox,
)
from agent_harness.sandboxes.modal import ModalSandbox

# ---------------------------------------------------------------------------
# Stub Modal SDK
# ---------------------------------------------------------------------------


class _StubProcess:
    """Stand-in for ``modal.Sandbox.exec``'s returned process object."""

    def __init__(self, *, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._exit_code = exit_code
        self.stdout = _StubStream(stdout)
        self.stderr = _StubStream(stderr)
        self.stdin_payload: str | None = None

    def wait(self) -> int:
        return self._exit_code


class _StubStream:
    def __init__(self, data: str) -> None:
        self._data = data

    def read(self) -> str:
        return self._data


class _StubModalSandbox:
    """Minimal facade matching the call sites in ``modal.py``."""

    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}
        self._dirs: set[str] = set()
        self.terminated = False
        self.last_exec: list[str] | None = None

    def read_file(self, path: str) -> bytes:
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path]

    def write_file(self, path: str, content: bytes) -> None:
        self._files[path] = content

    def stat(self, path: str) -> object:
        if path in self._files:
            return types.SimpleNamespace(
                path=path,
                size=len(self._files[path]),
                mtime=datetime(2026, 1, 1, tzinfo=UTC),
                is_dir=False,
            )
        if path in self._dirs:
            return types.SimpleNamespace(path=path, size=0, mtime=1_700_000_000, is_dir=True)
        raise FileNotFoundError(path)

    def listdir(self, path: str) -> list[object]:
        return [
            types.SimpleNamespace(name=n.split("/")[-1], is_dir=False)
            for n in self._files
            if n.startswith(path.rstrip("/") + "/")
        ]

    def exists(self, path: str) -> bool:
        return path in self._files or path in self._dirs

    def mkdir(self, path: str, parents: bool = False) -> None:
        del parents
        self._dirs.add(path)

    def rm(self, path: str, recursive: bool = False) -> None:
        del recursive
        self._files.pop(path, None)
        self._dirs.discard(path)

    def exec(self, *cmd: str, workdir: str | None = None, env: object = None) -> _StubProcess:
        del workdir, env
        self.last_exec = list(cmd)
        return _StubProcess(stdout="ok", stderr="", exit_code=0)

    def terminate(self) -> None:
        self.terminated = True


class _StubModalModule:
    """Stand-in for the ``modal`` SDK module."""

    def __init__(self) -> None:
        self._sandbox = _StubModalSandbox()

        # ``modal.App.lookup(...)`` returns an opaque object.
        class _App:
            @staticmethod
            def lookup(name: str, create_if_missing: bool = False) -> object:
                del name, create_if_missing
                return object()

        # ``modal.Image.debian_slim()`` returns an opaque image object.
        class _Image:
            @staticmethod
            def debian_slim() -> object:
                return object()

        sandbox_owner = self

        class _Sandbox:
            @staticmethod
            def create(*, image: object, app: object, workdir: str) -> _StubModalSandbox:
                del image, app, workdir
                return sandbox_owner._sandbox

        self.App = _App
        self.Image = _Image
        self.Sandbox = _Sandbox

    @property
    def underlying(self) -> _StubModalSandbox:
        return self._sandbox


# ---------------------------------------------------------------------------
# SDK-missing path
# ---------------------------------------------------------------------------


def test_missing_modal_sdk_raises_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the ``modal`` SDK installed, constructing
    :class:`ModalSandbox` must raise :class:`NotSupportedError` (lazy-
    import contract)."""

    # Pretend ``modal`` is absent even if the dev venv has it.
    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(NotSupportedError):
        ModalSandbox(root="/workspace")


# ---------------------------------------------------------------------------
# Construction + Protocol shape
# ---------------------------------------------------------------------------


def test_satisfies_sandbox_protocol() -> None:
    sb = ModalSandbox(root="/workspace", modal_module=_StubModalModule())
    assert isinstance(sb, Sandbox)
    assert sb.name == "modal"
    assert sb.root == "/workspace"


# ---------------------------------------------------------------------------
# File ops against the stub
# ---------------------------------------------------------------------------


async def test_write_then_read_round_trip() -> None:
    stub = _StubModalModule()
    sb = ModalSandbox(root="/workspace", modal_module=stub)
    await sb.write_file("a.txt", "hello")
    assert stub.underlying._files["a.txt"] == b"hello"
    assert await sb.read_file("a.txt") == "hello"


async def test_read_file_bytes_returns_bytes() -> None:
    stub = _StubModalModule()
    sb = ModalSandbox(root="/workspace", modal_module=stub)
    stub.underlying._files["bin"] = b"\x00\x01"
    assert await sb.read_file_bytes("bin") == b"\x00\x01"


async def test_stat_returns_file_stat() -> None:
    stub = _StubModalModule()
    sb = ModalSandbox(root="/workspace", modal_module=stub)
    stub.underlying._files["f"] = b"abc"
    stat = await sb.stat("f")
    assert isinstance(stat, FileStat)
    assert stat.size == 3
    assert stat.is_dir is False
    assert stat.mtime.tzinfo is not None


async def test_stat_normalizes_int_mtime() -> None:
    stub = _StubModalModule()
    sb = ModalSandbox(root="/workspace", modal_module=stub)
    stub.underlying._dirs.add("d")
    stat = await sb.stat("d")
    assert stat.is_dir is True
    assert stat.mtime.tzinfo is not None


async def test_readdir_returns_entries() -> None:
    stub = _StubModalModule()
    sb = ModalSandbox(root="/workspace", modal_module=stub)
    stub.underlying._files["/workspace/a"] = b""
    stub.underlying._files["/workspace/b"] = b""
    entries = await sb.readdir("/workspace")
    names = sorted(e.name for e in entries)
    assert names == ["a", "b"]
    assert all(isinstance(e, FileEntry) for e in entries)


async def test_exists_true_and_false() -> None:
    stub = _StubModalModule()
    sb = ModalSandbox(root="/workspace", modal_module=stub)
    stub.underlying._files["yes"] = b""
    assert await sb.exists("yes") is True
    assert await sb.exists("no") is False


async def test_mkdir_and_rm() -> None:
    stub = _StubModalModule()
    sb = ModalSandbox(root="/workspace", modal_module=stub)
    await sb.mkdir("d", parents=True)
    assert "d" in stub.underlying._dirs
    await sb.rm("d", recursive=True)
    assert "d" not in stub.underlying._dirs


async def test_exec_returns_exec_result() -> None:
    stub = _StubModalModule()
    sb = ModalSandbox(root="/workspace", modal_module=stub)
    result = await sb.exec(["echo", "hi"])
    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout == "ok"
    assert stub.underlying.last_exec == ["echo", "hi"]


async def test_exec_rejects_empty_cmd() -> None:
    sb = ModalSandbox(root="/workspace", modal_module=_StubModalModule())
    with pytest.raises(SandboxError):
        await sb.exec([])


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


async def test_underlying_failure_wrapped_as_sandbox_error() -> None:
    stub = _StubModalModule()
    sb = ModalSandbox(root="/workspace", modal_module=stub)

    def _boom(path: str) -> bytes:
        raise RuntimeError(f"boom-{path}")

    stub.underlying.read_file = _boom  # type: ignore[method-assign]
    with pytest.raises(SandboxError, match="modal read_file failed"):
        await sb.read_file("nope")


# ---------------------------------------------------------------------------
# Timeout contract (SB2 / SB3)
# ---------------------------------------------------------------------------


async def test_timeout_translates_to_sandbox_timeout_error() -> None:
    stub = _StubModalModule()
    sb = ModalSandbox(root="/workspace", modal_module=stub)

    async def _slow(self: object, path: str) -> str:
        del self, path
        await asyncio.sleep(5)
        return ""

    # Patch the *method on the instance* to be slow.
    async def _slow_read(_path: str) -> bytes:
        await asyncio.sleep(5)
        return b""

    stub.underlying.read_file = _slow_read  # type: ignore[method-assign,assignment]
    with pytest.raises(SandboxTimeoutError):
        await sb.read_file("anything", timeout=0.05)


async def test_close_terminates_underlying_sandbox() -> None:
    stub = _StubModalModule()
    sb = ModalSandbox(root="/workspace", modal_module=stub)
    await sb.open()
    assert stub.underlying.terminated is False
    await sb.close()
    assert stub.underlying.terminated is True


def test_underlying_stub_imports_typed() -> None:
    """Smoke test: the stub module is importable and exports the
    expected attributes so other tests can subclass / extend it."""

    stub = _StubModalModule()
    assert callable(stub.App.lookup)
    assert callable(stub.Image.debian_slim)
    assert callable(stub.Sandbox.create)


# Suppress an unused-import-style guard if the assertion plumbing
# changes; ``Any`` is referenced in the stubs above.
_ = Any
