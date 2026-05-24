"""Unit tests for ``agent_harness.core.sandbox`` — Protocol + config types only.

Concrete sandbox implementations (``InProcessSandbox``, ``ModalSandbox``,
``FlySandbox``) ship in Wave 3 under ``agent_harness.sandboxes``; this
module covers only the structural shape of the Protocol and the
declarative config / value types.
"""

from __future__ import annotations

import dataclasses
import inspect
from datetime import UTC, datetime

import pytest

from agent_harness.core.sandbox import (
    ExecResult,
    FileEntry,
    FileStat,
    Sandbox,
    SandboxConfig,
    SandboxFilesystemConfig,
    SandboxNetworkConfig,
)

# ---------------------------------------------------------------------------
# A minimal compliant stand-in. All methods exist with the documented
# signatures; bodies are trivial. Used to exercise the structural
# ``isinstance(x, Sandbox)`` check and the per-method timeout kwarg.
# ---------------------------------------------------------------------------


class _StubSandbox:
    """Reference structural implementation. Wave 3 replaces this with real
    backends; here it exists only to validate the Protocol shape."""

    name = "stub"
    root = "/workspace"
    config = SandboxConfig()

    async def read_file(self, path: str, *, timeout: float | None = None) -> str:
        return f"contents-of:{path}"

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        timeout: float | None = None,
    ) -> None:
        return None

    async def stat(self, path: str, *, timeout: float | None = None) -> FileStat:
        return FileStat(
            path=path,
            size=0,
            mtime=datetime(2026, 1, 1, tzinfo=UTC),
            is_dir=False,
        )

    async def readdir(
        self,
        path: str,
        *,
        timeout: float | None = None,
    ) -> list[FileEntry]:
        return [FileEntry(name="a.txt", is_dir=False)]

    async def exists(self, path: str, *, timeout: float | None = None) -> bool:
        return path == "/workspace"

    async def mkdir(
        self,
        path: str,
        *,
        parents: bool = False,
        timeout: float | None = None,
    ) -> None:
        return None

    async def rm(
        self,
        path: str,
        *,
        recursive: bool = False,
        timeout: float | None = None,
    ) -> None:
        return None

    async def read_file_bytes(
        self,
        path: str,
        *,
        timeout: float | None = None,
    ) -> bytes:
        return b"\x00\x01"

    async def exec(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        return ExecResult(exit_code=0, stdout=" ".join(cmd), stderr="")


# ---------------------------------------------------------------------------
# Config types
# ---------------------------------------------------------------------------


def test_filesystem_config_defaults_match_spec() -> None:
    cfg = SandboxFilesystemConfig()
    # Reads default-open, writes default-closed (safe-by-default).
    assert cfg.allow_read == ["**"]
    assert cfg.allow_write == []
    assert cfg.deny_read == []
    assert cfg.deny_write == []


def test_filesystem_config_default_factory_is_per_instance() -> None:
    """``default_factory`` semantics: each instance gets its own list, so
    mutating one default doesn't leak into another."""

    a = SandboxFilesystemConfig()
    b = SandboxFilesystemConfig()
    a.allow_read.append("danger/**")
    assert b.allow_read == ["**"]


def test_network_config_defaults_match_spec() -> None:
    cfg = SandboxNetworkConfig()
    assert cfg.allowed_domains == []
    assert cfg.allow_unix_sockets is False
    assert cfg.http_proxy_port is None
    assert cfg.socks_proxy_port is None
    assert cfg.seccomp_enabled is True


def test_sandbox_config_composes_defaults() -> None:
    cfg = SandboxConfig()
    assert isinstance(cfg.fs, SandboxFilesystemConfig)
    assert isinstance(cfg.net, SandboxNetworkConfig)
    assert cfg.fs.allow_read == ["**"]
    assert cfg.net.seccomp_enabled is True


def test_sandbox_config_rejects_unknown_fields() -> None:
    """All three config models use ``extra='forbid'``."""

    with pytest.raises(ValueError):
        # Rationale for the ignore below: ``port`` is not a declared field;
        # the test asserts ``extra='forbid'`` rejects it at validation time.
        SandboxNetworkConfig(port=9999)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        SandboxFilesystemConfig(unknown=True)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        SandboxConfig(extra_field=1)  # type: ignore[call-arg]


def test_sandbox_config_round_trips_via_json() -> None:
    """The composite config must survive json roundtrip — it's the shape
    the loop persists alongside the snapshot when re-attaching a sandbox."""

    cfg = SandboxConfig(
        fs=SandboxFilesystemConfig(allow_write=["src/**"]),
        net=SandboxNetworkConfig(allowed_domains=["api.example.com"], seccomp_enabled=False),
    )
    parsed = SandboxConfig.model_validate_json(cfg.model_dump_json())
    assert parsed.fs.allow_write == ["src/**"]
    assert parsed.net.allowed_domains == ["api.example.com"]
    assert parsed.net.seccomp_enabled is False


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


def test_file_stat_is_frozen_dataclass() -> None:
    stat = FileStat(
        path="/workspace/x", size=42, mtime=datetime(2026, 1, 1, tzinfo=UTC), is_dir=False
    )
    # FrozenInstanceError (dataclasses) inherits AttributeError.
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        stat.size = 100  # type: ignore[misc]


def test_file_entry_carries_only_name_and_is_dir() -> None:
    entry = FileEntry(name="README.md", is_dir=False)
    assert entry.name == "README.md"
    assert entry.is_dir is False


def test_exec_result_defaults_timed_out_false() -> None:
    r = ExecResult(exit_code=0, stdout="ok", stderr="")
    assert r.timed_out is False


def test_exec_result_timed_out_path() -> None:
    """An exec call that hit its deadline reports ``timed_out=True`` and an
    implementation-defined non-zero exit code."""

    r = ExecResult(exit_code=-1, stdout="", stderr="killed", timed_out=True)
    assert r.timed_out is True
    assert r.exit_code != 0


# ---------------------------------------------------------------------------
# Sandbox Protocol — structural / runtime_checkable shape
# ---------------------------------------------------------------------------


def test_stub_sandbox_satisfies_protocol() -> None:
    """A class that supplies all 9 methods + the three attributes is a
    structural ``Sandbox``."""

    assert isinstance(_StubSandbox(), Sandbox)


def test_missing_method_fails_protocol_check() -> None:
    """Drop ``exec`` — the structural check rejects the class."""

    class _NoExec:
        name = "broken"
        root = "/workspace"
        config = SandboxConfig()

        async def read_file(self, path: str, *, timeout: float | None = None) -> str:
            return ""

        async def write_file(
            self, path: str, content: str, *, timeout: float | None = None
        ) -> None:
            return None

        async def stat(self, path: str, *, timeout: float | None = None) -> FileStat:
            return FileStat(path=path, size=0, mtime=datetime(2026, 1, 1, tzinfo=UTC), is_dir=False)

        async def readdir(self, path: str, *, timeout: float | None = None) -> list[FileEntry]:
            return []

        async def exists(self, path: str, *, timeout: float | None = None) -> bool:
            return False

        async def mkdir(
            self, path: str, *, parents: bool = False, timeout: float | None = None
        ) -> None:
            return None

        async def rm(
            self, path: str, *, recursive: bool = False, timeout: float | None = None
        ) -> None:
            return None

        async def read_file_bytes(self, path: str, *, timeout: float | None = None) -> bytes:
            return b""

    # ``exec`` is missing → fails the structural Protocol check.
    assert not isinstance(_NoExec(), Sandbox)


def test_protocol_exposes_exactly_nine_methods() -> None:
    """SB1: the contract is 9 methods, no more, no less. Guard against
    accidental surface growth."""

    expected = {
        "read_file",
        "write_file",
        "stat",
        "readdir",
        "exists",
        "mkdir",
        "rm",
        "read_file_bytes",
        "exec",
    }
    actual = {
        name
        for name, value in inspect.getmembers(Sandbox, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert actual == expected


@pytest.mark.parametrize(
    "method_name",
    [
        "read_file",
        "write_file",
        "stat",
        "readdir",
        "exists",
        "mkdir",
        "rm",
        "read_file_bytes",
        "exec",
    ],
)
def test_every_method_accepts_a_timeout_kwarg(method_name: str) -> None:
    """SB2: every file op + exec accepts a keyword-only ``timeout`` of
    ``float | None`` defaulting to ``None``."""

    method = getattr(Sandbox, method_name)
    sig = inspect.signature(method)
    assert "timeout" in sig.parameters, f"{method_name} is missing the timeout kwarg"
    param = sig.parameters["timeout"]
    assert param.kind == inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


def test_protocol_documents_timeout_primary_cancellation() -> None:
    """SB2 / SB3: the Protocol docstring must call out the timeout-primary
    contract — implementers consult the docstring, so it's load-bearing."""

    assert Sandbox.__doc__ is not None
    doc = Sandbox.__doc__.lower()
    assert "timeout" in doc
    assert "primary" in doc
    # CancelledError is best-effort, not required.
    assert "cancelledError".lower() in doc or "cancellation" in doc


# ---------------------------------------------------------------------------
# Smoke: the stub's methods are awaitable and return the documented types.
# ---------------------------------------------------------------------------


async def test_stub_sandbox_read_file_returns_str() -> None:
    sb = _StubSandbox()
    out = await sb.read_file("/workspace/a.txt")
    assert out == "contents-of:/workspace/a.txt"


async def test_stub_sandbox_readdir_returns_file_entries() -> None:
    sb = _StubSandbox()
    entries = await sb.readdir("/workspace")
    assert len(entries) == 1
    assert isinstance(entries[0], FileEntry)


async def test_stub_sandbox_exec_returns_exec_result() -> None:
    sb = _StubSandbox()
    result = await sb.exec(["echo", "hi"])
    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout == "echo hi"
    assert result.timed_out is False


async def test_stub_sandbox_read_file_bytes_returns_bytes() -> None:
    sb = _StubSandbox()
    out = await sb.read_file_bytes("/workspace/bin")
    assert isinstance(out, bytes)


async def test_stub_sandbox_timeout_kwarg_is_passed_through() -> None:
    """The Protocol declares ``timeout`` is keyword-only; the stub accepts
    it by keyword, validating call shape."""

    sb = _StubSandbox()
    await sb.write_file("/workspace/a.txt", "x", timeout=1.0)
    await sb.mkdir("/workspace/d", parents=True, timeout=2.0)
    await sb.rm("/workspace/d", recursive=True, timeout=3.0)
