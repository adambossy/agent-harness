"""Universal Sandbox Protocol and declarative config shapes (Layer 0).

This module owns *only* the contract every sandbox backend must satisfy and
the typed config shapes that describe its filesystem / network policy
declaratively.

A :class:`Sandbox` exposes nine async operations — eight file operations plus
``exec`` — and nothing else. Filesystem *tools* (read / write / edit / grep /
glob / list_dir) that the model sees live in a separate Layer-1 ``Toolset``
that targets the active sandbox (FilesystemTools, Wave 3); they are
deliberately not defined here (SB9).

Cancellation contract (Flue's lesson, SB2 / SB3): **timeout is primary.**
Every method accepts a ``timeout: float | None`` keyword. Implementations
*may* additionally honor :class:`asyncio.CancelledError`, but they are not
required to. The harness is responsible for enforcing the deadline from the
outside (typically via :func:`asyncio.wait_for`) so even backends whose
underlying SDKs ignore cancellation abort calls at the configured time.

Concrete implementations (``InProcessSandbox``, ``ModalSandbox``,
``FlySandbox``) ship in ``agent_harness.sandboxes`` and are added in Wave 3;
this module deliberately defines no concrete subclass.

Example:
    >>> cfg = SandboxConfig()
    >>> cfg.fs.allow_read
    ['**']
    >>> cfg.net.seccomp_enabled
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Declarative configuration (Claude Code's pattern — fat configs as data)
# ---------------------------------------------------------------------------


class SandboxFilesystemConfig(BaseModel):
    """Path-scoping policy for a sandbox's filesystem surface.

    Allow / deny lists are glob patterns; deny overrides allow. By default
    reads are unrestricted (``["**"]``) and writes are denied (empty allow
    list). Concrete implementations enforce these patterns; the Protocol
    itself only describes the shape.

    Example:
        >>> cfg = SandboxFilesystemConfig(allow_write=["src/**"], deny_write=["**/*.lock"])
        >>> cfg.allow_read
        ['**']
        >>> cfg.deny_write
        ['**/*.lock']
    """

    model_config = ConfigDict(extra="forbid")

    allow_write: list[str] = Field(default_factory=list)
    """Globs that the sandbox is *allowed* to write under. Empty list = no
    writes (the safe default; opt in explicitly)."""

    deny_write: list[str] = Field(default_factory=list)
    """Globs that override ``allow_write`` to forbid writes."""

    allow_read: list[str] = Field(default_factory=lambda: ["**"])
    """Globs that the sandbox is *allowed* to read. Default ``["**"]``
    permits all reads inside the sandbox root."""

    deny_read: list[str] = Field(default_factory=list)
    """Globs that override ``allow_read`` to forbid reads (e.g. secrets)."""


class SandboxNetworkConfig(BaseModel):
    """Egress policy for a sandbox.

    Concrete implementations enforce reachability per their threat model
    (SB5). In-process backends may treat these as advisory; container-based
    backends typically route via iptables / a proxy / a seccomp filter.

    Example:
        >>> cfg = SandboxNetworkConfig(allowed_domains=["api.openai.com"])
        >>> cfg.allow_unix_sockets
        False
        >>> cfg.seccomp_enabled
        True
    """

    model_config = ConfigDict(extra="forbid")

    allowed_domains: list[str] = Field(default_factory=list)
    """Domains the sandbox may reach (supports ``*.example.com`` glob
    syntax). Empty list means no domain-level allow-list is configured;
    each backend decides whether that means "deny all" or "allow all" per
    its threat model — that choice belongs in the implementation, not the
    Protocol."""

    allow_unix_sockets: bool = False
    """Whether unix-domain sockets (e.g. ``/var/run/docker.sock``) are
    reachable from inside the sandbox."""

    http_proxy_port: int | None = None
    """If set, all HTTP/HTTPS egress is routed through ``localhost:<port>``."""

    socks_proxy_port: int | None = None
    """If set, all SOCKS egress is routed through ``localhost:<port>``."""

    seccomp_enabled: bool = True
    """Linux-only: whether a default seccomp profile is applied. Ignored on
    other platforms / by backends that don't use seccomp."""


class SandboxConfig(BaseModel):
    """Composite declarative policy for a sandbox (filesystem + network).

    The Protocol exposes this as the read-only ``config`` attribute.
    Implementations may extend with backend-specific fields via subclassing,
    but the two nested shapes are the universal portable surface.

    Example:
        >>> SandboxConfig().fs.allow_read
        ['**']
        >>> SandboxConfig(net=SandboxNetworkConfig(seccomp_enabled=False)).net.seccomp_enabled
        False
    """

    model_config = ConfigDict(extra="forbid")

    fs: SandboxFilesystemConfig = Field(default_factory=SandboxFilesystemConfig)
    net: SandboxNetworkConfig = Field(default_factory=SandboxNetworkConfig)


# ---------------------------------------------------------------------------
# Value types returned by Sandbox methods
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FileStat:
    """Metadata for a single filesystem entry.

    Returned by :meth:`Sandbox.stat`. ``mtime`` is a timezone-aware
    :class:`datetime` — backends that source ``mtime`` from a unix timestamp
    must attach a tzinfo (UTC by convention) rather than returning a naive
    value.

    Example:
        >>> from datetime import datetime, timezone
        >>> FileStat(
        ...     path="/workspace/README.md",
        ...     size=42,
        ...     mtime=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ...     is_dir=False,
        ... ).is_dir
        False
    """

    path: str
    size: int
    mtime: datetime
    is_dir: bool


@dataclass(frozen=True, slots=True)
class FileEntry:
    """A single entry in a directory listing.

    Returned by :meth:`Sandbox.readdir`. Names are *not* full paths — they
    are the leaf entry name relative to the directory being listed. Callers
    that need an absolute path join with the parent themselves.

    Example:
        >>> FileEntry(name="README.md", is_dir=False).name
        'README.md'
    """

    name: str
    is_dir: bool


@dataclass(frozen=True, slots=True)
class ExecResult:
    """Result of a single :meth:`Sandbox.exec` invocation.

    ``timed_out`` is True iff the call hit its deadline; in that case
    ``exit_code`` is implementation-defined (typically ``-1`` or ``124``).
    Implementations that prefer to raise :class:`SandboxTimeoutError`
    instead of returning ``timed_out=True`` are also conformant — the
    framework's outer ``asyncio.wait_for`` enforces the deadline either way
    (SB3).

    Example:
        >>> ExecResult(exit_code=0, stdout="hi\\n", stderr="").exit_code
        0
        >>> ExecResult(exit_code=-1, stdout="", stderr="", timed_out=True).timed_out
        True
    """

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Sandbox Protocol — the 9-method universal contract (SB1)
# ---------------------------------------------------------------------------


@runtime_checkable
class Sandbox(Protocol):
    """Universal sandbox contract (SB1).

    Built-in implementations: ``InProcessSandbox`` (dev / tests),
    ``ModalSandbox``, ``FlySandbox`` — all in Wave 3. Adding other backends
    (E2B, Daytona, Docker, ...) is user-implementation against this
    Protocol (SB6); no inheritance required, structural typing is enough.

    **Cancellation contract** (Flue's lesson, SB2 / SB3):

    ``timeout`` is the *primary* cancellation mechanism. Every method takes
    a ``timeout: float | None`` keyword. Implementations *may* additionally
    honor :class:`asyncio.CancelledError`, but doing so is best-effort and
    not required by the Protocol.

    The harness is responsible for enforcing the deadline from the outside,
    typically via :func:`asyncio.wait_for`, so even backends whose
    underlying SDKs ignore cancellation abort calls at the configured time.
    Implementations that prefer to raise
    :class:`~agent_harness.core.errors.SandboxTimeoutError` directly on
    deadline are also conformant.

    **Path semantics**: ``path`` arguments are interpreted relative to
    :attr:`root` unless absolute. Implementations MUST reject paths that
    climb above ``root`` (SB7). The Protocol itself documents the contract;
    enforcement is per implementation.

    **Threat model**: every concrete implementation MUST document its
    threat model in the class docstring (SB8). The Protocol carries no
    such guarantee — it is plumbing (SB9).

    Example:
        >>> class _StubSandbox:
        ...     name = "stub"
        ...     root = "/workspace"
        ...     config = SandboxConfig()
        ...
        ...     async def read_file(self, path, *, timeout=None):
        ...         return ""
        ...
        ...     async def write_file(self, path, content, *, timeout=None):
        ...         return None
        ...
        ...     async def stat(self, path, *, timeout=None):
        ...         from datetime import datetime, timezone
        ...
        ...         return FileStat(
        ...             path=path,
        ...             size=0,
        ...             mtime=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ...             is_dir=False,
        ...         )
        ...
        ...     async def readdir(self, path, *, timeout=None):
        ...         return []
        ...
        ...     async def exists(self, path, *, timeout=None):
        ...         return False
        ...
        ...     async def mkdir(self, path, *, parents=False, timeout=None):
        ...         return None
        ...
        ...     async def rm(self, path, *, recursive=False, timeout=None):
        ...         return None
        ...
        ...     async def read_file_bytes(self, path, *, timeout=None):
        ...         return b""
        ...
        ...     async def exec(self, cmd, *, cwd=None, env=None, stdin=None, timeout=None):
        ...         return ExecResult(exit_code=0, stdout="", stderr="")
        >>> isinstance(_StubSandbox(), Sandbox)
        True
    """

    name: str
    """Human-readable backend label (e.g. ``"in-process"``, ``"modal"``)."""

    root: str
    """Absolute path the sandbox treats as its filesystem root (e.g.
    ``"/workspace"``). All path arguments are scoped against this (SB7)."""

    config: SandboxConfig
    """Declarative filesystem + network policy (SB5)."""

    # --- File ops ----------------------------------------------------------

    async def read_file(self, path: str, *, timeout: float | None = None) -> str:
        """Read the file at ``path`` as UTF-8 text.

        Raises :class:`~agent_harness.core.errors.SandboxError` (or a
        subclass) on failure; :class:`SandboxTimeoutError` if the deadline
        expires.
        """
        ...

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        timeout: float | None = None,
    ) -> None:
        """Write ``content`` to ``path`` as UTF-8 text, replacing any
        existing file."""
        ...

    async def stat(self, path: str, *, timeout: float | None = None) -> FileStat:
        """Return metadata for the entry at ``path``."""
        ...

    async def readdir(
        self,
        path: str,
        *,
        timeout: float | None = None,
    ) -> list[FileEntry]:
        """List the entries of the directory at ``path`` (non-recursive)."""
        ...

    async def exists(self, path: str, *, timeout: float | None = None) -> bool:
        """Return True iff something exists at ``path``."""
        ...

    async def mkdir(
        self,
        path: str,
        *,
        parents: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Create the directory at ``path``. If ``parents=True``, create
        intermediate directories as needed (``mkdir -p``)."""
        ...

    async def rm(
        self,
        path: str,
        *,
        recursive: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Remove the entry at ``path``. If ``recursive=True``, remove
        directories and their contents."""
        ...

    async def read_file_bytes(
        self,
        path: str,
        *,
        timeout: float | None = None,
    ) -> bytes:
        """Read the file at ``path`` as raw bytes (no decoding)."""
        ...

    # --- Exec --------------------------------------------------------------

    async def exec(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run ``cmd`` (argv list — no shell interpretation by default) and
        return its :class:`ExecResult`.

        - ``cwd``: working directory (relative to :attr:`root` unless
          absolute); defaults to :attr:`root`.
        - ``env``: environment variables; ``None`` means "inherit per
          implementation policy."
        - ``stdin``: optional stdin payload (UTF-8).
        - ``timeout``: see the cancellation contract on the class
          docstring.
        """
        ...
