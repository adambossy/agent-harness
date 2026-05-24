"""Unit tests for :class:`agent_harness.sandboxes.fly.FlySandbox`.

``httpx`` is an optional dependency. The tests inject a stub
``http_client=`` so they do not require the SDK at runtime and do not
touch the network. Two paths get explicit coverage:

1. **httpx missing → ``NotSupportedError``** at construction.
2. **9-method surface against the stub HTTP client**, plus the timeout
   contract (SB2 / SB3).
"""

from __future__ import annotations

import asyncio
import base64
import sys
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
from agent_harness.sandboxes.fly import FlySandbox

# ---------------------------------------------------------------------------
# Stub HTTP client — records calls and returns scripted responses.
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, *, status_code: int = 200, body: object = None) -> None:
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = str(body) if body is not None else ""

    def json(self) -> object:
        return self._body


class _StubHttpClient:
    """Records every request; serves scripted responses keyed by
    ``(METHOD, PATH-PREFIX)``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.closed = False
        # Default machine-create response.
        self.responses: dict[tuple[str, str], _StubResponse] = {
            ("POST", "/apps/test-app/machines"): _StubResponse(body={"id": "machine-abc"}),
        }
        # Default exec response (so simple `exec` returns ok).
        self.default_exec_response: _StubResponse = _StubResponse(
            body={"exit_code": 0, "stdout": "", "stderr": ""}
        )

    async def post(self, path: str, json: dict[str, Any] | None = None) -> _StubResponse:
        self.calls.append(("POST", path, json))
        # Exact match first.
        if ("POST", path) in self.responses:
            return self.responses[("POST", path)]
        # Anything that hits /exec → default exec response unless overridden.
        if path.endswith("/exec"):
            return self.default_exec_response
        return _StubResponse(body={})

    async def delete(self, path: str) -> _StubResponse:
        self.calls.append(("DELETE", path, None))
        return _StubResponse(body={})

    async def get(self, path: str) -> _StubResponse:
        self.calls.append(("GET", path, None))
        return _StubResponse(body={})

    async def aclose(self) -> None:
        self.closed = True


def _make_sandbox(
    http_client: _StubHttpClient,
    *,
    root: str = "/workspace",
    app: str = "test-app",
) -> FlySandbox:
    return FlySandbox(
        app=app,
        api_token="t",
        root=root,
        http_client=http_client,
    )


# ---------------------------------------------------------------------------
# SDK-missing path
# ---------------------------------------------------------------------------


def test_missing_httpx_raises_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``httpx`` installed, constructing :class:`FlySandbox`
    without an injected client raises :class:`NotSupportedError`."""

    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(NotSupportedError):
        FlySandbox(app="x", api_token="t")


# ---------------------------------------------------------------------------
# Construction + Protocol shape
# ---------------------------------------------------------------------------


def test_satisfies_sandbox_protocol() -> None:
    sb = _make_sandbox(_StubHttpClient())
    assert isinstance(sb, Sandbox)
    assert sb.name == "fly"
    assert sb.root == "/workspace"


# ---------------------------------------------------------------------------
# Lifecycle (machine create + close)
# ---------------------------------------------------------------------------


async def test_open_creates_machine_via_api() -> None:
    client = _StubHttpClient()
    sb = _make_sandbox(client)
    await sb.open()
    methods = [(m, p) for (m, p, _b) in client.calls]
    assert ("POST", "/apps/test-app/machines") in methods


async def test_close_deletes_machine() -> None:
    client = _StubHttpClient()
    sb = _make_sandbox(client)
    await sb.open()
    await sb.close()
    deletes = [c for c in client.calls if c[0] == "DELETE"]
    assert deletes, "close should DELETE the machine"
    assert "machine-abc" in deletes[0][1]


async def test_close_no_op_when_never_opened() -> None:
    client = _StubHttpClient()
    sb = _make_sandbox(client)
    await sb.close()
    assert all(c[0] != "DELETE" for c in client.calls)


# ---------------------------------------------------------------------------
# File ops (all run on top of /exec)
# ---------------------------------------------------------------------------


async def test_write_file_uses_tee_via_exec() -> None:
    client = _StubHttpClient()
    sb = _make_sandbox(client)
    await sb.write_file("a.txt", "hello")
    exec_calls = [c for c in client.calls if c[1].endswith("/exec")]
    assert exec_calls, "write_file must POST /exec"
    payload = exec_calls[0][2]
    assert payload is not None
    assert payload["stdin"] == "hello"
    assert "tee" in " ".join(payload["command"])


async def test_read_file_uses_cat() -> None:
    client = _StubHttpClient()
    client.default_exec_response = _StubResponse(
        body={"exit_code": 0, "stdout": "hello", "stderr": ""}
    )
    sb = _make_sandbox(client)
    text = await sb.read_file("a.txt")
    assert text == "hello"
    exec_call = next(c for c in client.calls if c[1].endswith("/exec"))
    assert exec_call[2] is not None
    assert exec_call[2]["command"][0] == "cat"


async def test_read_file_nonzero_exit_raises() -> None:
    client = _StubHttpClient()
    client.default_exec_response = _StubResponse(
        body={"exit_code": 1, "stdout": "", "stderr": "No such file"}
    )
    sb = _make_sandbox(client)
    with pytest.raises(SandboxError, match="read_file failed"):
        await sb.read_file("missing")


async def test_read_file_bytes_decodes_base64() -> None:
    client = _StubHttpClient()
    encoded = base64.b64encode(b"\x00\x01\xff").decode("ascii")
    client.default_exec_response = _StubResponse(
        body={"exit_code": 0, "stdout": encoded, "stderr": ""}
    )
    sb = _make_sandbox(client)
    data = await sb.read_file_bytes("bin")
    assert data == b"\x00\x01\xff"


async def test_stat_parses_stat_output() -> None:
    client = _StubHttpClient()
    client.default_exec_response = _StubResponse(
        body={
            "exit_code": 0,
            "stdout": "42 1700000000 regular file\n",
            "stderr": "",
        }
    )
    sb = _make_sandbox(client)
    stat = await sb.stat("f")
    assert isinstance(stat, FileStat)
    assert stat.size == 42
    assert stat.is_dir is False
    assert stat.mtime.tzinfo is not None


async def test_stat_directory() -> None:
    client = _StubHttpClient()
    client.default_exec_response = _StubResponse(
        body={"exit_code": 0, "stdout": "4096 1700000000 directory\n", "stderr": ""}
    )
    sb = _make_sandbox(client)
    stat = await sb.stat("d")
    assert stat.is_dir is True


async def test_readdir_parses_ls_output() -> None:
    client = _StubHttpClient()
    client.default_exec_response = _StubResponse(
        body={
            "exit_code": 0,
            "stdout": "a.txt\nb.txt\nsubdir/\n",
            "stderr": "",
        }
    )
    sb = _make_sandbox(client)
    entries = await sb.readdir(".")
    assert entries == [
        FileEntry(name="a.txt", is_dir=False),
        FileEntry(name="b.txt", is_dir=False),
        FileEntry(name="subdir", is_dir=True),
    ]


async def test_exists_true_when_test_exits_zero() -> None:
    client = _StubHttpClient()
    client.default_exec_response = _StubResponse(body={"exit_code": 0, "stdout": "", "stderr": ""})
    sb = _make_sandbox(client)
    assert await sb.exists("x") is True


async def test_exists_false_when_test_exits_nonzero() -> None:
    client = _StubHttpClient()
    client.default_exec_response = _StubResponse(body={"exit_code": 1, "stdout": "", "stderr": ""})
    sb = _make_sandbox(client)
    assert await sb.exists("x") is False


async def test_mkdir_with_parents_passes_minus_p() -> None:
    client = _StubHttpClient()
    sb = _make_sandbox(client)
    await sb.mkdir("a/b", parents=True)
    exec_call = next(c for c in client.calls if c[1].endswith("/exec"))
    assert exec_call[2] is not None
    assert exec_call[2]["command"][:2] == ["mkdir", "-p"]


async def test_rm_recursive_passes_minus_rf() -> None:
    client = _StubHttpClient()
    sb = _make_sandbox(client)
    await sb.rm("a", recursive=True)
    exec_call = next(c for c in client.calls if c[1].endswith("/exec"))
    assert exec_call[2] is not None
    assert exec_call[2]["command"][:2] == ["rm", "-rf"]


async def test_rm_root_rejected() -> None:
    client = _StubHttpClient()
    sb = _make_sandbox(client, root="/workspace")
    with pytest.raises(SandboxError, match="cannot remove sandbox root"):
        await sb.rm("/workspace")


# ---------------------------------------------------------------------------
# Exec
# ---------------------------------------------------------------------------


async def test_exec_passes_cmd_cwd_env_stdin() -> None:
    client = _StubHttpClient()
    sb = _make_sandbox(client)
    result = await sb.exec(
        ["echo", "hi"],
        cwd="/workspace/sub",
        env={"K": "V"},
        stdin="data",
    )
    assert isinstance(result, ExecResult)
    exec_call = next(c for c in client.calls if c[1].endswith("/exec"))
    payload = exec_call[2]
    assert payload is not None
    assert payload["command"] == ["echo", "hi"]
    assert payload["workdir"] == "/workspace/sub"
    assert payload["env"] == {"K": "V"}
    assert payload["stdin"] == "data"


async def test_exec_empty_cmd_rejected() -> None:
    client = _StubHttpClient()
    sb = _make_sandbox(client)
    with pytest.raises(SandboxError):
        await sb.exec([])


# ---------------------------------------------------------------------------
# API failure handling
# ---------------------------------------------------------------------------


async def test_non_2xx_machine_create_raises_sandbox_error() -> None:
    client = _StubHttpClient()
    client.responses[("POST", "/apps/test-app/machines")] = _StubResponse(
        status_code=500, body={"error": "boom"}
    )
    sb = _make_sandbox(client)
    with pytest.raises(SandboxError, match="returned 500"):
        await sb.open()


async def test_missing_machine_id_in_response_raises() -> None:
    client = _StubHttpClient()
    client.responses[("POST", "/apps/test-app/machines")] = _StubResponse(
        body={}  # no "id"
    )
    sb = _make_sandbox(client)
    with pytest.raises(SandboxError, match="no machine id"):
        await sb.open()


# ---------------------------------------------------------------------------
# Timeout contract (SB2 / SB3)
# ---------------------------------------------------------------------------


async def test_exec_timeout_raises_sandbox_timeout() -> None:
    class _SlowClient(_StubHttpClient):
        async def post(self, path: str, json: dict[str, Any] | None = None) -> _StubResponse:
            del path, json
            await asyncio.sleep(5)
            return _StubResponse(body={"exit_code": 0, "stdout": "", "stderr": ""})

    sb = _make_sandbox(_SlowClient())
    with pytest.raises(SandboxTimeoutError):
        await sb.exec(["sleep", "5"], timeout=0.05)
