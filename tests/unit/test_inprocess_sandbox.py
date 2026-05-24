"""Unit tests for :class:`agent_harness.sandboxes.inprocess.InProcessSandbox`.

Covers the 9-method Protocol surface, the path-jail invariant (SB7), and
the timeout-primary cancellation contract (SB2 / SB3).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_harness.core.errors import SandboxError, SandboxTimeoutError
from agent_harness.core.sandbox import (
    ExecResult,
    FileEntry,
    FileStat,
    Sandbox,
    SandboxConfig,
    SandboxFilesystemConfig,
)
from agent_harness.sandboxes.inprocess import InProcessSandbox

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_satisfies_sandbox_protocol(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    assert isinstance(sb, Sandbox)
    assert sb.name == "in-process"
    assert sb.root == str(tmp_path.resolve())


def test_create_root_creates_missing_directory(tmp_path: Path) -> None:
    target = tmp_path / "new-root"
    assert not target.exists()
    sb = InProcessSandbox(root=target)
    assert Path(sb.root).is_dir()


def test_existing_required_root_must_exist(tmp_path: Path) -> None:
    with pytest.raises(SandboxError):
        InProcessSandbox(root=tmp_path / "missing", create_root=False)


def test_custom_config_is_propagated(tmp_path: Path) -> None:
    cfg = SandboxConfig(fs=SandboxFilesystemConfig(allow_write=["src/**"]))
    sb = InProcessSandbox(root=tmp_path, config=cfg)
    assert sb.config.fs.allow_write == ["src/**"]


# ---------------------------------------------------------------------------
# Path jail (SB7)
# ---------------------------------------------------------------------------


async def test_path_jail_rejects_climb_above_root(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path / "sub")
    with pytest.raises(SandboxError, match="escapes sandbox root"):
        await sb.read_file("../escape.txt")


async def test_path_jail_rejects_absolute_path_outside(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path / "sub")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    with pytest.raises(SandboxError, match="escapes sandbox root"):
        await sb.read_file(str(outside))


async def test_path_jail_accepts_absolute_path_inside(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    inside = tmp_path / "ok.txt"
    inside.write_text("hi")
    text = await sb.read_file(str(inside))
    assert text == "hi"


# ---------------------------------------------------------------------------
# File ops happy paths
# ---------------------------------------------------------------------------


async def test_write_then_read_round_trip(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    await sb.write_file("a.txt", "hello")
    assert await sb.read_file("a.txt") == "hello"


async def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    await sb.write_file("deep/nested/file.txt", "ok")
    assert (tmp_path / "deep" / "nested" / "file.txt").read_text() == "ok"


async def test_read_file_bytes_round_trip(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    (tmp_path / "bin").write_bytes(b"\x00\x01\x02")
    data = await sb.read_file_bytes("bin")
    assert data == b"\x00\x01\x02"


async def test_stat_returns_metadata(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    (tmp_path / "f").write_text("abc")
    stat = await sb.stat("f")
    assert isinstance(stat, FileStat)
    assert stat.size == 3
    assert stat.is_dir is False
    assert stat.mtime.tzinfo is not None


async def test_stat_marks_directories(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    (tmp_path / "d").mkdir()
    stat = await sb.stat("d")
    assert stat.is_dir is True


async def test_readdir_returns_sorted_entries(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "sub").mkdir()
    entries = await sb.readdir(".")
    assert entries == [
        FileEntry(name="a.txt", is_dir=False),
        FileEntry(name="b.txt", is_dir=False),
        FileEntry(name="sub", is_dir=True),
    ]


async def test_readdir_rejects_non_directory(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    (tmp_path / "file").write_text("")
    with pytest.raises(SandboxError, match="not a directory"):
        await sb.readdir("file")


async def test_readdir_rejects_missing_directory(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    with pytest.raises(SandboxError, match="not found"):
        await sb.readdir("missing")


async def test_exists_true_and_false(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    (tmp_path / "yes").write_text("")
    assert await sb.exists("yes") is True
    assert await sb.exists("no") is False


async def test_mkdir_creates_directory(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    await sb.mkdir("newdir")
    assert (tmp_path / "newdir").is_dir()


async def test_mkdir_parents_true_creates_intermediate(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    await sb.mkdir("a/b/c", parents=True)
    assert (tmp_path / "a" / "b" / "c").is_dir()


async def test_mkdir_existing_without_parents_raises(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    (tmp_path / "dup").mkdir()
    with pytest.raises(SandboxError):
        await sb.mkdir("dup")


async def test_rm_file(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    (tmp_path / "x").write_text("")
    await sb.rm("x")
    assert not (tmp_path / "x").exists()


async def test_rm_directory_recursive(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f").write_text("")
    await sb.rm("d", recursive=True)
    assert not (tmp_path / "d").exists()


async def test_rm_directory_non_recursive_fails_when_not_empty(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f").write_text("")
    with pytest.raises(SandboxError):
        await sb.rm("d")


async def test_rm_root_is_rejected(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    with pytest.raises(SandboxError, match="cannot remove sandbox root"):
        await sb.rm(".")


async def test_read_missing_file_raises_sandbox_error(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    with pytest.raises(SandboxError, match="file not found"):
        await sb.read_file("absent.txt")


# ---------------------------------------------------------------------------
# Exec
# ---------------------------------------------------------------------------


async def test_exec_runs_command(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    result = await sb.exec(["echo", "hello"])
    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    assert result.timed_out is False


async def test_exec_uses_root_as_default_cwd(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    result = await sb.exec(["pwd"])
    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


async def test_exec_honors_cwd(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    sb = InProcessSandbox(root=tmp_path)
    result = await sb.exec(["pwd"], cwd="sub")
    assert Path(result.stdout.strip()).resolve() == (tmp_path / "sub").resolve()


async def test_exec_honors_env(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    result = await sb.exec(
        ["sh", "-c", "echo $AGENT_HARNESS_TEST"],
        env={"AGENT_HARNESS_TEST": "marker"},
    )
    assert result.stdout.strip() == "marker"


async def test_exec_honors_stdin(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    result = await sb.exec(["cat"], stdin="from-test\n")
    assert result.stdout == "from-test\n"


async def test_exec_propagates_nonzero_exit_code(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    result = await sb.exec(["sh", "-c", "exit 7"])
    assert result.exit_code == 7


async def test_exec_empty_cmd_rejected(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    with pytest.raises(SandboxError):
        await sb.exec([])


async def test_exec_rejects_cwd_outside_root(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path / "in")
    with pytest.raises(SandboxError, match="escapes sandbox root"):
        await sb.exec(["pwd"], cwd="..")


# ---------------------------------------------------------------------------
# Timeout contract (SB2 / SB3)
# ---------------------------------------------------------------------------


async def test_exec_timeout_raises_sandbox_timeout(tmp_path: Path) -> None:
    sb = InProcessSandbox(root=tmp_path)
    with pytest.raises(SandboxTimeoutError):
        await sb.exec(["sleep", "5"], timeout=0.1)


async def test_read_file_timeout_raises_sandbox_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sb = InProcessSandbox(root=tmp_path)
    (tmp_path / "f").write_text("x")

    async def _slow(*_: object, **__: object) -> object:
        await asyncio.sleep(5)
        return ""

    monkeypatch.setattr("agent_harness.sandboxes.inprocess.asyncio.to_thread", _slow)
    with pytest.raises(SandboxTimeoutError):
        await sb.read_file("f", timeout=0.05)
