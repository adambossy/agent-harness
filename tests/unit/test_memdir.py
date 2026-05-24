"""Unit tests for :mod:`agent_harness.long_term.memdir`."""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_harness.core.memory import LongTermMemory, Memory
from agent_harness.long_term.memdir import (
    INDEX_FILE,
    LOGS_DIR,
    MAX_ENTRY_LINES,
    MEMORY_FILE,
    TOPICS_DIR,
    MemdirLongTermMemory,
    default_root,
)


@pytest.fixture
def tmp_root() -> Iterator[Path]:
    """A fresh memdir root that doesn't pollute the user's home."""
    root = Path(tempfile.mkdtemp(prefix="memdir-"))
    try:
        yield root
    finally:
        # Best-effort cleanup; tests shouldn't care if files linger.
        for p in sorted(root.rglob("*"), reverse=True):
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                p.rmdir()
        if root.exists():
            root.rmdir()


# --- layout + construction ---------------------------------------------------


def test_constructor_creates_layout(tmp_root: Path) -> None:
    MemdirLongTermMemory(root=tmp_root)
    assert (tmp_root / MEMORY_FILE).is_file()
    assert (tmp_root / TOPICS_DIR).is_dir()
    assert (tmp_root / LOGS_DIR).is_dir()
    assert (tmp_root / INDEX_FILE).is_file()
    assert json.loads((tmp_root / INDEX_FILE).read_text(encoding="utf-8")) == []


def test_protocol_satisfaction(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    assert isinstance(ltm, LongTermMemory)


def test_default_root_uses_provided_home(tmp_root: Path) -> None:
    home = tmp_root / "home"
    r = default_root(cwd=tmp_root, home=home)
    parts = r.parts
    assert "projects" in parts
    assert r.name == "memory"
    # 12-hex segment lives one level above 'memory'
    digest = r.parent.name
    assert len(digest) == 12
    assert all(c in "0123456789abcdef" for c in digest)


def test_default_root_outside_git_repo(tmp_root: Path) -> None:
    """When ``cwd`` is not in a git repo, fall back to cwd; no crash."""
    home = tmp_root / "home"
    r = default_root(cwd=tmp_root, home=home)
    assert r.is_absolute()


# --- remember + recall happy path -------------------------------------------


async def test_remember_returns_sha256_id(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    mid = await ltm.remember("user prefers tabs")
    assert isinstance(mid, str)
    assert len(mid) == 64
    assert all(c in "0123456789abcdef" for c in mid)


async def test_remember_default_writes_to_entry_file(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("prefers tabs over spaces")
    text = (tmp_root / MEMORY_FILE).read_text(encoding="utf-8")
    assert "prefers tabs over spaces" in text


async def test_remember_topic_writes_topic_file(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("uses circles", metadata={"topic": "Architecture"})
    expected = tmp_root / TOPICS_DIR / "architecture.md"
    assert expected.is_file()
    assert "uses circles" in expected.read_text(encoding="utf-8")


async def test_remember_log_writes_daily_log(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("ran the smoke test", metadata={"log": True})
    daily = list((tmp_root / LOGS_DIR).rglob("*.md"))
    assert len(daily) == 1
    assert "ran the smoke test" in daily[0].read_text(encoding="utf-8")


async def test_recall_finds_keyword_in_entry(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("the team prefers tabs")
    hits = await ltm.recall("tabs")
    assert len(hits) == 1
    assert "tabs" in hits[0].content
    assert hits[0].score is not None and hits[0].score >= 1.0


async def test_recall_ranks_by_match_count(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("rare keyword", metadata={"topic": "one"})
    await ltm.remember("rare keyword rare keyword rare keyword", metadata={"topic": "two"})
    hits = await ltm.recall("rare keyword", limit=2)
    assert len(hits) == 2
    # The "two" topic has more matches so it ranks first.
    assert hits[0].score is not None
    assert hits[1].score is not None
    assert hits[0].score > hits[1].score


async def test_recall_respects_limit(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    for i in range(5):
        await ltm.remember(f"hello world {i}", metadata={"topic": f"t{i}"})
    hits = await ltm.recall("hello", limit=3)
    assert len(hits) == 3


async def test_recall_excludes_logs_by_default(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("LOG entry kettle", metadata={"log": True})
    await ltm.remember("topic entry kettle", metadata={"topic": "kitchen"})
    hits = await ltm.recall("kettle")
    assert len(hits) == 1
    assert "topic entry" in hits[0].content


async def test_recall_can_include_logs_via_filter(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("LOG kettle whistle", metadata={"log": True})
    hits = await ltm.recall("whistle", filter={"log": True})
    assert len(hits) == 1


async def test_recall_filters_metadata_subset(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("alpha cluster", metadata={"topic": "infra"})
    await ltm.remember("beta cluster", metadata={"topic": "infra"})
    await ltm.remember("alpha cluster", metadata={"topic": "people"})
    hits = await ltm.recall("alpha", filter={"topic": "infra"})
    assert len(hits) == 1


async def test_recall_empty_query_returns_empty(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("something")
    assert await ltm.recall("") == []
    assert await ltm.recall("   ") == []


async def test_recall_limit_zero_returns_empty(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("anything")
    assert await ltm.recall("anything", limit=0) == []


async def test_recall_negative_limit_raises(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    with pytest.raises(ValueError, match="non-negative"):
        await ltm.recall("anything", limit=-1)


async def test_recall_no_matches_returns_empty(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("apple banana cherry")
    assert await ltm.recall("zucchini") == []


# --- list_memories ----------------------------------------------------------


async def test_list_memories_returns_all(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("one")
    await ltm.remember("two", metadata={"topic": "t"})
    rows = await ltm.list_memories()
    assert len(rows) == 2
    assert {m.content[:3] for m in rows} == {"one", "two"}


async def test_list_memories_newest_first(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    a = await ltm.remember("first")
    # sleep tiny to ensure distinct timestamps
    await asyncio.sleep(0.005)
    b = await ltm.remember("second")
    rows = await ltm.list_memories()
    assert rows[0].id == b
    assert rows[1].id == a


async def test_list_memories_respects_limit(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    for _ in range(4):
        await ltm.remember("x")
    rows = await ltm.list_memories(limit=2)
    assert len(rows) == 2


async def test_list_memories_zero_returns_empty(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("x")
    assert await ltm.list_memories(limit=0) == []


async def test_list_memories_negative_limit_raises(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    with pytest.raises(ValueError, match="non-negative"):
        await ltm.list_memories(limit=-1)


# --- forget -----------------------------------------------------------------


async def test_forget_removes_index_row_and_block(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    mid = await ltm.remember("ephemeral note about kittens")
    assert await ltm.recall("kittens")  # baseline
    await ltm.forget(mid)
    assert await ltm.recall("kittens") == []
    assert await ltm.list_memories() == []


async def test_forget_unknown_id_is_noop(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("kept")
    await ltm.forget("nonexistent-id")
    assert len(await ltm.list_memories()) == 1


# --- entry-file caps + concurrency ------------------------------------------


async def test_entry_file_is_trimmed_to_line_cap(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    # Each remember writes ~3 lines. Push well past the line cap.
    for i in range(MAX_ENTRY_LINES):
        await ltm.remember(f"entry-{i}")
    text = (tmp_root / MEMORY_FILE).read_text(encoding="utf-8")
    lines = text.splitlines()
    assert len(lines) <= MAX_ENTRY_LINES


async def test_concurrent_writes_all_land(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)

    async def writer(prefix: str, n: int) -> list[str]:
        ids: list[str] = []
        for i in range(n):
            ids.append(await ltm.remember(f"{prefix}-{i}", metadata={"topic": prefix}))
        return ids

    a, b = await asyncio.gather(writer("alpha", 5), writer("beta", 5))
    rows = await ltm.list_memories(limit=100)
    ids = {m.id for m in rows}
    for mid in [*a, *b]:
        assert mid in ids


# --- read_entry helper ------------------------------------------------------


async def test_read_entry_returns_current_memory_md(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    assert await ltm.read_entry() == ""
    await ltm.remember("seed entry")
    body = await ltm.read_entry()
    assert "seed entry" in body


# --- Memory record sanity ---------------------------------------------------


async def test_recall_results_are_memory_dataclasses(tmp_root: Path) -> None:
    ltm = MemdirLongTermMemory(root=tmp_root)
    await ltm.remember("a single note about gardening")
    hits = await ltm.recall("gardening")
    assert hits
    assert isinstance(hits[0], Memory)
    assert isinstance(hits[0].metadata, dict)
    assert hits[0].id and hits[0].content
