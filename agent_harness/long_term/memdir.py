"""File-backed :class:`LongTermMemory` — the v1 default (LT3, LT4, LT5).

``MemdirLongTermMemory`` mirrors Claude Code's ``memdir/`` layout:

* ``<root>/MEMORY.md`` — the always-loaded entry file (LT5). Capped at
  ~400 lines / 50 KB so prepending it to the system prompt stays cheap.
* ``<root>/topics/<topic>.md`` — topic files, loaded on demand by
  :meth:`MemdirLongTermMemory.recall` or surfaced to the model via the
  ``recall`` tool.
* ``<root>/logs/YYYY/MM/YYYY-MM-DD.md`` — append-only daily logs.

Root is canonicalized per-project (LT4): the resolved git root is hashed
to a 12-hex-digit prefix and the memory store lives at
``~/.agent_harness/projects/<sha256(git_root)[:12]>/memory/``. The same
project across worktrees shares one store; unrelated projects are kept
separate.

This backend is zero-dependency and stores plain markdown — operations
are blocking I/O dispatched to a worker thread via :func:`asyncio.to_thread`
so the async surface stays non-blocking.

Example:
    >>> import asyncio, tempfile
    >>> from pathlib import Path
    >>> root = Path(tempfile.mkdtemp())
    >>> ltm = MemdirLongTermMemory(root=root)
    >>> mid = asyncio.run(ltm.remember("user prefers tabs"))
    >>> isinstance(mid, str) and len(mid) == 64
    True
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_harness.core.memory import LongTermMemory, Memory

# --- caps + canonical paths --------------------------------------------------

MEMORY_FILE = "MEMORY.md"
TOPICS_DIR = "topics"
LOGS_DIR = "logs"
INDEX_FILE = ".index.json"

# Entry-file caps (LT5). 400 lines OR 50 KB, whichever bites first. The cap is
# enforced on ``remember`` writes that target MEMORY.md, not on reads — a
# pre-existing oversized file is left alone so users can't lose memory simply
# by upgrading the harness.
MAX_ENTRY_LINES = 400
MAX_ENTRY_BYTES = 50 * 1024

# Slug rules for topic file names. Keep them filesystem-friendly: lowercase
# alnum + hyphens, single-segment (no ``/`` traversal even if user supplies it).
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(topic: str) -> str:
    """Return a safe lowercase-alnum-with-hyphens filename stem."""
    cleaned = _SLUG_RE.sub("-", topic.strip().lower()).strip("-")
    return cleaned or "untitled"


def _now() -> datetime:
    return datetime.now(UTC)


def _git_root(cwd: Path | None = None) -> Path:
    """Return the absolute git toplevel for ``cwd``; fall back to ``cwd``."""
    start = Path(cwd) if cwd is not None else Path.cwd()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start),
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return start.resolve()
    if res.returncode == 0:
        out = res.stdout.strip()
        if out:
            return Path(out).resolve()
    return start.resolve()


def default_root(cwd: Path | None = None, *, home: Path | None = None) -> Path:
    """Compute the canonical per-project memory root (LT4).

    ``~/.agent_harness/projects/<sha256(git_root)[:12]>/memory/``

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as td:
        ...     r = default_root(cwd=Path(td), home=Path(td) / "home")
        ...     "projects" in r.parts and r.name == "memory"
        True
    """
    git_root = _git_root(cwd)
    digest = hashlib.sha256(str(git_root).encode("utf-8")).hexdigest()[:12]
    base = (Path(home) if home is not None else Path.home()).expanduser()
    return base / ".agent_harness" / "projects" / digest / "memory"


# --- index record ------------------------------------------------------------


@dataclass(slots=True)
class _IndexRecord:
    """One row in ``.index.json`` — bookkeeping for ``forget`` and ``list``."""

    id: str
    path: str  # POSIX-style path relative to root
    snippet: str
    created_at: str  # ISO-8601 UTC
    metadata: dict[str, Any]


# --- the backend -------------------------------------------------------------


class MemdirLongTermMemory(LongTermMemory):
    """File-backed long-term memory in the Claude Code ``memdir/`` layout.

    Stores written content in three places depending on ``metadata``:

    * ``metadata["topic"]`` set: appends to ``topics/<slug>.md``.
    * ``metadata["log"]`` truthy: appends to today's daily log under
      ``logs/YYYY/MM/YYYY-MM-DD.md``.
    * Otherwise: appends to ``MEMORY.md`` (entry file, capped).

    All writes are also recorded in a small JSON index so ``forget`` and
    ``list_memories`` work without re-scanning the tree.

    Example:
        >>> import asyncio, tempfile
        >>> from pathlib import Path
        >>> root = Path(tempfile.mkdtemp())
        >>> ltm = MemdirLongTermMemory(root=root)
        >>> mid = asyncio.run(ltm.remember("hello", metadata={"topic": "greetings"}))
        >>> hits = asyncio.run(ltm.recall("hello"))
        >>> bool(hits) and hits[0].content.startswith("hello")
        True
    """

    def __init__(
        self,
        *,
        root: Path | None = None,
        cwd: Path | None = None,
        home: Path | None = None,
    ) -> None:
        self.root: Path = (
            Path(root).expanduser().resolve()
            if root is not None
            else default_root(cwd=cwd, home=home)
        )
        self._lock = asyncio.Lock()
        self._ensure_layout()

    # --- layout / IO helpers (sync; called from to_thread) -----------------

    def _ensure_layout(self) -> None:
        """Create the memdir on first touch — idempotent."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / TOPICS_DIR).mkdir(exist_ok=True)
        (self.root / LOGS_DIR).mkdir(exist_ok=True)
        entry = self.root / MEMORY_FILE
        if not entry.exists():
            entry.write_text("", encoding="utf-8")
        idx = self.root / INDEX_FILE
        if not idx.exists():
            idx.write_text("[]", encoding="utf-8")

    def _read_index(self) -> list[_IndexRecord]:
        raw = (self.root / INDEX_FILE).read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        out: list[_IndexRecord] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            try:
                out.append(
                    _IndexRecord(
                        id=str(row["id"]),
                        path=str(row["path"]),
                        snippet=str(row.get("snippet", "")),
                        created_at=str(row["created_at"]),
                        metadata=dict(row.get("metadata") or {}),
                    )
                )
            except KeyError:
                continue
        return out

    def _write_index(self, rows: Iterable[_IndexRecord]) -> None:
        serialised = [
            {
                "id": r.id,
                "path": r.path,
                "snippet": r.snippet,
                "created_at": r.created_at,
                "metadata": r.metadata,
            }
            for r in rows
        ]
        (self.root / INDEX_FILE).write_text(
            json.dumps(serialised, indent=2),
            encoding="utf-8",
        )

    def _target_path(self, metadata: dict[str, Any]) -> Path:
        topic = metadata.get("topic")
        if topic:
            return self.root / TOPICS_DIR / f"{_slugify(str(topic))}.md"
        if metadata.get("log"):
            now = _now()
            day = now.strftime("%Y-%m-%d")
            return self.root / LOGS_DIR / f"{now.year:04d}" / f"{now.month:02d}" / f"{day}.md"
        return self.root / MEMORY_FILE

    def _append_block(
        self,
        path: Path,
        body: str,
        *,
        memory_id: str,
        ts: datetime,
        key: str | None,
    ) -> None:
        """Append one fenced block to a markdown file under root."""
        path.parent.mkdir(parents=True, exist_ok=True)
        header_bits = [f"id={memory_id[:12]}", f"ts={ts.isoformat()}"]
        if key:
            header_bits.append(f"key={key}")
        block = f"\n<!-- {' '.join(header_bits)} -->\n{body.rstrip()}\n"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        path.write_text(existing + block, encoding="utf-8")
        # Cap the entry file (LT5). Apply only to MEMORY.md to keep topic /
        # log files unconstrained.
        if path == (self.root / MEMORY_FILE):
            self._trim_entry_file(path)

    def _trim_entry_file(self, path: Path) -> None:
        """Trim MEMORY.md to the configured line + byte caps, oldest-first."""
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        # Drop oldest leading lines until both caps are satisfied.
        while len(lines) > MAX_ENTRY_LINES or (
            len("\n".join(lines).encode("utf-8")) > MAX_ENTRY_BYTES and lines
        ):
            lines.pop(0)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # --- LongTermMemory API ------------------------------------------------

    async def remember(
        self,
        content: str,
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist ``content`` and return a stable memory id (LT1).

        The id is ``sha256(content + iso-timestamp + key)``; deterministic
        for a given ``(content, ts, key)`` triple so concurrent retries
        don't double-write.
        """
        meta = dict(metadata or {})
        ts = _now()
        material = f"{content}␟{ts.isoformat()}␟{key or ''}"
        memory_id = hashlib.sha256(material.encode("utf-8")).hexdigest()

        async with self._lock:
            await asyncio.to_thread(self._remember_sync, content, memory_id, ts, key, meta)
        return memory_id

    def _remember_sync(
        self,
        content: str,
        memory_id: str,
        ts: datetime,
        key: str | None,
        metadata: dict[str, Any],
    ) -> None:
        path = self._target_path(metadata)
        self._append_block(path, content, memory_id=memory_id, ts=ts, key=key)
        rows = self._read_index()
        rel = path.relative_to(self.root).as_posix()
        snippet = content.strip().splitlines()[0][:120] if content.strip() else ""
        rows.append(
            _IndexRecord(
                id=memory_id,
                path=rel,
                snippet=snippet,
                created_at=ts.isoformat(),
                metadata={**metadata, **({"key": key} if key else {})},
            )
        )
        self._write_index(rows)

    async def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Memory]:
        """Keyword-rank recall across MEMORY.md and topic files (LT3).

        Splits ``query`` into lowercase tokens and counts case-insensitive
        token occurrences within each candidate region. Returns up to
        ``limit`` :class:`Memory` records, highest score first. ``filter``
        is matched as a subset on each record's metadata.
        """
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
        return await asyncio.to_thread(self._recall_sync, query, limit, filter or {})

    def _recall_sync(
        self,
        query: str,
        limit: int,
        filter_: dict[str, Any],
    ) -> list[Memory]:
        tokens = [t for t in re.split(r"\s+", query.lower()) if t]
        if not tokens:
            return []
        rows = self._read_index()
        scored: list[tuple[float, Memory]] = []
        for row in rows:
            # filter is a strict subset match on metadata
            if filter_ and not all(row.metadata.get(k) == v for k, v in filter_.items()):
                continue
            # Skip log-only rows from recall by default; logs are an audit
            # trail (LT3), not a recall target. They can be opted into via
            # filter={"log": True}.
            if row.metadata.get("log") and not filter_.get("log"):
                continue
            full_path = self.root / row.path
            if not full_path.exists():
                continue
            body = self._extract_block(full_path, row.id) or row.snippet
            haystack = body.lower()
            score = float(sum(haystack.count(t) for t in tokens))
            if score <= 0:
                continue
            created = _parse_iso(row.created_at)
            scored.append(
                (
                    score,
                    Memory(
                        id=row.id,
                        content=body,
                        created_at=created,
                        metadata=dict(row.metadata),
                        score=score,
                    ),
                )
            )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def _extract_block(self, path: Path, memory_id: str) -> str | None:
        """Return the body of the fenced block with ``id={memory_id[:12]}``."""
        marker = f"id={memory_id[:12]}"
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        body: list[str] = []
        capturing = False
        for line in lines:
            if line.startswith("<!--") and marker in line:
                capturing = True
                body = []
                continue
            if capturing:
                if line.startswith("<!--"):  # next block starts
                    break
                body.append(line)
        result = "\n".join(body).strip()
        return result or None

    async def forget(self, memory_id: str) -> None:
        """Drop the on-disk block and its index row (LT1)."""
        async with self._lock:
            await asyncio.to_thread(self._forget_sync, memory_id)

    def _forget_sync(self, memory_id: str) -> None:
        rows = self._read_index()
        kept: list[_IndexRecord] = []
        removed: _IndexRecord | None = None
        for row in rows:
            if row.id == memory_id and removed is None:
                removed = row
                continue
            kept.append(row)
        if removed is None:
            return
        self._write_index(kept)
        path = self.root / removed.path
        if not path.exists():
            return
        marker = f"id={memory_id[:12]}"
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        out: list[str] = []
        skipping = False
        for line in lines:
            if line.startswith("<!--") and marker in line:
                skipping = True
                continue
            if skipping:
                if line.startswith("<!--"):
                    skipping = False
                else:
                    continue
            out.append(line)
        path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")

    async def list_memories(self, *, limit: int = 100) -> list[Memory]:
        """Return up to ``limit`` index records as :class:`Memory` (LT1).

        Ordered newest-first by ``created_at`` so callers see the most
        recent additions without paging.
        """
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
        return await asyncio.to_thread(self._list_sync, limit)

    def _list_sync(self, limit: int) -> list[Memory]:
        rows = self._read_index()
        rows.sort(key=lambda r: r.created_at, reverse=True)
        out: list[Memory] = []
        for row in rows[:limit]:
            full_path = self.root / row.path
            content = (
                (self._extract_block(full_path, row.id) or row.snippet)
                if full_path.exists()
                else row.snippet
            )
            out.append(
                Memory(
                    id=row.id,
                    content=content,
                    created_at=_parse_iso(row.created_at),
                    metadata=dict(row.metadata),
                    score=None,
                )
            )
        return out

    # --- entry-file accessor used by the loop (LT5) ------------------------

    async def read_entry(self) -> str:
        """Return the current contents of ``MEMORY.md``.

        The loop is expected to prepend this to the system prompt at the
        start of every turn (LT5). The file is capped on write so callers
        do not need to truncate again.
        """
        return await asyncio.to_thread((self.root / MEMORY_FILE).read_text, encoding="utf-8")


def _parse_iso(value: str) -> datetime:
    """Parse a stored ISO-8601 timestamp; fall back to ``epoch`` on failure."""
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromtimestamp(0, tz=UTC)
