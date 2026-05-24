"""SQLite :class:`Session` backend (SS3).

The default *persistent* short-term store. Uses stdlib ``sqlite3`` (zero
extra deps); blocking calls run in :func:`asyncio.to_thread` so the event
loop is never blocked. A single connection plus a per-session
:class:`asyncio.Lock` serialises writes; SQLite handles cross-process
contention via its own file lock when multiple processes open the same DB.

Two tables share one database file — messages and run-state snapshots
live next to each other so a paused conversation's two halves stay atomic
(SS1, SS2):

* ``messages(rowid, session_id, ord, payload)`` — JSON dump per row.
* ``run_states(rowid, session_id, ord, payload)`` — JSON dump per row.

Insertion order is captured by an explicit ``ord`` column (monotonic per
session) so reads are deterministic without relying on ``rowid``.

Compaction is *not* a Session concern (SS5); this backend just stores
what it is given.

Example:
    >>> import asyncio
    >>> from pathlib import Path
    >>> tmp = Path("/tmp/agent_harness_sqlite_demo.db")
    >>> if tmp.exists():
    ...     tmp.unlink()
    >>> sess = SqliteSession(session_id="s1", path=tmp)
    >>> asyncio.run(sess.get_messages())
    []
    >>> tmp.unlink()
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from agent_harness.core.memory import Session
from agent_harness.core.models import Message
from agent_harness.core.run_state import RunStateSnapshot

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    ord         INTEGER NOT NULL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_sess_ord ON messages (session_id, ord);

CREATE TABLE IF NOT EXISTS run_states (
    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    ord         INTEGER NOT NULL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS run_states_sess_ord ON run_states (session_id, ord);
"""


class SqliteSession(Session):
    """Disk-backed :class:`Session` implementation.

    ``path`` is the SQLite database file. Multiple :class:`SqliteSession`
    instances may point at the same file with different ``session_id``s —
    rows are partitioned by ``session_id``.

    Example:
        >>> import asyncio, tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     db = Path(tmp) / "s.db"
        ...     sess = SqliteSession(session_id="demo", path=db)
        ...     asyncio.run(sess.get_latest_run_state()) is None
        True
    """

    session_id: str

    def __init__(self, session_id: str, path: str | Path) -> None:
        self.session_id = session_id
        self._path = Path(path)
        self._lock = asyncio.Lock()
        # `check_same_thread=False`: we route every call through
        # ``asyncio.to_thread`` so the connection may be touched by any
        # worker thread. The per-session asyncio lock keeps the
        # interleaving safe.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- messages -----------------------------------------------------------

    async def get_messages(self) -> list[Message]:
        rows = await asyncio.to_thread(self._fetch_payloads, "messages")
        return [Message.model_validate_json(p) for p in rows]

    async def add_messages(self, msgs: list[Message]) -> None:
        if not msgs:
            return
        payloads = [m.model_dump_json() for m in msgs]
        async with self._lock:
            await asyncio.to_thread(self._append_payloads, "messages", payloads)

    # --- snapshots ----------------------------------------------------------

    async def get_run_states(self, *, limit: int | None = None) -> list[RunStateSnapshot]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
        rows = await asyncio.to_thread(self._fetch_payloads, "run_states", limit)
        return [RunStateSnapshot.from_json(p) for p in rows]

    async def add_run_state(self, snap: RunStateSnapshot) -> None:
        payload = snap.to_json()
        async with self._lock:
            await asyncio.to_thread(self._append_payloads, "run_states", [payload])

    async def get_latest_run_state(self) -> RunStateSnapshot | None:
        rows = await asyncio.to_thread(self._fetch_payloads, "run_states", 1)
        if not rows:
            return None
        return RunStateSnapshot.from_json(rows[-1])

    # --- clear --------------------------------------------------------------

    async def clear(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._clear_session)

    # --- close --------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying SQLite connection.

        Optional helper for tests / explicit shutdown; the connection is
        otherwise closed when the object is garbage-collected.
        """
        await asyncio.to_thread(self._conn.close)

    # --- blocking helpers (run inside asyncio.to_thread) --------------------

    def _fetch_payloads(self, table: str, limit: int | None = None) -> list[str]:
        """Return ``payload`` rows for ``self.session_id``, oldest-first.

        When ``limit`` is given, returns the most-recent ``limit`` rows
        still in oldest-first order (matches :class:`InMemorySession`).
        """
        if limit is None:
            cur = self._conn.execute(
                f"SELECT payload FROM {table} WHERE session_id = ? ORDER BY ord ASC",
                (self.session_id,),
            )
            return [row[0] for row in cur.fetchall()]
        cur = self._conn.execute(
            f"SELECT payload FROM {table} WHERE session_id = ? ORDER BY ord DESC LIMIT ?",
            (self.session_id, limit),
        )
        return [row[0] for row in reversed(cur.fetchall())]

    def _append_payloads(self, table: str, payloads: list[str]) -> None:
        """Append ``payloads`` to ``table`` for ``self.session_id``.

        Wraps the insert burst in a single transaction so partial appends
        cannot land on crash.
        """
        with self._conn:
            cur = self._conn.execute(
                f"SELECT COALESCE(MAX(ord), -1) FROM {table} WHERE session_id = ?",
                (self.session_id,),
            )
            next_ord = int(cur.fetchone()[0]) + 1
            self._conn.executemany(
                f"INSERT INTO {table} (session_id, ord, payload) VALUES (?, ?, ?)",
                [(self.session_id, next_ord + i, payload) for i, payload in enumerate(payloads)],
            )

    def _clear_session(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
            self._conn.execute("DELETE FROM run_states WHERE session_id = ?", (self.session_id,))
