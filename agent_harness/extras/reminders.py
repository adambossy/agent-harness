"""System reminders: backend-enqueued state flushed into the next user message.

Hosts enqueue reminders keyed by session; the run loop drains them when the
user prompt is appended and attaches each as a ``<system-reminder>`` text
block. Only the most recent reminder of a kind reflects current state, so
``override=True`` (the default) replaces queued same-kind reminders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Reminder:
    kind: str
    content: str


def wrap_system_reminder(reminder: Reminder) -> str:
    return f'<system-reminder kind="{reminder.kind}">\n{reminder.content}\n</system-reminder>'


@runtime_checkable
class ReminderQueue(Protocol):
    async def enqueue(
        self, session_id: str, kind: str, content: str, *, override: bool = True
    ) -> None: ...

    async def drain(self, session_id: str) -> list[Reminder]: ...


class InMemoryReminderQueue:
    def __init__(self) -> None:
        self._queues: dict[str, list[Reminder]] = {}

    async def enqueue(
        self, session_id: str, kind: str, content: str, *, override: bool = True
    ) -> None:
        queue = self._queues.setdefault(session_id, [])
        if override:
            queue[:] = [r for r in queue if r.kind != kind]
        queue.append(Reminder(kind=kind, content=content))

    async def drain(self, session_id: str) -> list[Reminder]:
        return self._queues.pop(session_id, [])
