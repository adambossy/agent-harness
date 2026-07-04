from __future__ import annotations

from agent_harness.extras.reminders import (
    InMemoryReminderQueue,
    Reminder,
    wrap_system_reminder,
)


async def test_override_replaces_same_kind() -> None:
    q = InMemoryReminderQueue()
    await q.enqueue("s1", "onboarding", "state v1")
    await q.enqueue("s1", "onboarding", "state v2")  # override default
    await q.enqueue("s1", "plaid_link", "linked", override=True)
    drained = await q.drain("s1")
    assert [(r.kind, r.content) for r in drained] == [
        ("onboarding", "state v2"),
        ("plaid_link", "linked"),
    ]
    assert await q.drain("s1") == []  # drained empty


async def test_no_override_appends() -> None:
    q = InMemoryReminderQueue()
    await q.enqueue("s1", "note", "a", override=False)
    await q.enqueue("s1", "note", "b", override=False)
    assert [r.content for r in await q.drain("s1")] == ["a", "b"]


def test_wrap_format() -> None:
    text = wrap_system_reminder(Reminder(kind="onboarding", content="hello"))
    assert text == '<system-reminder kind="onboarding">\nhello\n</system-reminder>'
