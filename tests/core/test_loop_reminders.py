"""The run loop flushes queued reminders into the outgoing user message.

No ``scripted_agent_factory`` fixture ships with the repo, so we mirror the
neighboring loop tests (:mod:`tests.unit.test_loop_prepareturn`): a one-turn
``FakeModel`` script driven against an :class:`InMemorySession`. The factory
below returns ``(agent, session)`` exactly as the plan's fixture would.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from agent_harness.core.agent import Agent
from agent_harness.extras.reminders import InMemoryReminderQueue, ReminderQueue
from agent_harness.sessions.inmemory import InMemorySession
from tests.fakes import FakeTurn, make_model


@pytest.fixture
def scripted_agent_factory() -> Callable[..., tuple[Agent[Any, Any], InMemorySession]]:
    def _make(*, reminders: ReminderQueue | None = None) -> tuple[Agent[Any, Any], InMemorySession]:
        session = InMemorySession(session_id="s1")
        agent: Agent[Any, Any] = Agent(
            name="d",
            model=make_model(FakeTurn(text="ok")),
            toolsets=[],
            session=session,
            reminders=reminders,
        )
        return agent, session

    return _make


async def test_user_message_carries_drained_reminders(
    scripted_agent_factory: Callable[..., tuple[Agent[Any, Any], InMemorySession]],
) -> None:
    """Attach a queue with one reminder, run once, assert the recorded message."""
    reminders = InMemoryReminderQueue()
    agent, session = scripted_agent_factory(reminders=reminders)
    await reminders.enqueue(session.session_id, "onboarding", "connect plaid")

    await agent.run("hello")

    msgs = await session.get_messages()
    user = next(m for m in msgs if m.role == "user")
    texts = [b.text for b in user.content if getattr(b, "text", None)]
    assert texts[0] == "hello"
    assert texts[1] == ('<system-reminder kind="onboarding">\nconnect plaid\n</system-reminder>')
    assert user.metadata["system_reminder_kinds"] == ["onboarding"]
    assert await reminders.drain(session.session_id) == []


async def test_no_queue_means_untouched_message(
    scripted_agent_factory: Callable[..., tuple[Agent[Any, Any], InMemorySession]],
) -> None:
    agent, session = scripted_agent_factory(reminders=None)
    await agent.run("hello")
    user = next(m for m in await session.get_messages() if m.role == "user")
    assert len(user.content) == 1
