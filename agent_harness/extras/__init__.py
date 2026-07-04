"""Optional, opt-in extras.

These are NOT part of the inviolate core: they require additional setup
(git binary for `CheckpointTracker`) or solve specific UX problems
(`Mentions`, `IgnoreSet`). Import the ones you want; the loop / agent
work without any of them.
"""

from .checkpoints import Checkpoint, CheckpointTracker
from .ignoreset import IgnoreSet
from .mentions import Mention, MentionResolver, ResolvedMention
from .reminders import (
    InMemoryReminderQueue,
    Reminder,
    ReminderQueue,
    wrap_system_reminder,
)

__all__ = [
    "Checkpoint",
    "CheckpointTracker",
    "IgnoreSet",
    "InMemoryReminderQueue",
    "Mention",
    "MentionResolver",
    "Reminder",
    "ReminderQueue",
    "ResolvedMention",
    "wrap_system_reminder",
]
