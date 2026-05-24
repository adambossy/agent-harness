"""Unit tests for the LLM-driven :data:`HistoryProcessor` built-ins.

Covers :class:`MicroCompact`, :class:`ContextCollapse`,
:class:`SummarizeOldTurns`, :class:`ProviderSideCompaction` — cache
preservation (HP4), cost attribution (HP5), event emission (HP2),
provider-side delegation no-op (HP10), and validation error paths.

The ``# type: ignore[union-attr]`` comments on a few assertions are
intentional: indexing into a discriminated ``ContentBlock`` union returns
``TextBlock | ToolResultBlock | …`` and mypy can't narrow which variant
without an ``isinstance`` check; tests prefer concise asserts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from agent_harness.core.events import (
    CompactionEnd,
    CompactionStart,
    InMemoryEventBus,
    MessageEnd,
    ModelEnd,
    ModelStart,
)
from agent_harness.core.history import apply_processor
from agent_harness.core.history_llm import (
    COLLAPSE_MARKER_PREFIX,
    DEFAULT_SUMMARY_TEMPLATE,
    MICROCOMPACT_MARKER_PREFIX,
    SUMMARY_MARKER_PREFIX,
    ContextCollapse,
    MicroCompact,
    ProviderSideCompaction,
    SummarizeOldTurns,
)
from agent_harness.core.models import (
    Message,
    ModelCapabilities,
    ModelSettings,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    Usage,
)


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


# --- Tiny FakeModel for these tests ----------------------------------------


@dataclass(slots=True)
class _ScriptedSummaryModel:
    """Minimal model emitting one canned ``MessageEnd`` + ``ModelEnd``.

    Records every prompt body so tests can assert the LLM was/wasn't asked
    again. Doesn't satisfy :class:`Model` structurally (no provider attr),
    which is fine — our processors only call ``request`` + read
    ``capabilities``.
    """

    summary_text: str = "scripted summary"
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=7, output_tokens=11))
    capabilities: ModelCapabilities = field(
        default_factory=lambda: ModelCapabilities(context_window=200_000)
    )
    seen_prompts: list[str] = field(default_factory=list)
    call_count: int = 0

    async def request(
        self,
        messages: list[Message],
        tools: list[Any],
        settings: ModelSettings,
    ) -> AsyncIterator[Any]:
        del tools, settings
        self.call_count += 1
        # Stash the user-prompt body for cache assertions.
        self.seen_prompts.append(messages[-1].text)
        msg_id = f"msg_{self.call_count:03d}"
        final = Message(
            role="assistant",
            content=[TextBlock(text=self.summary_text)],
            timestamp=_ts(),
        )
        yield ModelStart(model_name="scripted")
        yield MessageEnd(message_id=msg_id, final=final, usage=self.usage)
        yield ModelEnd(message_id=msg_id, usage=self.usage)


def _ctx(model: Any, *, bus: InMemoryEventBus | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        agent=SimpleNamespace(model=model),
        usage=Usage(),
        event_bus=bus,
    )


# --- MicroCompact -----------------------------------------------------------


async def test_microcompact_summarizes_big_results_and_skips_small_ones() -> None:
    big = ToolResultBlock(tool_call_id="c1", content="x" * 200)
    small = ToolResultBlock(tool_call_id="c2", content="ok")
    msgs = [
        Message(role="tool", content=[big], timestamp=_ts()),
        Message(role="tool", content=[small], timestamp=_ts()),
    ]
    model = _ScriptedSummaryModel(summary_text="big thing summarized")
    ctx = _ctx(model)
    proc = MicroCompact(trigger_at_result_bytes=50)
    out = await apply_processor(proc, msgs, ctx=ctx)
    assert out[0].content[0].content.startswith(MICROCOMPACT_MARKER_PREFIX)  # type: ignore[union-attr]
    assert "big thing summarized" in out[0].content[0].content  # type: ignore[union-attr]
    # Small result untouched.
    assert out[1].content[0].content == "ok"  # type: ignore[union-attr]
    # Only one LLM call.
    assert model.call_count == 1


async def test_microcompact_attributes_usage_to_ctx() -> None:
    big = ToolResultBlock(tool_call_id="c1", content="x" * 200)
    msgs = [Message(role="tool", content=[big], timestamp=_ts())]
    model = _ScriptedSummaryModel(usage=Usage(input_tokens=42, output_tokens=9))
    ctx = _ctx(model)
    await apply_processor(MicroCompact(trigger_at_result_bytes=50), msgs, ctx=ctx)
    assert ctx.usage.input_tokens == 42
    assert ctx.usage.output_tokens == 9


async def test_microcompact_is_idempotent_byte_identical_across_turns() -> None:
    """HP4: re-running against the same prefix must produce identical bytes."""
    big = ToolResultBlock(tool_call_id="c1", content="x" * 300)
    msgs = [Message(role="tool", content=[big], timestamp=_ts())]
    model = _ScriptedSummaryModel(summary_text="canonical")
    ctx1 = _ctx(model)
    once = await apply_processor(MicroCompact(trigger_at_result_bytes=50), msgs, ctx=ctx1)
    # Second pass with the SAME processor instance — should hit cache.
    proc = MicroCompact(trigger_at_result_bytes=50)
    twice = await apply_processor(proc, once, ctx=_ctx(model))
    third = await apply_processor(proc, twice, ctx=_ctx(model))
    assert [m.model_dump_json() for m in once] == [m.model_dump_json() for m in twice]
    assert [m.model_dump_json() for m in twice] == [m.model_dump_json() for m in third]


async def test_microcompact_skips_already_summarized_blocks() -> None:
    """Older summarized markers stay byte-identical even if the threshold
    would otherwise trigger them (the marker itself is small anyway)."""
    pre_summarized = ToolResultBlock(
        tool_call_id="c1",
        content=f"{MICROCOMPACT_MARKER_PREFIX}abc123] old summary",
    )
    msgs = [Message(role="tool", content=[pre_summarized], timestamp=_ts())]
    model = _ScriptedSummaryModel()
    ctx = _ctx(model)
    out = await apply_processor(MicroCompact(trigger_at_result_bytes=0), msgs, ctx=ctx)
    assert out == msgs
    assert model.call_count == 0


async def test_microcompact_noop_when_no_model_in_ctx() -> None:
    big = ToolResultBlock(tool_call_id="c1", content="x" * 200)
    msgs = [Message(role="tool", content=[big], timestamp=_ts())]
    out = await apply_processor(MicroCompact(trigger_at_result_bytes=50), msgs, ctx=None)
    assert out == msgs


async def test_microcompact_publishes_compaction_events() -> None:
    bus = InMemoryEventBus()
    sub = bus.subscribe()
    big = ToolResultBlock(tool_call_id="c1", content="x" * 200)
    msgs = [Message(role="tool", content=[big], timestamp=_ts())]
    model = _ScriptedSummaryModel()
    ctx = _ctx(model, bus=bus)
    await apply_processor(MicroCompact(trigger_at_result_bytes=50), msgs, ctx=ctx)
    await bus.close()
    seen: list[Any] = [ev async for ev in sub]
    kinds = [type(ev).__name__ for ev in seen]
    assert "CompactionStart" in kinds
    assert "CompactionEnd" in kinds


async def test_microcompact_rejects_negative_trigger() -> None:
    with pytest.raises(ValueError, match="trigger_at_result_bytes"):
        MicroCompact(trigger_at_result_bytes=-1)


# --- ContextCollapse --------------------------------------------------------


def _tc_msg(call_id: str, name: str, **args: Any) -> Message:
    return Message(
        role="assistant",
        content=[ToolCallBlock(id=call_id, name=name, arguments=args)],
        timestamp=_ts(),
    )


def _tr_msg(call_id: str, body: str) -> Message:
    return Message(
        role="tool",
        content=[ToolResultBlock(tool_call_id=call_id, content=body)],
        timestamp=_ts(),
    )


async def test_context_collapse_folds_read_edit_pair() -> None:
    msgs = [
        _tc_msg("c1", "Read", path="/a"),
        _tr_msg("c1", "contents"),
        _tc_msg("c2", "Edit", path="/a"),
        _tr_msg("c2", "ok"),
        Message(role="user", content=[TextBlock(text="thanks")], timestamp=_ts()),
    ]
    out = await apply_processor(ContextCollapse(), msgs)
    assert len(out) == 2
    text = out[0].content[0].text  # type: ignore[union-attr]
    assert text.startswith(COLLAPSE_MARKER_PREFIX)
    assert "Read+Edit on /a" in text


async def test_context_collapse_picks_up_verify_pair() -> None:
    msgs = [
        _tc_msg("c1", "Read", path="/a"),
        _tr_msg("c1", "contents"),
        _tc_msg("c2", "Edit", path="/a"),
        _tr_msg("c2", "ok"),
        _tc_msg("c3", "Grep"),
        _tr_msg("c3", "no matches"),
    ]
    out = await apply_processor(ContextCollapse(), msgs)
    assert len(out) == 1  # The whole six-message run collapsed.


async def test_context_collapse_is_deterministic_idempotent() -> None:
    msgs = [
        _tc_msg("c1", "Read", path="/a"),
        _tr_msg("c1", "contents"),
        _tc_msg("c2", "Edit", path="/a"),
        _tr_msg("c2", "ok"),
    ]
    once = await apply_processor(ContextCollapse(), msgs)
    twice = await apply_processor(ContextCollapse(), once)
    assert [m.model_dump_json() for m in once] == [m.model_dump_json() for m in twice]


async def test_context_collapse_passes_through_when_no_pattern_matches() -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts()),
        _tc_msg("c1", "SomeOtherTool"),
        _tr_msg("c1", "x"),
    ]
    out = await apply_processor(ContextCollapse(), msgs)
    assert out == msgs


# --- SummarizeOldTurns ------------------------------------------------------


def _user_msg(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)], timestamp=_ts())


def _assistant_msg(text: str) -> Message:
    return Message(role="assistant", content=[TextBlock(text=text)], timestamp=_ts())


async def test_summarize_old_turns_noop_under_threshold() -> None:
    msgs = [_user_msg("hi"), _assistant_msg("hello")]
    model = _ScriptedSummaryModel()
    ctx = _ctx(model)
    out = await apply_processor(SummarizeOldTurns(trigger_at_tokens=1_000_000), msgs, ctx=ctx)
    assert out == msgs
    assert model.call_count == 0


async def test_summarize_old_turns_summarizes_old_keeps_recent() -> None:
    msgs = [
        _user_msg("ancient 1"),
        _assistant_msg("ancient 2"),
        _user_msg("ancient 3"),
        _user_msg("recent 1"),
        _assistant_msg("recent 2"),
    ]
    model = _ScriptedSummaryModel(summary_text="<<COMPRESSED>>")
    ctx = _ctx(model)
    out = await apply_processor(
        SummarizeOldTurns(trigger_at_tokens=0, keep_recent_turns=2), msgs, ctx=ctx
    )
    assert len(out) == 3
    head = out[0].content[0].text  # type: ignore[union-attr]
    assert head.startswith(SUMMARY_MARKER_PREFIX)
    assert "<<COMPRESSED>>" in head
    # Recent two are byte-identical to the originals.
    assert out[1] == msgs[-2]
    assert out[2] == msgs[-1]


async def test_summarize_old_turns_attributes_usage() -> None:
    msgs = [_user_msg(f"turn {i}") for i in range(10)]
    model = _ScriptedSummaryModel(usage=Usage(input_tokens=5, output_tokens=3))
    ctx = _ctx(model)
    await apply_processor(
        SummarizeOldTurns(trigger_at_tokens=0, keep_recent_turns=2), msgs, ctx=ctx
    )
    assert ctx.usage.input_tokens == 5
    assert ctx.usage.output_tokens == 3


async def test_summarize_old_turns_caches_marker_for_cache_preservation() -> None:
    """HP4: same prefix on a later turn ⇒ same marker bytes ⇒ no LLM call."""
    msgs = [_user_msg(f"old {i}") for i in range(8)]
    proc = SummarizeOldTurns(trigger_at_tokens=0, keep_recent_turns=2)
    model = _ScriptedSummaryModel(summary_text="canonical")
    ctx1 = _ctx(model)
    out1 = await apply_processor(proc, msgs, ctx=ctx1)
    calls_after_first = model.call_count
    ctx2 = _ctx(model)
    out2 = await apply_processor(proc, msgs, ctx=ctx2)
    assert model.call_count == calls_after_first  # Cache hit ⇒ no new LLM call.
    assert [m.model_dump_json() for m in out1] == [m.model_dump_json() for m in out2]


async def test_summarize_old_turns_skips_already_summarized_prefix() -> None:
    """If older messages collapsed to a single summary marker, do not
    re-summarize on the next pass."""
    msgs = [
        Message(
            role="user",
            content=[TextBlock(text=f"{SUMMARY_MARKER_PREFIX}abc] prior")],
            timestamp=_ts(),
        ),
        _user_msg("recent 1"),
        _assistant_msg("recent 2"),
    ]
    model = _ScriptedSummaryModel()
    ctx = _ctx(model)
    out = await apply_processor(
        SummarizeOldTurns(trigger_at_tokens=0, keep_recent_turns=2), msgs, ctx=ctx
    )
    assert out == msgs
    assert model.call_count == 0


async def test_summarize_old_turns_uses_default_template_by_default() -> None:
    proc = SummarizeOldTurns(trigger_at_tokens=0)
    assert proc.summary_template == DEFAULT_SUMMARY_TEMPLATE


async def test_summarize_old_turns_rejects_negative_trigger() -> None:
    with pytest.raises(ValueError, match="trigger_at_tokens"):
        SummarizeOldTurns(trigger_at_tokens=-1)


async def test_summarize_old_turns_rejects_negative_keep_recent_turns() -> None:
    with pytest.raises(ValueError, match="keep_recent_turns"):
        SummarizeOldTurns(trigger_at_tokens=0, keep_recent_turns=-1)


async def test_summarize_old_turns_publishes_events() -> None:
    bus = InMemoryEventBus()
    sub = bus.subscribe()
    msgs = [_user_msg(f"turn {i}") for i in range(10)]
    model = _ScriptedSummaryModel()
    ctx = _ctx(model, bus=bus)
    await apply_processor(
        SummarizeOldTurns(trigger_at_tokens=0, keep_recent_turns=2), msgs, ctx=ctx
    )
    await bus.close()
    seen = [ev async for ev in sub]
    assert any(isinstance(ev, CompactionStart) for ev in seen)
    assert any(isinstance(ev, CompactionEnd) for ev in seen)


# --- ProviderSideCompaction -------------------------------------------------


@dataclass(slots=True)
class _CompactCapableModel:
    """A model that *does* implement server-side compaction."""

    compacted_to: list[Message] = field(default_factory=list)
    capabilities: ModelCapabilities = field(
        default_factory=lambda: ModelCapabilities(context_window=200_000, supports_compaction=True)
    )

    async def request(
        self,
        messages: list[Message],
        tools: list[Any],
        settings: ModelSettings,
    ) -> AsyncIterator[Any]:  # pragma: no cover - not called
        del messages, tools, settings
        _never: bool = False
        if _never:
            yield None

    async def compact_messages(self, msgs: list[Message]) -> list[Message]:
        self.compacted_to = list(msgs[-1:])
        return self.compacted_to


async def test_provider_side_compaction_delegates_when_supported() -> None:
    msgs = [_user_msg("a"), _user_msg("b"), _user_msg("c")]
    model = _CompactCapableModel()
    out = await apply_processor(ProviderSideCompaction(), msgs, ctx=_ctx(model))
    assert len(out) == 1
    assert out[0] == msgs[-1]


async def test_provider_side_compaction_noop_when_unsupported() -> None:
    msgs = [_user_msg("a"), _user_msg("b")]
    model = _ScriptedSummaryModel()  # supports_compaction defaults to False
    out = await apply_processor(ProviderSideCompaction(), msgs, ctx=_ctx(model))
    assert out == msgs


async def test_provider_side_compaction_noop_without_ctx() -> None:
    msgs = [_user_msg("a")]
    out = await apply_processor(ProviderSideCompaction(), msgs, ctx=None)
    assert out == msgs


async def test_provider_side_compaction_publishes_events_when_supported() -> None:
    bus = InMemoryEventBus()
    sub = bus.subscribe()
    msgs = [_user_msg("a"), _user_msg("b")]
    model = _CompactCapableModel()
    ctx = _ctx(model, bus=bus)
    await apply_processor(ProviderSideCompaction(), msgs, ctx=ctx)
    await bus.close()
    seen = [ev async for ev in sub]
    assert any(isinstance(ev, CompactionStart) for ev in seen)
    assert any(isinstance(ev, CompactionEnd) for ev in seen)
