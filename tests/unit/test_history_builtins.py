"""Unit tests for built-in :data:`HistoryProcessor`s.

Covers :class:`TokenBudgetCap`, :class:`DedupFileReads`, :class:`HistorySnip` —
happy paths, idempotency, cache-preservation (HP4), and error paths.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_harness.core.history import (
    DEDUP_MARKER_FMT,
    TRUNCATED_MARKER,
    DedupFileReads,
    HistorySnip,
    TokenBudgetCap,
    apply_processor,
)
from agent_harness.core.models import (
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _tool_call(call_id: str, name: str, **args: object) -> Message:
    return Message(
        role="assistant",
        content=[ToolCallBlock(id=call_id, name=name, arguments=dict(args))],
        timestamp=_ts(),
    )


def _tool_result(call_id: str, body: str) -> Message:
    return Message(
        role="tool",
        content=[ToolResultBlock(tool_call_id=call_id, content=body)],
        timestamp=_ts(),
    )


# --- TokenBudgetCap ---------------------------------------------------------


async def test_token_budget_cap_truncates_older_results_first() -> None:
    msgs = [
        _tool_result("c1", "a" * 1000),
        _tool_result("c2", "b" * 1000),
        _tool_result("c3", "c" * 200),
    ]
    out = await apply_processor(TokenBudgetCap(max_bytes=500), msgs)
    # Newest (c3) is preserved; older two are replaced with the marker.
    assert out[2].content[0].content == "c" * 200  # type: ignore[union-attr]
    assert out[1].content[0].content == TRUNCATED_MARKER  # type: ignore[union-attr]
    assert out[0].content[0].content == TRUNCATED_MARKER  # type: ignore[union-attr]


async def test_token_budget_cap_keeps_everything_when_under_budget() -> None:
    msgs = [_tool_result("c1", "x" * 10), _tool_result("c2", "y" * 10)]
    out = await apply_processor(TokenBudgetCap(max_bytes=1_000_000), msgs)
    assert out == msgs


async def test_token_budget_cap_is_idempotent() -> None:
    """Running the processor twice must produce byte-identical output —
    that is the cache-preservation invariant (HP4).
    """
    msgs = [
        _tool_result("c1", "a" * 1000),
        _tool_result("c2", "b" * 100),
    ]
    once = await apply_processor(TokenBudgetCap(max_bytes=500), msgs)
    twice = await apply_processor(TokenBudgetCap(max_bytes=500), once)
    assert [m.model_dump_json() for m in once] == [m.model_dump_json() for m in twice]


async def test_token_budget_cap_no_tool_results_is_a_noop() -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts()),
        Message(role="assistant", content=[TextBlock(text="hello")], timestamp=_ts()),
    ]
    out = await apply_processor(TokenBudgetCap(max_bytes=10), msgs)
    assert out == msgs


async def test_token_budget_cap_rejects_negative_budget() -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        TokenBudgetCap(max_bytes=-1)


async def test_token_budget_cap_preserves_other_content_blocks() -> None:
    msg = Message(
        role="tool",
        content=[
            TextBlock(text="preface"),
            ToolResultBlock(tool_call_id="c1", content="x" * 1000),
        ],
        timestamp=_ts(),
    )
    msgs = [msg, _tool_result("c2", "small")]
    out = await apply_processor(TokenBudgetCap(max_bytes=10), msgs)
    # Surrounding TextBlock is untouched; only the ToolResultBlock changed.
    assert isinstance(out[0].content[0], TextBlock)
    assert out[0].content[0].text == "preface"
    assert out[0].content[1].content == TRUNCATED_MARKER  # type: ignore[union-attr]


# --- DedupFileReads ---------------------------------------------------------


async def test_dedup_file_reads_replaces_earlier_duplicate() -> None:
    msgs = [
        _tool_call("c1", "Read", path="/a"),
        _tool_result("c1", "v1"),
        _tool_call("c2", "Read", path="/a"),
        _tool_result("c2", "v2"),
    ]
    out = await apply_processor(DedupFileReads(), msgs)
    assert out[1].content[0].content == DEDUP_MARKER_FMT  # type: ignore[union-attr]
    assert out[3].content[0].content == "v2"  # type: ignore[union-attr]


async def test_dedup_file_reads_preserves_unique_paths() -> None:
    msgs = [
        _tool_call("c1", "Read", path="/a"),
        _tool_result("c1", "A"),
        _tool_call("c2", "Read", path="/b"),
        _tool_result("c2", "B"),
    ]
    out = await apply_processor(DedupFileReads(), msgs)
    assert out == msgs


async def test_dedup_file_reads_honors_keep_latest_n() -> None:
    msgs = [
        _tool_call("c1", "Read", path="/a"),
        _tool_result("c1", "v1"),
        _tool_call("c2", "Read", path="/a"),
        _tool_result("c2", "v2"),
        _tool_call("c3", "Read", path="/a"),
        _tool_result("c3", "v3"),
    ]
    out = await apply_processor(DedupFileReads(keep_latest_n=2), msgs)
    # Only the very first read (v1) should be deduped; v2 and v3 kept.
    assert out[1].content[0].content == DEDUP_MARKER_FMT  # type: ignore[union-attr]
    assert out[3].content[0].content == "v2"  # type: ignore[union-attr]
    assert out[5].content[0].content == "v3"  # type: ignore[union-attr]


async def test_dedup_file_reads_supports_file_path_arg_name() -> None:
    msgs = [
        _tool_call("c1", "Read", file_path="/x"),
        _tool_result("c1", "v1"),
        _tool_call("c2", "Read", file_path="/x"),
        _tool_result("c2", "v2"),
    ]
    out = await apply_processor(DedupFileReads(), msgs)
    assert out[1].content[0].content == DEDUP_MARKER_FMT  # type: ignore[union-attr]


async def test_dedup_file_reads_is_idempotent() -> None:
    msgs = [
        _tool_call("c1", "Read", path="/a"),
        _tool_result("c1", "v1"),
        _tool_call("c2", "Read", path="/a"),
        _tool_result("c2", "v2"),
    ]
    once = await apply_processor(DedupFileReads(), msgs)
    twice = await apply_processor(DedupFileReads(), once)
    assert [m.model_dump_json() for m in once] == [m.model_dump_json() for m in twice]


async def test_dedup_file_reads_rejects_zero_keep_latest_n() -> None:
    with pytest.raises(ValueError, match="keep_latest_n"):
        DedupFileReads(keep_latest_n=0)


async def test_dedup_file_reads_ignores_non_read_tools() -> None:
    msgs = [
        _tool_call("c1", "Write", path="/a"),
        _tool_result("c1", "wrote"),
        _tool_call("c2", "Write", path="/a"),
        _tool_result("c2", "wrote again"),
    ]
    out = await apply_processor(DedupFileReads(), msgs)
    assert out == msgs


# --- HistorySnip ------------------------------------------------------------


async def test_history_snip_drops_empty_assistant_message() -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts()),
        Message(role="assistant", content=[], timestamp=_ts()),
        Message(role="assistant", content=[TextBlock(text="reply")], timestamp=_ts()),
    ]
    out = await apply_processor(HistorySnip(), msgs)
    assert len(out) == 2
    assert [m.role for m in out] == ["user", "assistant"]


async def test_history_snip_preserves_empty_system_messages() -> None:
    msgs = [
        Message(role="system", content=[], timestamp=_ts()),
        Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts()),
    ]
    out = await apply_processor(HistorySnip(), msgs)
    assert len(out) == 2


async def test_history_snip_is_idempotent() -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts()),
        Message(role="assistant", content=[], timestamp=_ts()),
    ]
    once = await apply_processor(HistorySnip(), msgs)
    twice = await apply_processor(HistorySnip(), once)
    assert [m.model_dump_json() for m in once] == [m.model_dump_json() for m in twice]


async def test_history_snip_passes_through_when_nothing_to_drop() -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="hi")], timestamp=_ts()),
        Message(role="assistant", content=[TextBlock(text="hello")], timestamp=_ts()),
    ]
    out = await apply_processor(HistorySnip(), msgs)
    assert out == msgs
