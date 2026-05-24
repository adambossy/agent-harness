"""Real-LLM end-to-end verification.

Same scenario as ``tests/unit/test_loop_integration.py::test_smoke_two_turns_one_tool_call``
(read a small file, answer a question about it), but driven by the
**actual Anthropic Messages API** via ``AnthropicMessagesModel``. Assertions
relaxed where the model has freedom (output wording, exact event counts).

Run with:
    uv run --env-file .env python scripts/e2e_real.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

from agent_harness.core.agent import Agent
from agent_harness.core.events import (
    AgentEnd,
    AgentStart,
    InMemoryEventBus,
    MessageDelta,
    NodeEnter,
    RunEnd,
    RunStart,
    ToolExecEnd,
    ToolExecStart,
)
from agent_harness.core.filesystem import FilesystemTools
from agent_harness.providers.anthropic import AnthropicMessagesModel, AnthropicProvider
from agent_harness.providers.google import GeminiModel, GoogleProvider
from agent_harness.providers.openai import OpenAIProvider, OpenAIResponsesModel
from agent_harness.sandboxes.inprocess import InProcessSandbox
from agent_harness.sessions.inmemory import InMemorySession


def make_model_from_env() -> tuple[object, object, str]:
    """Pick whichever provider's key works. Returns (model, provider, label)."""
    # Order: Anthropic, OpenAI, Google. First valid key wins.
    preferred = (os.environ.get("AGENT_HARNESS_PROVIDER") or "").lower()
    candidates: list[tuple[str, callable]] = [
        (
            "anthropic",
            lambda: (
                AnthropicMessagesModel(provider=(p := AnthropicProvider())),
                p,
                f"anthropic / {AnthropicMessagesModel(provider=p).name}",
            ),
        ),
        (
            "openai",
            lambda: (
                OpenAIResponsesModel(provider=(p := OpenAIProvider())),
                p,
                f"openai / {OpenAIResponsesModel(provider=p).name}",
            ),
        ),
        (
            "google",
            lambda: (
                GeminiModel(provider=(p := GoogleProvider())),
                p,
                f"google / {GeminiModel(provider=p).name}",
            ),
        ),
    ]
    if preferred:
        candidates = [c for c in candidates if c[0] == preferred] or candidates
    for name, factory in candidates:
        key_env = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }[name]
        if not os.environ.get(key_env):
            continue
        try:
            return factory()
        except Exception as exc:
            print(f"  (skipping {name}: {exc})")
    raise RuntimeError(
        "No provider keys worked. Set ANTHROPIC_API_KEY / " "OPENAI_API_KEY / GOOGLE_API_KEY."
    )


async def main() -> int:
    # Sanity: at least one API key present.
    if not any(
        os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")
    ):
        print("❌ No provider API key set. Did you `uv run --env-file .env ...`?")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        (workspace / "foo.txt").write_text("Hello, world!\nThis is a test file.")

        sandbox = InProcessSandbox(root=str(workspace))
        session = InMemorySession(session_id="real-e2e-001")
        bus = InMemoryEventBus()

        model, provider, label = make_model_from_env()

        agent = Agent(
            name="reader",
            model=model,
            toolsets=[FilesystemTools(sandbox=sandbox)],
            session=session,
            sandbox=sandbox,
            instructions=(
                "You read files from the workspace and explain their contents. "
                "When asked about a file, use the `read` tool to fetch its "
                "contents, then summarize them for the user."
            ),
        )

        collected: list = []
        # Subscribe BEFORE creating the task so the queue is registered before
        # agent.run starts publishing — otherwise RunStart/PrepareTurn etc. are
        # racy and may be lost.
        sub = bus.subscribe()

        async def collect() -> None:
            async for ev in sub:
                collected.append(ev)

        collector = asyncio.create_task(collect())

        t0 = time.perf_counter()
        result = await agent.run(prompt="What's in foo.txt?", event_bus=bus)
        elapsed = time.perf_counter() - t0

        # Signal end-of-stream so the collector iterator exits cleanly.
        await bus.close()
        try:
            await asyncio.wait_for(collector, timeout=2.0)
        except TimeoutError:
            collector.cancel()

        # ─── Render ─────────────────────────────────────────────────
        print("=" * 72)
        print(f"REAL E2E — {label}")
        print(f"Wall-clock: {elapsed:.2f}s")
        print("=" * 72)
        print()
        print("--- result.output ---")
        print(result.output)
        print()
        print(
            f"--- usage --- input={result.usage.input_tokens} "
            f"output={result.usage.output_tokens}"
        )
        print()

        # ─── Assertions (relaxed for real model) ───────────────────
        msgs = await session.get_messages()
        node_seq = [ev.node for ev in collected if isinstance(ev, NodeEnter)]
        tool_execs = [ev for ev in collected if isinstance(ev, ToolExecStart)]
        tool_ends = [ev for ev in collected if isinstance(ev, ToolExecEnd)]
        deltas_by_msg: dict = {}
        for ev in collected:
            if isinstance(ev, MessageDelta):
                deltas_by_msg.setdefault(ev.message_id, []).append(ev)

        snapshots = await session.get_run_states()
        checks = []

        def chk(name: str, ok: bool, detail: str = "") -> None:
            checks.append((name, ok, detail))

        chk("1. result.output is not None", result.output is not None)
        # Fuzzy content match — the model is free to phrase the answer however.
        lower = (result.output or "").lower()
        chk(
            "2. output mentions file contents",
            "hello" in lower or "test" in lower or "world" in lower,
            f"got: {result.output[:80] if result.output else 'None'!r}",
        )
        chk("3. no pending approvals", result.pending_approvals == [])
        chk("4. session captured ≥2 user/assistant messages", len(msgs) >= 2, f"got {len(msgs)}")
        chk("5. at least one tool execution", len(tool_execs) >= 1, f"got {len(tool_execs)}")
        chk(
            "6. tool was 'read' on foo.txt",
            any(
                t.tool_name == "read" and t.arguments.get("path", "").endswith("foo.txt")
                for t in tool_execs
            ),
            f"calls: {[(t.tool_name, t.arguments) for t in tool_execs]}",
        )
        chk(
            "7. tool execution succeeded (no error)",
            all(e.error is None for e in tool_ends),
            f"errors: {[e.error for e in tool_ends if e.error]}",
        )
        chk(
            "8. ≥1 RunStateSnapshot per node visit",
            len(snapshots) >= len(node_seq),
            f"snapshots={len(snapshots)} nodes={len(node_seq)}",
        )
        chk(
            "9. usage tokens > 0",
            result.usage.input_tokens > 0 and result.usage.output_tokens > 0,
            f"in={result.usage.input_tokens} out={result.usage.output_tokens}",
        )
        chk(
            "10. node sequence starts with PrepareTurn → ModelRequest",
            node_seq[:2] == ["PrepareTurn", "ModelRequest"],
            f"seq[:4]={node_seq[:4]}",
        )
        chk(
            "11. loop visited at least 2 ModelRequest nodes (turn + synthesis)",
            node_seq.count("ModelRequest") >= 2,
            f"ModelRequest count={node_seq.count('ModelRequest')}",
        )
        chk(
            "12. RunStart + RunEnd both emitted",
            any(isinstance(e, RunStart) for e in collected)
            and any(isinstance(e, RunEnd) for e in collected),
        )
        chk(
            "13. AgentStart + AgentEnd both emitted",
            any(isinstance(e, AgentStart) for e in collected)
            and any(isinstance(e, AgentEnd) for e in collected),
        )
        # Cumulative-partial invariant: every MessageDelta.partial.text starts-with
        # the previous one within the same message_id.
        cumulative_ok = True
        cumulative_detail = ""
        for mid, deltas in deltas_by_msg.items():
            prev = ""
            for d in deltas:
                cur = d.partial.text
                if not cur.startswith(prev):
                    cumulative_ok = False
                    cumulative_detail = f"non-cumulative in {mid}: {prev!r} → {cur!r}"
                    break
                prev = cur
            if not cumulative_ok:
                break
        chk(
            "14. MessageDelta.partial is cumulative per message_id",
            cumulative_ok,
            cumulative_detail,
        )

        print("--- Assertions ---")
        passed = sum(1 for _, ok, _ in checks if ok)
        for name, ok, detail in checks:
            mark = "✅" if ok else "❌"
            line = f"  {mark} {name}"
            if detail and not ok:
                line += f"\n       {detail}"
            print(line)
        print()
        print(f"--- {passed}/{len(checks)} passed ---")
        print()
        print(f"--- node sequence ({len(node_seq)} nodes) ---")
        print(f"  {' -> '.join(node_seq)}")
        print()
        print(f"--- tool execs ({len(tool_execs)}) ---")
        for t in tool_execs:
            print(f"  {t.tool_name}({t.arguments})")

        return 0 if passed == len(checks) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
