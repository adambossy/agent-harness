# agent-harness

A simple-but-complete Python agent harness. ~3,000 LOC core; 12 components behind Protocols; no god-classes.

**Design**: see `/Users/adambossy/code/agent-harness-research/` for the full proposal (open `README.html` in a browser). Worker agents implementing this project MUST read `AGENTS.md` in this repo before starting.

## Quickstart

```bash
# install deps + dev tools
uv sync --dev

# install pre-commit hooks (one-time)
uv run pre-commit install

# run lints / formatters
uv run pre-commit run --all-files

# run tests
uv run pytest
```

## Layout

```
agent_harness/
├── core/           the loop, types, Protocols (target <3k LOC)
├── providers/      built-in Model + Provider adapters (gemini-3.5-flash, opus-4.7, gpt-5.5)
├── sandboxes/      InProcessSandbox, ModalSandbox, FlySandbox
├── sessions/       InMemorySession, SqliteSession, RedisSession
├── long_term/      MemdirLongTermMemory (default), VectorLongTermMemory (skeleton; pgvector)
├── tracing/        console subscriber (core); OTEL subscriber (skeleton)
└── extras/         shadow-git checkpoints (activates when git is present), mentions, ignoreset
tests/
├── fakes.py        FakeModel, FakeProvider, FakeSandbox
├── unit/           per-Protocol unit tests
└── integration/    end-to-end; the smoke test lives here
```

## Coding standards

- Python 3.13, async-first.
- **Strict typing project-wide**: `mypy --strict` over `agent_harness/` and `tests/`. Pre-commit enforces.
- Formatting + linting: `ruff` (config in `pyproject.toml`). Pre-commit enforces.
- Tests: `pytest` + `pytest-asyncio`. Use `FakeModel` + `InMemorySession` for integration tests.
- Don't import a concrete `Provider`/`Sandbox`/`Session` from `core/`. Only Protocols.
- Don't grow a god-file. Largest core file: <500 LOC.

## Status

v0.0.1 — skeleton. Implementation orchestrated from the research repo. See its `proposal/orchestration.md`.
