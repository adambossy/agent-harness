# agent-harness

A simple-but-complete Python agent harness. ~3,000 LOC core; 12 components behind Protocols; no god-classes.

Async-first, strictly typed (`mypy --strict`), and ships its own types (`py.typed`). The core defines the contract — `Agent`, the model/tool/session/sandbox Protocols, the event stream — and concrete backends plug in behind it so you only pull the provider SDKs you actually use.

## Install

```bash
# core only
pip install agent-harness

# with the backends you need (extras are optional)
pip install "agent-harness[anthropic,redis,modal]"
```

Available extras: `anthropic`, `openai`, `google`, `redis`, `modal`, `fly`, `otel`, `vector`, `mcp`.

## Usage

```python
import asyncio

from agent_harness import Agent
from agent_harness.providers.anthropic import AnthropicProvider, AnthropicMessagesModel

provider = AnthropicProvider(api_key="sk-...")
model = AnthropicMessagesModel(provider=provider)

agent = Agent(name="assistant", model=model)
result = asyncio.run(agent.run("Hello!"))
print(result.output)
```

The stable surface is re-exported from the top level (`from agent_harness import Agent, tool, Message, ...`); swappable backends live in their sub-packages (`agent_harness.providers`, `.sessions`, `.sandboxes`, `.long_term`, `.tracing`).

## Layout

```
agent_harness/
├── core/           the loop, types, Protocols (target <3k LOC)
├── providers/      built-in Model + Provider adapters (Anthropic, OpenAI, Google)
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

## Development

```bash
# install deps + dev tools
uv sync --dev

# install pre-commit hooks (one-time)
uv run pre-commit install

# lint / format / type-check (all enforced by pre-commit)
uv run pre-commit run --all-files

# run tests
uv run pytest
```

### Coding standards

- Python 3.13, async-first.
- **Strict typing project-wide**: `mypy --strict` over `agent_harness/` and `tests/`.
- Formatting + linting: `ruff` (config in `pyproject.toml`).
- Tests: `pytest` + `pytest-asyncio`. Use `FakeModel` + `InMemorySession` for integration tests.
- Don't import a concrete `Provider`/`Sandbox`/`Session` from `core/`. Only Protocols.
- Don't grow a god-file. Largest core file stays under ~600 LOC.

## Status

v0.0.1 — early. The core loop, providers, sessions, and sandboxes work; the OTEL and vector-memory backends are skeletons.
