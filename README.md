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

## Getting started

A minimal agent is a model plus a prompt:

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

### Giving the agent tools

Decorate a function with `@tool` (the input schema is read from its type hints,
the descriptions from its docstring) and hand it to the agent in a toolset:

```python
import asyncio

from agent_harness import Agent, StaticToolset, tool
from agent_harness.providers.anthropic import AnthropicProvider, AnthropicMessagesModel


@tool
async def get_weather(city: str) -> str:
    """Look up the current weather for a city.

    Args:
        city: Name of the city to look up.
    """
    # ... call a real weather API here ...
    return f"It's sunny in {city}."


provider = AnthropicProvider(api_key="sk-...")
model = AnthropicMessagesModel(provider=provider)

agent = Agent(
    name="assistant",
    model=model,
    instructions="You are a helpful assistant.",
    toolsets=[StaticToolset(name="tools", tools=[get_weather])],
)

result = asyncio.run(agent.run("What's the weather in Paris?"))
print(result.output)  # -> e.g. "It's sunny in Paris."
```

### OpenRouter (GLM-5.2, Kimi K3, …)

OpenRouter's `/api/v1/messages` endpoint is Anthropic-compatible, so the
Anthropic model adapter drives it directly. Every request carries a routing
policy restricting which upstream providers may serve it:

```python
from agent_harness.providers.openrouter import (
    CAPS_GLM_5_2, GLM_5_2, OpenRouterModel, OpenRouterProvider,
)

provider = OpenRouterProvider(api_key="sk-or-…")
model = OpenRouterModel(provider=provider, name=GLM_5_2, capabilities=CAPS_GLM_5_2)
agent = Agent(name="assistant", model=model)
```

The default `US_FP8_ZDR` policy routes only to US-headquartered,
FP8-or-better, zero-data-retention endpoints, cheapest first. For Kimi K3,
pass `routing=MOONSHOT_DIRECT` — it has a single upstream endpoint
(Singapore, int4), so the stricter default would match nothing.

Swap the model for another provider (`agent_harness.providers.openai`,
`.google`), persist history with a session
(`from agent_harness.sessions import SqliteSession`), or run tools inside an
isolated sandbox (`from agent_harness.sandboxes import ModalSandbox`) — all via
the same `Agent` constructor.

> **Testing without an API key:** `tests/fakes.py` ships a `FakeModel` that
> replays scripted turns, so you can exercise the full loop (including tool
> dispatch) without a network call.

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
