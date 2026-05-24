# AGENTS.md — Standing brief for every worker agent

> **READ THIS FIRST**. Every worker agent implementing a component of `agent-harness` MUST read this file at the start of its work.

---

## What this project is

`agent-harness` is a Python 3.13 agent harness, ~3,000 LOC core, 12 components behind Protocols, async-first, strictly typed, no god-classes.

The **full design** lives in `/Users/adambossy/code/agent-harness-research/`. Browse the HTML build (`open /Users/adambossy/code/agent-harness-research/README.html`) or read the markdown sources directly.

Required reading before you start:

- `proposal/README.md` — the design proposal entry point
- `proposal/architecture.md` — top-level architecture diagram
- `proposal/components/<your-component>.md` — interface, requirements, rationale for the component(s) you own
- `proposal/requirements.md` — system + per-component requirements (look for your component's ID prefix)
- `proposal/data-flow.md` — single-turn walkthrough; understand how your component participates
- `proposal/open-questions.md` — **decisions that may affect you**
- `proposal/dependency-graph.md` — what you can rely on from prior waves
- `proposal/orchestration.md` — how this work is structured

---

## What "done" means

A component is **done** when ALL of the following hold:

- ✅ Implementation matches the documented Python interface in `proposal/components/<file>.md`
- ✅ Unit tests in `tests/unit/test_<file>.py` cover happy path + at least one error path; all pass
- ✅ `uv run pre-commit run --all-files` is clean (ruff lint, ruff format, mypy --strict, etc.)
- ✅ Per-component requirements satisfied (numbered IDs in the spec — AG/LP/MD/PV/SS/LT/SB/FT/HK/EV/EX/HC/etc.)
- ✅ Per-file LOC under the target (see `proposal/directory-structure.md`)
- ✅ Every public type has a docstring with one usage example
- ✅ Conflicts logged to `agent-harness-research/work-products/conflicts/<your-team>.md`
- ✅ Assumptions logged to `agent-harness-research/work-products/assumptions/<your-team>.md`

---

## Conventions

### Python

- Python 3.13. Use modern syntax (`X | Y` for unions, `list[X]` for generics, `match` statements where they help).
- Async-first. Use `async def` for anything that may touch I/O, the model, or the sandbox.
- **Type annotations on everything.** mypy strict will fail otherwise.
- No `typing.Any` unless commented with a reason.
- Prefer `Protocol` over `ABC` for duck-typed contracts.
- Prefer `dataclass(frozen=True, slots=True)` or `pydantic.BaseModel` for record types; pick `BaseModel` when validation is needed, `dataclass` when not.
- Use `pathlib.Path` for filesystem paths, never `os.path`.

### Imports

- Absolute imports from external libs.
- Relative imports within the same subpackage (`from .events import EventBus`).
- Cross-subpackage imports: absolute (`from agent_harness.core.events import EventBus`).
- **Core may NOT import from `providers/`, `sandboxes/`, `sessions/`, `long_term/`, `extras/`.** Core knows Protocols only.

### Errors

- Raise types from `agent_harness.core.errors`. Don't invent new exception hierarchies in non-core modules.
- Don't catch and re-raise without adding context (don't `except: raise` to "be safe").

### Logging / observability

- Don't add a logging library. Use the `EventBus` (S14). Publish typed `Event`s.
- For dev/local diagnostics, the `tracing/console.py` subscriber prints events.

### Naming

- Modules: `snake_case.py`.
- Classes: `PascalCase`.
- Functions, variables: `snake_case`.
- Type variables: `T`, `Out`, `Deps` (single capital) for simple cases; `PascalCase` for descriptive.
- Constants: `UPPER_SNAKE_CASE`.

---

## Workflow

1. Read this file (you are doing so).
2. Read the spec for each file you'll implement.
3. Read `proposal/requirements.md` — find your component's requirement IDs.
4. Read `proposal/open-questions.md` — find any decision that affects your component.
5. Implement.
6. Write unit tests as you go (TDD if it helps; aim for ≥90% line coverage on core).
7. **Log conflicts** the moment you encounter them (see below).
8. Run `uv run pre-commit run --all-files` until clean.
9. Run `uv run pytest tests/unit/test_<file>.py -v` until green.
10. **Log assumptions** on completion (see below).
11. Report back: files changed, summary of choices, paths to logs, anything flagged.

---

## Logging conflicts

A **conflict** is a contradiction between two specs / requirements / decisions / docs. Examples:

- The component spec says X but the requirements say Y.
- An open-question decision contradicts a component's documented interface.
- The data-flow walkthrough implies a behavior that requirements.md doesn't list.

**When to log**: as soon as you spot a conflict. Don't wait.

**Where**: append a new section to
```
/Users/adambossy/code/agent-harness-research/work-products/conflicts/<your-team>.md
```

If the file doesn't exist yet, create it with the header from
`/Users/adambossy/code/agent-harness-research/work-products/conflicts/README.md`.

**Entry template** (one per conflict):

```markdown
### YYYY-MM-DD: <one-line title>

**Where**: `<file path in agent_harness/>`

**What I found**: <concrete description>

**Spec A**: <citation — e.g., `proposal/components/foo.md` § X>

**Spec B**: <conflicting citation>

**Why it's a conflict**: <explanation>

**My current resolution (pending review)**: <what I did to proceed>

**Severity**: blocking | should-fix | nit
```

You can proceed past a conflict by making a judgment call and noting it. The orchestrator will review.

---

## Logging assumptions

An **assumption** is a judgment call you made because the spec was ambiguous or silent. Examples:

- Spec didn't say what happens when a list is empty — you picked one behavior.
- Spec implied an ordering but didn't specify; you chose a reasonable one.
- An optional field's default value wasn't specified.

**When to log**: on completion, before reporting done.

**Where**:
```
/Users/adambossy/code/agent-harness-research/work-products/assumptions/<your-team>.md
```

**Entry template**:

```markdown
### YYYY-MM-DD: <one-line title>

**Where**: `<file path in agent_harness/>`

**Spec gap**: <what was unclear or silent>

**Assumption**: <what you assumed>

**Rationale**: <why this seemed reasonable>

**Risk if wrong**: <what breaks if the user disagrees>
```

---

## Things you must NOT do

- ❌ Don't import a concrete provider/sandbox/session/ltm from `core/`. Core knows only Protocols.
- ❌ Don't grow a god-file. Core files target <500 LOC each. Split if you're approaching.
- ❌ Don't add a new `HookEvent` name without flagging it as a conflict. The taxonomy is load-bearing.
- ❌ Don't suppress mypy errors with `# type: ignore` without a comment explaining why.
- ❌ Don't disable pre-commit hooks or skip them with `--no-verify`.
- ❌ Don't add a logging dependency. Use the `EventBus`.
- ❌ Don't add a new top-level dependency without flagging it. Stick to what's already in `pyproject.toml`.
- ❌ Don't refactor across components. Stay within your scope; flag anything cross-cutting as a conflict.

---

## Where things live

```
agent_harness/                              # this project, code goes here
├── core/                                   # the inviolate core (~2,800 LOC)
├── providers/                              # gemini-3.5-flash, opus-4.7, gpt-5.5
├── sandboxes/                              # InProcess, Modal, Fly
├── sessions/                               # InMemory, Sqlite, Redis
├── long_term/                              # Memdir (default), Vector (skeleton)
├── tracing/                                # console (impl), otel (skeleton)
└── extras/                                 # checkpoints, mentions, ignoreset

tests/
├── fakes.py                                # FakeModel, FakeProvider, FakeSandbox
├── unit/test_<file>.py                     # per-Protocol unit tests
└── integration/test_smoke.py               # the E2E smoke test

/Users/adambossy/code/agent-harness-research/
└── proposal/                               # READ THIS — the design specs
    ├── components/<file>.md                # per-component spec
    ├── requirements.md                     # numbered requirement IDs
    └── open-questions.md                   # decisions

/Users/adambossy/code/agent-harness-research/work-products/
├── conflicts/<team>.md                     # YOUR conflict log
└── assumptions/<team>.md                   # YOUR assumption log
```

---

## Final checklist before reporting done

- [ ] All implementation files created and match the spec interface
- [ ] All requirement IDs (per component) are satisfied
- [ ] Unit tests written; happy path + at least one error path; all green
- [ ] `uv run pre-commit run --all-files` clean
- [ ] `uv run pytest` green for files you touched
- [ ] Conflicts log exists; severities noted; resolutions documented
- [ ] Assumptions log exists; risks noted
- [ ] No `# type: ignore` without rationale
- [ ] No god-file; per-file LOC under target
- [ ] Public types have docstrings with usage examples
- [ ] You're ready for an architecture + code-quality review

When in doubt, **flag it as a conflict or assumption** rather than guessing silently.
