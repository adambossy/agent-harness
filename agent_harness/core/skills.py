"""Skills — progressive-context loading primitive (core, not extras).

A **skill** is a markdown file with YAML frontmatter:

```markdown
---
name: code-review
description: Reviews a diff for correctness, security, style.
when_to_use: After a non-trivial code change. ALWAYS before merging.
allowed_tools: [Read, Grep, Glob]
---

# Body text the model loads on demand …
```

The model is shown only **name + description + when_to_use** in its initial
system prompt (cheap, token-wise). Frontmatter alone is loaded at registry
init; the body lives on disk and is fetched via the built-in ``Skill`` tool
returned by :func:`build_skill_tool`. That tool's body is just
``Path(body_path).read_text()`` — there's no execution of skill content.

The :class:`SkillRegistry` walks two filesystem layers (project + user) and
indexes by ``name``; project skills shadow user skills, matching the
convention from `claude_code` / `Cursor`. Discovery is non-recursive — each
skill lives in its own directory as ``<root>/<name>/SKILL.md``.

Per [open-questions.md §8](../../agent-harness-research/proposal/open-questions.md),
skills are a **core** primitive. Activation requires a filesystem surface
(`FilesystemTools` or equivalent) but the loader, registry, and activation
contract live here.

Example:
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     root = Path(tmp)
    ...     d = root / ".agent" / "skills" / "demo"
    ...     d.mkdir(parents=True)
    ...     _ = (d / "SKILL.md").write_text(
    ...         "---\\nname: demo\\ndescription: A demo skill.\\n---\\nhi"
    ...     )
    ...     reg = SkillRegistry.load(project_root=root, user_root=None)
    ...     reg.get("demo").description
    'A demo skill.'
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError, ToolError
from .tools import Tool, ToolPolicy, tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKILL_FILENAME: str = "SKILL.md"
"""Per-skill markdown file. Each skill lives at ``<root>/<name>/SKILL.md``."""

PROJECT_SKILLS_DIRNAME: str = ".agent/skills"
"""Project-scoped skills root, relative to the project root."""

USER_SKILLS_HOME_SUBPATH: str = ".agent_harness/skills"
"""User-scoped skills root, relative to ``~``."""

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<front>.*?)\n---\s*(?:\n(?P<body>.*))?\Z",
    re.DOTALL,
)
"""Match a leading YAML frontmatter block plus optional body."""

# Skill names follow filesystem-friendly conventions: lowercase letters,
# digits, dashes, underscores. The directory name on disk is the canonical
# identifier — we reject anything that wouldn't survive a round-trip through
# the filesystem.
_NAME_RE = re.compile(r"\A[a-zA-Z0-9_][a-zA-Z0-9_\-]*\Z")


# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Skill:
    """Metadata-only skill record.

    The ``body`` lives on disk at :attr:`body_path`; the model fetches it via
    the built-in ``Skill`` tool (see :func:`build_skill_tool`). Keeping the
    body out of memory until requested is the whole point of progressive
    disclosure — it keeps the system-prompt token cost proportional to the
    number of skills, not their total length.

    Fields:
        name: filesystem-friendly identifier; matches the parent dirname.
        description: one-line summary surfaced in the system prompt.
        when_to_use: prose triggers; surfaced alongside description.
        allowed_tools: optional whitelist scoping which tools the skill body
            says it expects. Loop-level enforcement is the loader's concern;
            this field is the *declared* scope.
        body_path: absolute path to the source ``SKILL.md``; body text lives
            after the frontmatter and is read on demand.
        source: which root this skill came from. ``"project"`` shadows
            ``"user"`` when names collide.

    Example:
        >>> Skill(
        ...     name="demo",
        ...     description="d",
        ...     when_to_use=None,
        ...     allowed_tools=(),
        ...     body_path=Path("/tmp/demo/SKILL.md"),
        ...     source="project",
        ... ).name
        'demo'
    """

    name: str
    description: str
    when_to_use: str | None
    allowed_tools: tuple[str, ...]
    body_path: Path
    source: str = "project"

    def read_body(self) -> str:
        """Read and return the markdown body (everything after the frontmatter).

        Raises:
            ToolError: if the body file is missing or unreadable.

        Example:
            >>> import tempfile
            >>> from pathlib import Path
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     p = Path(tmp) / "SKILL.md"
            ...     _ = p.write_text("---\\nname: d\\ndescription: x\\n---\\nbody!")
            ...     s = Skill(
            ...         name="d",
            ...         description="x",
            ...         when_to_use=None,
            ...         allowed_tools=(),
            ...         body_path=p,
            ...         source="project",
            ...     )
            ...     s.read_body()
            'body!'
        """
        try:
            raw = self.body_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolError(
                f"failed to read skill body at {self.body_path}: {exc}",
                cause=exc,
                context={"skill": self.name, "path": str(self.body_path)},
            ) from exc
        match = _FRONTMATTER_RE.match(raw)
        body = match.group("body") if match else raw
        return body or ""


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str, *, path: Path) -> tuple[dict[str, Any], int]:
    """Parse the leading YAML frontmatter block.

    Returns ``(parsed_dict, body_offset_chars)``. Raises :class:`ConfigError`
    if the file lacks a frontmatter block or the YAML is malformed.

    Example:
        >>> from pathlib import Path
        >>> data, _ = _parse_frontmatter("---\\nname: x\\ndescription: y\\n---\\nbody", path=Path("/x"))
        >>> data["name"]
        'x'
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise ConfigError(
            f"skill file {path} is missing a YAML frontmatter block "
            "(`---\\n...\\n---\\n` at the top)",
            context={"path": str(path)},
        )
    front_text = match.group("front")
    try:
        loaded = yaml.safe_load(front_text)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"skill file {path} has malformed YAML frontmatter: {exc}",
            cause=exc,
            context={"path": str(path)},
        ) from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"skill file {path} frontmatter must be a mapping; got " f"{type(loaded).__name__}",
            context={"path": str(path)},
        )
    return loaded, match.end("front")


def _coerce_allowed_tools(value: Any, *, path: Path) -> tuple[str, ...]:
    """Normalize ``allowed_tools`` into a tuple of strings.

    Accepts a list of strings or ``None`` / missing. Anything else is a
    config error — we keep this strict so typos surface early.
    """
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ConfigError(
            f"skill file {path}: `allowed_tools` must be a list of strings",
            context={"path": str(path), "got": repr(value)},
        )
    return tuple(value)


def _load_skill_file(path: Path, *, source: str) -> Skill:
    """Load a single ``SKILL.md`` and return its :class:`Skill` record.

    Frontmatter only — the body remains on disk. The parent directory's name
    is the *canonical* skill name; we surface a config error when the
    frontmatter ``name:`` field disagrees, since the disk name is what the
    user types and the registry indexes by.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"failed to read skill file {path}: {exc}",
            cause=exc,
            context={"path": str(path)},
        ) from exc
    data, _ = _parse_frontmatter(text, path=path)

    dir_name = path.parent.name
    name = data.get("name") or dir_name
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ConfigError(
            f"skill file {path}: `name` must match {_NAME_RE.pattern}",
            context={"path": str(path), "got": repr(name)},
        )
    if name != dir_name:
        raise ConfigError(
            f"skill file {path}: frontmatter `name` {name!r} does not match "
            f"its directory name {dir_name!r}",
            context={"path": str(path), "name": name, "dir": dir_name},
        )

    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ConfigError(
            f"skill file {path}: `description` is required and must be a " "non-empty string",
            context={"path": str(path)},
        )

    when_to_use = data.get("when_to_use")
    if when_to_use is not None and not isinstance(when_to_use, str):
        raise ConfigError(
            f"skill file {path}: `when_to_use` must be a string if present",
            context={"path": str(path), "got": repr(when_to_use)},
        )

    allowed = _coerce_allowed_tools(data.get("allowed_tools"), path=path)

    return Skill(
        name=name,
        description=description.strip(),
        when_to_use=when_to_use.strip() if when_to_use else None,
        allowed_tools=allowed,
        body_path=path,
        source=source,
    )


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SkillRegistry:
    """Indexed collection of discovered skills.

    Project skills (``<project_root>/.agent/skills/<name>/SKILL.md``) shadow
    user skills (``~/.agent_harness/skills/<name>/SKILL.md``) when names
    collide, mirroring the resolution order used by Claude Code and similar
    tools. Construction does not read skill bodies — only frontmatter — so a
    100-skill registry costs O(skill-count) frontmatter parses, not O(total
    body size).

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     d = root / ".agent" / "skills" / "x"
        ...     d.mkdir(parents=True)
        ...     _ = (d / "SKILL.md").write_text("---\\nname: x\\ndescription: y\\n---\\nbody")
        ...     reg = SkillRegistry.load(project_root=root, user_root=None)
        ...     [s.name for s in reg.list_skills()]
        ['x']
    """

    skills: dict[str, Skill] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        *,
        project_root: Path | None = None,
        user_root: Path | None = None,
    ) -> SkillRegistry:
        """Walk both roots and build the registry.

        ``project_root`` is treated as the project's working directory; the
        skills root is ``<project_root>/.agent/skills``. ``user_root`` is the
        user-scope root; the skills root is ``<user_root>/.agent_harness/skills``.

        Passing ``None`` for either ``project_root`` or ``user_root`` **skips
        that scope entirely** — there is no implicit ``Path.home()`` fallback.
        Callers that want the user scope pass ``user_root=Path.home()``
        explicitly. Project entries shadow user entries with the same name.

        Discovery is non-recursive: each skill lives at
        ``<scope_root>/<name>/SKILL.md``. Other files are ignored.
        """
        reg = cls()
        # User scope first so project entries can overwrite (shadow) them.
        if user_root is not None:
            reg._ingest(user_root / USER_SKILLS_HOME_SUBPATH, source="user")
        if project_root is not None:
            reg._ingest(project_root / PROJECT_SKILLS_DIRNAME, source="project")
        return reg

    def _ingest(self, root: Path, *, source: str) -> None:
        """Discover and load every ``<root>/<name>/SKILL.md``.

        Silently skips a missing ``root`` — the user need not create the
        directory just to declare "no skills." Errors inside an individual
        skill file propagate as :class:`ConfigError`.
        """
        if not root.is_dir():
            return
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            skill_file = entry / SKILL_FILENAME
            if not skill_file.is_file():
                continue
            skill = _load_skill_file(skill_file, source=source)
            # Later sources shadow earlier ones; ``load`` ingests user first
            # then project so this assignment achieves project>user precedence.
            self.skills[skill.name] = skill

    def list_skills(self) -> list[Skill]:
        """All loaded skills, name-sorted for deterministic output."""
        return sorted(self.skills.values(), key=lambda s: s.name)

    def get(self, name: str) -> Skill:
        """Look up a skill by name; raise :class:`ToolError` if unknown."""
        try:
            return self.skills[name]
        except KeyError as exc:
            raise ToolError(
                f"no skill named {name!r}",
                context={"requested": name, "available": sorted(self.skills)},
            ) from exc

    def manifest(self) -> list[dict[str, Any]]:
        """Compact metadata listing for the model's system prompt.

        Returns name + description + when_to_use only — never the body or
        ``allowed_tools`` (the body is fetched on demand; ``allowed_tools``
        is a *loader* concern, not a model-facing hint).
        """
        return [
            {
                "name": s.name,
                "description": s.description,
                "when_to_use": s.when_to_use,
            }
            for s in self.list_skills()
        ]


# ---------------------------------------------------------------------------
# Built-in Skill tool
# ---------------------------------------------------------------------------


def build_skill_tool(registry: SkillRegistry) -> Tool:
    """Construct the built-in ``Skill`` tool bound to ``registry``.

    The model invokes ``Skill(skill_name="code-review")`` to fetch the body
    of a skill it saw in the manifest. This function is a factory rather
    than a module-level ``@tool`` because the registry is per-run.

    Example:
        >>> import asyncio, tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     d = root / ".agent" / "skills" / "demo"
        ...     d.mkdir(parents=True)
        ...     _ = (d / "SKILL.md").write_text("---\\nname: demo\\ndescription: x\\n---\\nhello")
        ...     reg = SkillRegistry.load(project_root=root, user_root=None)
        ...     tool_ = build_skill_tool(reg)
        ...     asyncio.run(tool_.fn(skill_name="demo"))
        'hello'
    """

    @tool(
        name="Skill",
        description=(
            "Fetch the body of a skill by name. The model sees only "
            "name + description + when_to_use in the system prompt; "
            "this tool returns the full markdown content on demand."
        ),
        policy=ToolPolicy(is_read_only=True, is_concurrency_safe=True, always_load=True),
    )
    async def skill_body(skill_name: str) -> str:
        """Return the markdown body for the named skill.

        Args:
            skill_name: identifier from the skill manifest (the skill's
                directory name on disk).
        """
        skill = registry.get(skill_name)
        return skill.read_body()

    return skill_body


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def iter_skill_paths(roots: Iterable[Path]) -> list[Path]:
    """Yield every ``SKILL.md`` reachable from ``roots`` (non-recursive).

    Useful for tooling (linters, doc generators) that needs to enumerate
    skills without parsing them. ``roots`` are skill-scope roots — i.e.
    each directly contains one subdirectory per skill.

    Example:
        >>> iter_skill_paths([])
        []
    """
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            candidate = entry / SKILL_FILENAME
            if entry.is_dir() and candidate.is_file():
                out.append(candidate)
    return out
