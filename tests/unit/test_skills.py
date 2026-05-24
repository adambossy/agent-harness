"""Unit tests for ``agent_harness.core.skills``."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.core.errors import ConfigError, ToolError
from agent_harness.core.skills import (
    PROJECT_SKILLS_DIRNAME,
    SKILL_FILENAME,
    USER_SKILLS_HOME_SUBPATH,
    Skill,
    SkillRegistry,
    build_skill_tool,
    iter_skill_paths,
)
from agent_harness.core.tools import Tool, ToolPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "default description",
    when_to_use: str | None = None,
    allowed_tools: list[str] | None = None,
    body: str = "body content",
    frontmatter_name: str | None = None,
    project_scope: bool = True,
) -> Path:
    """Create ``<root>/<scope>/<name>/SKILL.md`` and return its path."""
    scope = PROJECT_SKILLS_DIRNAME if project_scope else USER_SKILLS_HOME_SUBPATH
    skill_dir = root / scope / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    front_lines = [f"name: {frontmatter_name or name}", f"description: {description}"]
    if when_to_use is not None:
        front_lines.append(f"when_to_use: {when_to_use}")
    if allowed_tools is not None:
        front_lines.append(f"allowed_tools: {allowed_tools!r}")
    front = "\n".join(front_lines)
    path = skill_dir / SKILL_FILENAME
    path.write_text(f"---\n{front}\n---\n{body}", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Skill dataclass + read_body
# ---------------------------------------------------------------------------


def test_skill_is_immutable() -> None:
    s = Skill(
        name="x",
        description="d",
        when_to_use=None,
        allowed_tools=(),
        body_path=Path("/tmp/x/SKILL.md"),
    )
    with pytest.raises(AttributeError):
        s.name = "y"  # type: ignore[misc]


def test_skill_read_body_returns_only_body(tmp_path: Path) -> None:
    _write_skill(tmp_path, "demo", body="hello-body")
    reg = SkillRegistry.load(project_root=tmp_path, user_root=None)
    skill = reg.get("demo")
    assert skill.read_body() == "hello-body"


def test_skill_read_body_handles_missing_file(tmp_path: Path) -> None:
    bogus = Skill(
        name="ghost",
        description="missing on disk",
        when_to_use=None,
        allowed_tools=(),
        body_path=tmp_path / "nope.md",
    )
    with pytest.raises(ToolError, match="failed to read skill body"):
        bogus.read_body()


# ---------------------------------------------------------------------------
# SkillRegistry.load — discovery
# ---------------------------------------------------------------------------


def test_load_discovers_project_skills(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", description="A skill")
    _write_skill(tmp_path, "beta", description="Beta skill")
    reg = SkillRegistry.load(project_root=tmp_path, user_root=None)
    names = [s.name for s in reg.list_skills()]
    assert names == ["alpha", "beta"]


def test_load_with_missing_roots_yields_empty_registry(tmp_path: Path) -> None:
    """No ``.agent/skills`` directory means no skills, not an error."""
    reg = SkillRegistry.load(project_root=tmp_path, user_root=tmp_path)
    assert reg.list_skills() == []


def test_load_with_none_roots_is_a_noop() -> None:
    reg = SkillRegistry.load(project_root=None, user_root=None)
    assert reg.list_skills() == []


def test_project_shadows_user(tmp_path: Path) -> None:
    """When a name exists in both scopes, project wins."""
    user_home = tmp_path / "home"
    project = tmp_path / "proj"
    _write_skill(user_home, "shared", description="user version", project_scope=False)
    _write_skill(project, "shared", description="project version", project_scope=True)
    reg = SkillRegistry.load(project_root=project, user_root=user_home)
    assert reg.get("shared").description == "project version"
    assert reg.get("shared").source == "project"


def test_user_skill_is_loaded_when_no_project_override(tmp_path: Path) -> None:
    user_home = tmp_path / "home"
    project = tmp_path / "proj"
    _write_skill(user_home, "only-user", description="user", project_scope=False)
    project.mkdir()
    reg = SkillRegistry.load(project_root=project, user_root=user_home)
    s = reg.get("only-user")
    assert s.source == "user"
    assert s.description == "user"


def test_non_directory_entries_skipped(tmp_path: Path) -> None:
    """Stray files at the skills root must not crash the loader."""
    _write_skill(tmp_path, "good")
    stray = tmp_path / PROJECT_SKILLS_DIRNAME / "stray.txt"
    stray.write_text("not a skill", encoding="utf-8")
    reg = SkillRegistry.load(project_root=tmp_path, user_root=None)
    assert [s.name for s in reg.list_skills()] == ["good"]


def test_directory_without_skill_md_skipped(tmp_path: Path) -> None:
    """A subdir lacking SKILL.md is silently ignored."""
    (tmp_path / PROJECT_SKILLS_DIRNAME / "empty").mkdir(parents=True)
    _write_skill(tmp_path, "good")
    reg = SkillRegistry.load(project_root=tmp_path, user_root=None)
    assert [s.name for s in reg.list_skills()] == ["good"]


# ---------------------------------------------------------------------------
# Frontmatter parsing — error paths
# ---------------------------------------------------------------------------


def test_missing_frontmatter_raises(tmp_path: Path) -> None:
    skill_dir = tmp_path / PROJECT_SKILLS_DIRNAME / "nofront"
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILENAME).write_text("no frontmatter here", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing a YAML frontmatter"):
        SkillRegistry.load(project_root=tmp_path, user_root=None)


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    skill_dir = tmp_path / PROJECT_SKILLS_DIRNAME / "broken"
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILENAME).write_text(
        "---\nname: broken\ndescription: [unclosed\n---\nbody",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="malformed YAML"):
        SkillRegistry.load(project_root=tmp_path, user_root=None)


def test_missing_description_raises(tmp_path: Path) -> None:
    skill_dir = tmp_path / PROJECT_SKILLS_DIRNAME / "x"
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILENAME).write_text(
        "---\nname: x\n---\nbody",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="`description` is required"):
        SkillRegistry.load(project_root=tmp_path, user_root=None)


def test_dir_name_mismatch_raises(tmp_path: Path) -> None:
    _write_skill(tmp_path, "right-dir", frontmatter_name="WRONG_NAME")
    with pytest.raises(ConfigError, match="does not match its directory name"):
        SkillRegistry.load(project_root=tmp_path, user_root=None)


def test_invalid_name_in_frontmatter_raises(tmp_path: Path) -> None:
    skill_dir = tmp_path / PROJECT_SKILLS_DIRNAME / "bad name"
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILENAME).write_text(
        "---\nname: bad name\ndescription: x\n---\nbody",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="`name` must match"):
        SkillRegistry.load(project_root=tmp_path, user_root=None)


def test_allowed_tools_must_be_string_list(tmp_path: Path) -> None:
    skill_dir = tmp_path / PROJECT_SKILLS_DIRNAME / "x"
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILENAME).write_text(
        "---\nname: x\ndescription: y\nallowed_tools: [1, 2]\n---\nbody",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="`allowed_tools` must be a list of strings"):
        SkillRegistry.load(project_root=tmp_path, user_root=None)


def test_when_to_use_must_be_string(tmp_path: Path) -> None:
    skill_dir = tmp_path / PROJECT_SKILLS_DIRNAME / "x"
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILENAME).write_text(
        "---\nname: x\ndescription: y\nwhen_to_use: [a, b]\n---\nbody",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="`when_to_use` must be a string"):
        SkillRegistry.load(project_root=tmp_path, user_root=None)


def test_frontmatter_mapping_required(tmp_path: Path) -> None:
    skill_dir = tmp_path / PROJECT_SKILLS_DIRNAME / "x"
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILENAME).write_text(
        "---\n- just\n- a list\n---\nbody",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="must be a mapping"):
        SkillRegistry.load(project_root=tmp_path, user_root=None)


# ---------------------------------------------------------------------------
# Registry API
# ---------------------------------------------------------------------------


def test_get_unknown_raises_tool_error(tmp_path: Path) -> None:
    reg = SkillRegistry.load(project_root=tmp_path, user_root=None)
    with pytest.raises(ToolError, match="no skill named"):
        reg.get("ghost")


def test_manifest_strips_body_and_allowed_tools(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "demo",
        description="A demo skill",
        when_to_use="When demoing",
        allowed_tools=["Read", "Grep"],
        body="lots of body text",
    )
    reg = SkillRegistry.load(project_root=tmp_path, user_root=None)
    [m] = reg.manifest()
    assert m == {
        "name": "demo",
        "description": "A demo skill",
        "when_to_use": "When demoing",
    }


def test_allowed_tools_round_trip(tmp_path: Path) -> None:
    _write_skill(tmp_path, "demo", allowed_tools=["Read", "Grep"])
    reg = SkillRegistry.load(project_root=tmp_path, user_root=None)
    assert reg.get("demo").allowed_tools == ("Read", "Grep")


# ---------------------------------------------------------------------------
# Built-in Skill tool
# ---------------------------------------------------------------------------


async def test_build_skill_tool_returns_body(tmp_path: Path) -> None:
    _write_skill(tmp_path, "demo", body="THE BODY")
    reg = SkillRegistry.load(project_root=tmp_path, user_root=None)
    tool_ = build_skill_tool(reg)
    assert isinstance(tool_, Tool)
    assert tool_.name == "Skill"
    body = await tool_.fn(skill_name="demo")
    assert body == "THE BODY"


def test_build_skill_tool_policy_flags(tmp_path: Path) -> None:
    reg = SkillRegistry.load(project_root=tmp_path, user_root=None)
    tool_ = build_skill_tool(reg)
    assert isinstance(tool_.policy, ToolPolicy)
    assert tool_.policy.is_read_only
    assert tool_.policy.always_load
    assert tool_.policy.is_concurrency_safe is True


async def test_skill_tool_raises_for_unknown_skill(tmp_path: Path) -> None:
    reg = SkillRegistry.load(project_root=tmp_path, user_root=None)
    tool_ = build_skill_tool(reg)
    with pytest.raises(ToolError, match="no skill named"):
        await tool_.fn(skill_name="ghost")


# ---------------------------------------------------------------------------
# iter_skill_paths utility
# ---------------------------------------------------------------------------


def test_iter_skill_paths_empty_inputs() -> None:
    assert iter_skill_paths([]) == []


def test_iter_skill_paths_picks_up_files(tmp_path: Path) -> None:
    a = _write_skill(tmp_path, "a")
    b = _write_skill(tmp_path, "b")
    found = iter_skill_paths([tmp_path / PROJECT_SKILLS_DIRNAME])
    assert sorted(found) == sorted([a, b])


def test_iter_skill_paths_skips_missing_root(tmp_path: Path) -> None:
    assert iter_skill_paths([tmp_path / "does-not-exist"]) == []
