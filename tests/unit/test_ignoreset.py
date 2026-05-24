"""Unit tests for :mod:`agent_harness.extras.ignoreset`."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.extras.ignoreset import IgnoreSet, glob_matches

# ---------------------------------------------------------------------------
# from_patterns: basic semantics
# ---------------------------------------------------------------------------


def test_empty_ignoreset_matches_nothing() -> None:
    ig = IgnoreSet.from_patterns([])
    assert not ig
    assert ig.matches("anything.py") is False


def test_bare_name_matches_at_any_depth() -> None:
    ig = IgnoreSet.from_patterns(["__pycache__"])
    assert ig.matches("__pycache__") is True
    assert ig.matches("__pycache__/x.pyc") is True
    assert ig.matches("src/__pycache__/x.pyc") is True


def test_extension_glob_matches() -> None:
    ig = IgnoreSet.from_patterns(["*.log"])
    assert ig.matches("debug.log") is True
    assert ig.matches("src/debug.log") is True
    assert ig.matches("debug.txt") is False


def test_anchored_pattern_only_matches_at_root() -> None:
    ig = IgnoreSet.from_patterns(["/build"])
    assert ig.matches("build") is True
    assert ig.matches("build/x") is True
    assert ig.matches("src/build/x") is False


def test_directory_pattern_matches_contents() -> None:
    ig = IgnoreSet.from_patterns(["secrets/"])
    assert ig.matches("secrets") is True
    assert ig.matches("secrets/key.pem") is True


def test_double_star_matches_many_segments() -> None:
    ig = IgnoreSet.from_patterns(["src/**/*.tmp"])
    assert ig.matches("src/a.tmp") is True
    assert ig.matches("src/a/b/c.tmp") is True
    assert ig.matches("tests/a.tmp") is False


# ---------------------------------------------------------------------------
# Negation / ordering
# ---------------------------------------------------------------------------


def test_negation_reincludes_previously_ignored_path() -> None:
    ig = IgnoreSet.from_patterns(["*.log", "!important.log"])
    assert ig.matches("important.log") is False
    assert ig.matches("other.log") is True


def test_later_rule_can_re_exclude() -> None:
    """Rules are order-sensitive — a later positive rule overrides an earlier
    negation."""

    ig = IgnoreSet.from_patterns(["*.log", "!important.log", "important.log"])
    assert ig.matches("important.log") is True


def test_negation_first_then_exclude_is_still_excluded() -> None:
    ig = IgnoreSet.from_patterns(["!keep.log", "keep.log"])
    assert ig.matches("keep.log") is True


# ---------------------------------------------------------------------------
# Comments / blank lines
# ---------------------------------------------------------------------------


def test_blank_lines_and_comments_are_ignored() -> None:
    ig = IgnoreSet.from_patterns(["", "  ", "# comment", "*.bak"])
    assert len(ig) == 1
    assert ig.matches("x.bak") is True


# ---------------------------------------------------------------------------
# from_workspace: file discovery
# ---------------------------------------------------------------------------


def test_from_workspace_reads_agentignore(tmp_path: Path) -> None:
    (tmp_path / ".agentignore").write_text("node_modules/\n*.log\n", encoding="utf-8")
    ig = IgnoreSet.from_workspace(str(tmp_path))
    assert ig.matches("node_modules/x.js") is True
    assert ig.matches("debug.log") is True
    assert ig.matches("src/main.py") is False


def test_from_workspace_reads_multiple_conventions(tmp_path: Path) -> None:
    (tmp_path / ".clineignore").write_text("*.log\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    ig = IgnoreSet.from_workspace(str(tmp_path))
    assert ig.matches("x.log") is True
    assert ig.matches("build/main") is True


def test_from_workspace_reads_cursor_rules(tmp_path: Path) -> None:
    rules_dir = tmp_path / ".cursor" / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "01-secrets.mdc").write_text("secrets/\n", encoding="utf-8")
    (rules_dir / "02-cache.md").write_text("cache/\n", encoding="utf-8")
    ig = IgnoreSet.from_workspace(str(tmp_path))
    assert ig.matches("secrets/k") is True
    assert ig.matches("cache/x") is True


def test_from_workspace_with_no_files_is_empty(tmp_path: Path) -> None:
    ig = IgnoreSet.from_workspace(str(tmp_path))
    assert not ig
    assert ig.matches("anything") is False


def test_from_workspace_handles_negation_across_files(tmp_path: Path) -> None:
    """Patterns from later files override earlier ones."""

    (tmp_path / ".agentignore").write_text("secrets/\n", encoding="utf-8")
    (tmp_path / ".clineignore").write_text("!secrets/public.txt\n", encoding="utf-8")
    ig = IgnoreSet.from_workspace(str(tmp_path))
    assert ig.matches("secrets/key.pem") is True
    assert ig.matches("secrets/public.txt") is False


# ---------------------------------------------------------------------------
# Absolute paths + workspace root
# ---------------------------------------------------------------------------


def test_absolute_path_inside_root_is_normalized(tmp_path: Path) -> None:
    (tmp_path / ".agentignore").write_text("*.log\n", encoding="utf-8")
    ig = IgnoreSet.from_workspace(str(tmp_path))
    inside = tmp_path / "subdir" / "x.log"
    assert ig.matches(str(inside)) is True


def test_absolute_path_outside_root_falls_back_to_basename(tmp_path: Path) -> None:
    """A path outside the workspace can still match by its basename."""

    (tmp_path / ".agentignore").write_text("secret.key\n", encoding="utf-8")
    ig = IgnoreSet.from_workspace(str(tmp_path))
    # Use a path that we know is outside tmp_path.
    other = Path("/tmp/secret.key").as_posix()
    assert ig.matches(other) is True


# ---------------------------------------------------------------------------
# Encoding / file IO error path
# ---------------------------------------------------------------------------


def test_unreadable_file_does_not_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A read error on an ignore file is swallowed so a stray permissions
    issue never breaks the loop."""

    (tmp_path / ".agentignore").write_text("*.log\n", encoding="utf-8")

    orig_read_text = Path.read_text

    def boom(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == ".agentignore":
            raise OSError("permission denied")
        return orig_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", boom)
    ig = IgnoreSet.from_workspace(str(tmp_path))
    assert not ig


# ---------------------------------------------------------------------------
# Bonus: glob_matches stand-alone
# ---------------------------------------------------------------------------


def test_glob_matches_simple() -> None:
    assert glob_matches("*.py", "a.py") is True
    assert glob_matches("*.py", "a.txt") is False


def test_glob_matches_double_star() -> None:
    assert glob_matches("src/**/*.py", "src/a/b/c.py") is True
    assert glob_matches("src/**/*.py", "tests/a.py") is False


def test_glob_matches_blank_pattern_returns_false() -> None:
    assert glob_matches("", "anything") is False
    assert glob_matches("   ", "anything") is False
