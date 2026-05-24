"""Unit tests for :mod:`agent_harness.extras.checkpoints`.

Tests run against the real ``git`` binary. ``shutil.which("git")`` is the
gate: if git isn't available in the test environment the tracker tests are
skipped (a real concern in some CI sandboxes).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_harness.extras.checkpoints import Checkpoint, CheckpointTracker

# ---------------------------------------------------------------------------
# Skip the whole module if git is not available
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git binary not available in this environment"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Iterator[Path]:
    """A freshly initialized git repository in ``tmp_path``."""

    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    # Set local identity so commits don't need a global config.
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "t@test.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "tester"],
        check=True,
        capture_output=True,
    )
    # Create an initial commit so HEAD exists.
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "README.md"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "initial"],
        check=True,
        capture_output=True,
    )
    yield tmp_path


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def test_tracker_inactive_outside_git_repo(tmp_path: Path) -> None:
    """``tmp_path`` is a plain directory, not a git repo → no-op tracker."""

    tracker = CheckpointTracker.create(str(tmp_path), run_id="r1")
    assert tracker.is_active() is False
    # All methods are no-ops.
    assert tracker.commit("label") is None
    assert tracker.revert("anyref") is False
    assert tracker.list_checkpoints() == []


def test_tracker_inactive_for_nonexistent_path() -> None:
    tracker = CheckpointTracker.create("/this/path/does/not/exist", run_id="r1")
    assert tracker.is_active() is False


def test_tracker_active_inside_git_repo(git_repo: Path) -> None:
    tracker = CheckpointTracker.create(str(git_repo), run_id="r1")
    assert tracker.is_active() is True
    # Shadow git-dir was created.
    shadow = git_repo / ".agent_harness" / "checkpoints" / "r1" / ".git"
    assert shadow.is_dir()
    assert (shadow / "info" / "exclude").exists()


# ---------------------------------------------------------------------------
# commit + revert flow
# ---------------------------------------------------------------------------


def test_commit_creates_checkpoint(git_repo: Path) -> None:
    tracker = CheckpointTracker.create(str(git_repo), run_id="r1")

    (git_repo / "new.txt").write_text("v1", encoding="utf-8")
    cp = tracker.commit("after_batch_1")
    assert cp is not None
    assert isinstance(cp, Checkpoint)
    assert cp.label == "after_batch_1"
    assert len(cp.sha) == 40
    assert tracker.list_checkpoints() == [cp]


def test_multiple_commits_track_history(git_repo: Path) -> None:
    tracker = CheckpointTracker.create(str(git_repo), run_id="r1")

    (git_repo / "a.txt").write_text("1", encoding="utf-8")
    cp1 = tracker.commit("batch_1")

    (git_repo / "a.txt").write_text("2", encoding="utf-8")
    cp2 = tracker.commit("batch_2")

    assert cp1 is not None
    assert cp2 is not None
    assert cp1.sha != cp2.sha
    assert tracker.list_checkpoints() == [cp1, cp2]


def test_revert_to_checkpoint_restores_file_contents(git_repo: Path) -> None:
    tracker = CheckpointTracker.create(str(git_repo), run_id="r1")

    target = git_repo / "data.txt"
    target.write_text("v1", encoding="utf-8")
    cp1 = tracker.commit("v1")
    assert cp1 is not None

    target.write_text("v2", encoding="utf-8")
    cp2 = tracker.commit("v2")
    assert cp2 is not None
    assert target.read_text(encoding="utf-8") == "v2"

    ok = tracker.revert(cp1)
    assert ok is True
    assert target.read_text(encoding="utf-8") == "v1"


def test_revert_accepts_raw_sha(git_repo: Path) -> None:
    tracker = CheckpointTracker.create(str(git_repo), run_id="r1")

    target = git_repo / "data.txt"
    target.write_text("v1", encoding="utf-8")
    cp1 = tracker.commit("v1")
    assert cp1 is not None

    target.write_text("v2", encoding="utf-8")
    tracker.commit("v2")

    # Pass the raw SHA string, not the Checkpoint dataclass.
    ok = tracker.revert(cp1.sha)
    assert ok is True
    assert target.read_text(encoding="utf-8") == "v1"


def test_revert_to_invalid_ref_returns_false(git_repo: Path) -> None:
    tracker = CheckpointTracker.create(str(git_repo), run_id="r1")
    ok = tracker.revert("not-a-real-ref")
    assert ok is False


def test_list_checkpoints_returns_defensive_copy(git_repo: Path) -> None:
    tracker = CheckpointTracker.create(str(git_repo), run_id="r1")
    (git_repo / "x.txt").write_text("1", encoding="utf-8")
    tracker.commit("c")
    snapshot = tracker.list_checkpoints()
    snapshot.clear()
    assert tracker.list_checkpoints() != []


# ---------------------------------------------------------------------------
# Isolation: shadow git does not touch the real .git
# ---------------------------------------------------------------------------


def test_shadow_git_does_not_modify_real_git(git_repo: Path) -> None:
    """Cline-style: the shadow has its *own* git-dir, so commits there don't
    show up in the user's real history."""

    tracker = CheckpointTracker.create(str(git_repo), run_id="r1")

    real_log_before = subprocess.run(
        ["git", "-C", str(git_repo), "log", "--oneline"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    (git_repo / "new.txt").write_text("hi", encoding="utf-8")
    tracker.commit("shadow only")

    real_log_after = subprocess.run(
        ["git", "-C", str(git_repo), "log", "--oneline"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    # Real history is unchanged by the shadow commit.
    assert real_log_before == real_log_after


# ---------------------------------------------------------------------------
# Idempotency: re-creating a tracker for the same run reuses the shadow
# ---------------------------------------------------------------------------


def test_create_is_idempotent_for_same_run_id(git_repo: Path) -> None:
    t1 = CheckpointTracker.create(str(git_repo), run_id="r1")
    (git_repo / "a.txt").write_text("1", encoding="utf-8")
    cp = t1.commit("c1")
    assert cp is not None

    # Re-create with the same run_id; the shadow should already exist and
    # the new tracker should see HEAD pointing at cp's SHA.
    t2 = CheckpointTracker.create(str(git_repo), run_id="r1")
    assert t2.is_active() is True
    # Verify HEAD via a fresh commit chain.
    (git_repo / "a.txt").write_text("2", encoding="utf-8")
    cp2 = t2.commit("c2")
    assert cp2 is not None
    assert cp2.sha != cp.sha


# ---------------------------------------------------------------------------
# Multiple run IDs get separate shadow dirs
# ---------------------------------------------------------------------------


def test_different_run_ids_get_separate_shadows(git_repo: Path) -> None:
    t1 = CheckpointTracker.create(str(git_repo), run_id="r1")
    t2 = CheckpointTracker.create(str(git_repo), run_id="r2")

    (git_repo / "a.txt").write_text("1", encoding="utf-8")
    cp1 = t1.commit("c1")
    assert cp1 is not None
    # r2 is a fresh shadow — commits to r1 should not be visible there.
    (git_repo / "b.txt").write_text("1", encoding="utf-8")
    cp2 = t2.commit("c2")
    assert cp2 is not None

    assert (git_repo / ".agent_harness" / "checkpoints" / "r1" / ".git").is_dir()
    assert (git_repo / ".agent_harness" / "checkpoints" / "r2" / ".git").is_dir()
    assert cp1.sha != cp2.sha
