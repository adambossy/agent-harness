"""Shadow-git checkpoint tracker (Cline pattern, open-question #7).

This is an *optional* component: it activates only when (a) the ``git``
binary is on ``$PATH`` and (b) the working directory is a git repository.
When either condition fails, every method becomes a no-op so the loop runs
unchanged.

Mechanism (Cline's design):

* Under ``.agent_harness/checkpoints/<run_id>/`` we keep a **separate** git
  directory whose ``core.worktree`` points at the live tree. That gives us a
  parallel history we control without touching the user's ``.git/``.
* After each tool batch the loop calls :meth:`CheckpointTracker.commit` with
  a label. We ``git add -A`` against the shadow worktree and create a
  commit.
* To roll back, callers invoke :meth:`CheckpointTracker.revert` with a
  checkpoint identifier; we run ``git reset --hard`` against the shadow
  worktree.

The implementation uses ``subprocess.run(["git", ...])`` exclusively so we
do not pull in a git library and so tests can drive everything with the
real ``git`` binary in a ``tmp_path``.

Example:
    >>> import tempfile, pathlib, subprocess
    >>> with tempfile.TemporaryDirectory() as td:
    ...     _ = subprocess.run(["git", "init", "-q", td], check=True, capture_output=True)
    ...     _ = subprocess.run(
    ...         ["git", "-C", td, "config", "user.email", "t@t.com"],
    ...         check=True,
    ...         capture_output=True,
    ...     )
    ...     _ = subprocess.run(
    ...         ["git", "-C", td, "config", "user.name", "t"],
    ...         check=True,
    ...         capture_output=True,
    ...     )
    ...     tracker = CheckpointTracker.create(td, run_id="r1")
    ...     tracker.is_active()
    True
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A single shadow-git commit.

    ``sha`` is the full 40-char commit hash. ``label`` is the short
    human-readable string the caller passed to :meth:`CheckpointTracker.commit`.

    Example:
        >>> Checkpoint(sha="abc123", label="after_tool_batch_1").label
        'after_tool_batch_1'
    """

    sha: str
    label: str


# ---------------------------------------------------------------------------
# CheckpointTracker
# ---------------------------------------------------------------------------


class CheckpointTracker:
    """Per-run shadow-git checkpoint manager.

    Construct via :meth:`create`. The classmethod returns an *inactive*
    tracker (one whose methods are no-ops) when git is unavailable or the
    workspace is not a git repo; this lets the caller wire the tracker in
    unconditionally and rely on the activation check.

    Example:
        >>> CheckpointTracker.create("/nonexistent-path", run_id="r1").is_active()
        False
    """

    __slots__ = ("_active", "_git_dir", "_run_id", "_worktree", "checkpoints")

    def __init__(
        self,
        *,
        worktree: Path,
        git_dir: Path,
        run_id: str,
        active: bool,
    ) -> None:
        self._worktree = worktree
        self._git_dir = git_dir
        self._run_id = run_id
        self._active = active
        self.checkpoints: list[Checkpoint] = []

    # -- Construction ------------------------------------------------------

    @classmethod
    def create(cls, worktree: str, *, run_id: str) -> CheckpointTracker:
        """Build a tracker for ``worktree``, activating only when possible.

        Activation requires:

        * ``git`` on ``$PATH`` (``shutil.which("git")``).
        * ``worktree`` exists and is a git repository (``rev-parse`` succeeds).

        When inactive, every other method returns immediately. The caller
        can still call :meth:`is_active` to decide whether to surface
        checkpoint UI.
        """

        worktree_path = Path(worktree).resolve()

        if shutil.which("git") is None or not worktree_path.is_dir():
            return cls(
                worktree=worktree_path,
                git_dir=worktree_path / ".agent_harness" / "checkpoints" / run_id / ".git",
                run_id=run_id,
                active=False,
            )

        # Confirm it's actually a git repo (so we don't spawn a shadow for a
        # random folder a user happens to be running in).
        probe = subprocess.run(
            ["git", "-C", str(worktree_path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0 or probe.stdout.strip() != "true":
            return cls(
                worktree=worktree_path,
                git_dir=worktree_path / ".agent_harness" / "checkpoints" / run_id / ".git",
                run_id=run_id,
                active=False,
            )

        shadow_dir = worktree_path / ".agent_harness" / "checkpoints" / run_id / ".git"
        tracker = cls(
            worktree=worktree_path,
            git_dir=shadow_dir,
            run_id=run_id,
            active=True,
        )
        tracker._init_shadow()
        return tracker

    # -- Lifecycle ---------------------------------------------------------

    def is_active(self) -> bool:
        """Return True iff this tracker actually creates checkpoints.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as td:
            ...     CheckpointTracker.create(td, run_id="r").is_active()
            False
        """

        return self._active

    def commit(self, label: str) -> Checkpoint | None:
        """Record a checkpoint after a tool batch.

        Stages everything in the live tree (``git add -A``) and creates a
        commit in the shadow repo. Returns the :class:`Checkpoint` (or
        ``None`` if the tracker is inactive / nothing changed since the
        previous commit).
        """

        if not self._active:
            return None

        self._git("add", "-A")
        # ``commit --allow-empty`` so callers don't have to special-case
        # "no changes since last commit"; matches Cline's behaviour where
        # checkpoints map 1:1 to tool batches.
        commit_proc = self._git(
            "commit",
            "--allow-empty",
            "-m",
            label,
            check=False,
        )
        if commit_proc.returncode != 0:
            return None

        sha_proc = self._git("rev-parse", "HEAD")
        sha = sha_proc.stdout.strip()
        checkpoint = Checkpoint(sha=sha, label=label)
        self.checkpoints.append(checkpoint)
        return checkpoint

    def revert(self, checkpoint: Checkpoint | str) -> bool:
        """Roll the worktree back to ``checkpoint``.

        ``checkpoint`` may be a :class:`Checkpoint` or a raw SHA / refspec.
        Runs ``git reset --hard`` against the shadow worktree. Returns True
        on success, False if inactive / git failed.
        """

        if not self._active:
            return False

        ref = checkpoint.sha if isinstance(checkpoint, Checkpoint) else checkpoint
        proc = self._git("reset", "--hard", ref, check=False)
        return proc.returncode == 0

    def list_checkpoints(self) -> list[Checkpoint]:
        """Return a defensive copy of the checkpoint history.

        Example:
            >>> tr = CheckpointTracker(
            ...     worktree=Path("/tmp"),
            ...     git_dir=Path("/tmp/.g"),
            ...     run_id="r",
            ...     active=False,
            ... )
            >>> tr.list_checkpoints()
            []
        """

        return list(self.checkpoints)

    # -- Internals ---------------------------------------------------------

    def _init_shadow(self) -> None:
        """Create the shadow git directory if it doesn't exist yet.

        Idempotent: re-running ``create`` for the same ``run_id`` re-uses
        the existing shadow rather than wiping it.
        """

        if self._git_dir.exists():
            return

        self._git_dir.parent.mkdir(parents=True, exist_ok=True)

        # ``git init --separate-git-dir=<shadow>`` would also touch the live
        # tree (write a ``.git`` *file* pointing at the shadow), so we do it
        # by hand: bare init, then configure core.worktree.
        subprocess.run(
            ["git", "init", "--bare", "-q", str(self._git_dir)],
            check=True,
            capture_output=True,
        )

        # Point the shadow at the live worktree.
        self._git("config", "core.worktree", str(self._worktree))
        # And give it identity so commits don't fail in CI environments
        # where ``user.name`` / ``user.email`` aren't globally set.
        self._git("config", "user.name", "agent-harness")
        self._git("config", "user.email", "agent-harness@local")
        # Don't trip Apple Silicon's "dubious ownership" guard when the
        # shadow lives under the workspace.
        self._git("config", "safe.directory", str(self._worktree))

        # Ignore the shadow itself so commits don't recurse into it.
        info_dir = self._git_dir / "info"
        info_dir.mkdir(parents=True, exist_ok=True)
        (info_dir / "exclude").write_text(
            ".agent_harness/\n",
            encoding="utf-8",
        )

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a git command against the shadow git-dir + live worktree."""

        cmd = [
            "git",
            f"--git-dir={self._git_dir}",
            f"--work-tree={self._worktree}",
            *args,
        ]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
        )
