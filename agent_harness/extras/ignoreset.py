"""Workspace ignore-set with ``.gitignore``-style semantics + ``!include`` rules.

This module owns one optional component (FT4 / lineage: Cline's
``.clineignore``): a fast, in-memory set of glob patterns that gates which
paths the filesystem tools will touch. Compared to plain ``.gitignore``, the
extension is straightforward — a rule prefixed with ``!`` re-includes a path
that an earlier pattern would have excluded. The class also cross-reads the
neighboring conventions agents already use in the wild (``AGENTS.md``,
``CLAUDE.md``, ``.cursor/rules/``, ``.windsurfrules``) so a workspace doesn't
need a second ignore file for the agent harness.

The implementation is deliberately self-contained — no ``pathspec``
dependency — because the pattern set is small (~tens of lines) and the
matching is hot. Patterns are compiled to regular expressions once at
construction time and cached.

Example:
    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as td:
    ...     root = pathlib.Path(td)
    ...     _ = (root / ".agentignore").write_text("secrets/\\n!secrets/public.txt\\n")
    ...     ig = IgnoreSet.from_workspace(str(root))
    ...     ig.matches("secrets/key.pem"), ig.matches("secrets/public.txt")
    (True, False)
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

# Filenames + directories that we cross-read for compatibility with neighbouring
# agent conventions. Each contributes its own pattern set.
_CROSSREAD_FILES: tuple[str, ...] = (
    ".agentignore",
    ".clineignore",
    ".aiderignore",
    ".gitignore",
    ".windsurfrules",
)
"""Ignore files read directly. Order is significant — later files can re-include
paths that earlier files excluded."""

_CROSSREAD_MARKER_FILES: tuple[str, ...] = (
    "AGENTS.md",
    "CLAUDE.md",
)
"""Marker files. Their presence does not itself produce ignore patterns, but
the workspace root is identified by them when constructing relative paths."""

_CROSSREAD_RULE_DIRS: tuple[str, ...] = (".cursor/rules",)
"""Directories whose ``*.mdc`` / ``*.md`` files are read line-by-line as a single
flat ruleset. Cursor stores per-project rules here."""


# ---------------------------------------------------------------------------
# Internal: a single compiled pattern
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Rule:
    """One compiled ignore rule.

    ``negate`` is True for ``!include`` patterns; ``directory_only`` is True
    for patterns that end in ``/`` (match directories only). The compiled
    regex matches against POSIX-style relative paths.
    """

    pattern: str
    regex: re.Pattern[str]
    negate: bool
    directory_only: bool


def _compile_pattern(raw: str) -> _Rule | None:
    """Compile one line from an ignore file.

    Returns ``None`` for blank lines / comments. Mirrors the subset of
    ``.gitignore`` semantics we need: ``!`` for negation, trailing ``/`` for
    directory-only, leading ``/`` anchors to the workspace root, and the
    standard glob characters (``*``, ``?``, ``**``).
    """

    line = raw.rstrip("\n").rstrip("\r").strip()
    if not line or line.startswith("#"):
        return None

    negate = line.startswith("!")
    if negate:
        line = line[1:]

    directory_only = line.endswith("/")
    if directory_only:
        line = line[:-1]

    anchored = line.startswith("/")
    if anchored:
        line = line[1:]

    # Translate the gitignore-ish glob to a regex.
    # ``fnmatch.translate`` gives us ``*``/``?`` handling; we then post-process
    # to support ``**`` (any number of path segments) and the anchoring rules.
    # Use placeholders for the multi-segment forms so the subsequent ``*`` ->
    # ``[^/]*`` rewrite doesn't corrupt them.
    slash_doublestar_slash = "\x00SDS\x00"
    doublestar = "\x00DS\x00"
    # ``/**/`` collapses to "/ followed by any number of segments" — including
    # zero segments — so ``a/**/b`` matches both ``a/b`` and ``a/x/y/b``.
    pat = line.replace("/**/", slash_doublestar_slash)
    pat = pat.replace("**", doublestar)

    # Per-segment glob: ``*`` matches anything except ``/``.
    regex_body = re.escape(pat)
    # Unescape the meta-characters we want to interpret as globs.
    regex_body = regex_body.replace(re.escape("*"), "[^/]*")
    regex_body = regex_body.replace(re.escape("?"), "[^/]")
    regex_body = regex_body.replace(re.escape(slash_doublestar_slash), "(?:/.*/|/)")
    regex_body = regex_body.replace(re.escape(doublestar), ".*")

    if anchored or "/" in line:
        # Pattern is anchored to the workspace root.
        full = rf"\A{regex_body}(?:/.*)?\Z"
    else:
        # Bare name — matches at any depth.
        full = rf"(?:\A|.*/){regex_body}(?:/.*)?\Z"

    return _Rule(
        pattern=raw.strip(),
        regex=re.compile(full),
        negate=negate,
        directory_only=directory_only,
    )


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class IgnoreSet:
    """Compiled, in-memory set of workspace ignore rules.

    Construct one with :meth:`from_workspace` to auto-discover the
    conventional files in a workspace root, or with :meth:`from_patterns`
    when you have an explicit list (tests, custom callers).

    Matching is order-sensitive: later rules override earlier ones, exactly
    like ``.gitignore``. A negated rule (``!path``) re-includes a path that
    an earlier rule excluded. Bare names match at any depth; a leading ``/``
    or any embedded ``/`` anchors the pattern to the workspace root.

    Example:
        >>> ig = IgnoreSet.from_patterns(["*.log", "!important.log"])
        >>> ig.matches("debug.log"), ig.matches("important.log")
        (True, False)
    """

    __slots__ = ("_root", "_rules")

    def __init__(self, rules: list[_Rule], root: Path | None = None) -> None:
        self._rules = rules
        self._root = root

    # -- Construction ------------------------------------------------------

    @classmethod
    def from_patterns(cls, patterns: list[str], *, root: str | None = None) -> IgnoreSet:
        """Build an IgnoreSet directly from a list of pattern strings.

        Example:
            >>> ig = IgnoreSet.from_patterns(["__pycache__/", "*.pyc"])
            >>> ig.matches("__pycache__/m.pyc"), ig.matches("src/m.pyc")
            (True, True)
        """

        rules: list[_Rule] = []
        for raw in patterns:
            compiled = _compile_pattern(raw)
            if compiled is not None:
                rules.append(compiled)
        return cls(rules, Path(root) if root is not None else None)

    @classmethod
    def from_workspace(cls, root: str) -> IgnoreSet:
        """Auto-discover ignore conventions in a workspace root.

        Reads (in order, so later files override earlier ones):

        1. ``.agentignore``
        2. ``.clineignore``
        3. ``.aiderignore``
        4. ``.gitignore``
        5. ``.windsurfrules``
        6. Every ``*.mdc`` / ``*.md`` file under ``.cursor/rules/``

        ``AGENTS.md`` and ``CLAUDE.md`` are *recognized* as conventional
        marker files but do not themselves contribute patterns — they
        describe agents, not ignore lists. If none of the conventional files
        exist, the IgnoreSet is empty (``matches`` is always False).

        Example:
            >>> import tempfile, pathlib
            >>> with tempfile.TemporaryDirectory() as td:
            ...     _ = pathlib.Path(td, ".agentignore").write_text("node_modules/\\n")
            ...     ig = IgnoreSet.from_workspace(td)
            ...     ig.matches("node_modules/lib/x.js")
            True
        """

        root_path = Path(root)
        rules: list[_Rule] = []

        for filename in _CROSSREAD_FILES:
            target = root_path / filename
            if target.is_file():
                rules.extend(_read_pattern_file(target))

        cursor_dir = root_path / ".cursor" / "rules"
        if cursor_dir.is_dir():
            for entry in sorted(cursor_dir.iterdir()):
                if entry.is_file() and entry.suffix in {".md", ".mdc"}:
                    rules.extend(_read_pattern_file(entry))

        return cls(rules, root_path)

    # -- Matching ----------------------------------------------------------

    def matches(self, path: str) -> bool:
        """Return True iff ``path`` should be ignored.

        ``path`` is treated as a POSIX-style relative path. If a workspace
        root was supplied at construction time and ``path`` is absolute and
        underneath that root, it is normalized to a relative path first.

        Example:
            >>> ig = IgnoreSet.from_patterns(["*.tmp", "!keep.tmp"])
            >>> ig.matches("a/b/scratch.tmp"), ig.matches("a/b/keep.tmp")
            (True, False)
        """

        normalized = self._normalize(path)
        ignored = False
        for rule in self._rules:
            if rule.regex.match(normalized):
                ignored = not rule.negate
        return ignored

    def __len__(self) -> int:
        return len(self._rules)

    def __bool__(self) -> bool:
        return bool(self._rules)

    # -- Internal helpers --------------------------------------------------

    def _normalize(self, path: str) -> str:
        """POSIX-style relative path for matching.

        Strips a leading ``./`` and (when the workspace root is known)
        rewrites absolute paths into root-relative ones.
        """

        p = Path(path)
        if self._root is not None and p.is_absolute():
            try:
                p = p.relative_to(self._root)
            except ValueError:
                # Path is absolute but not under our root — fall back to
                # using its name; the rules can still match by basename.
                p = Path(p.name)
        s = p.as_posix()
        if s.startswith("./"):
            s = s[2:]
        return s


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------


def _read_pattern_file(target: Path) -> list[_Rule]:
    """Read a pattern file from disk, compiling each non-blank line.

    Encoding errors are swallowed (replaced with U+FFFD) — we never want a
    bad byte in a stray ignore file to crash the agent.
    """

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: list[_Rule] = []
    for raw in text.splitlines():
        compiled = _compile_pattern(raw)
        if compiled is not None:
            out.append(compiled)
    return out


# ---------------------------------------------------------------------------
# Convenience: also expose a stand-alone matcher for callers that want to
# test a single glob without going through the IgnoreSet machinery.
# ---------------------------------------------------------------------------


def glob_matches(pattern: str, path: str) -> bool:
    """Lightweight stand-alone glob match against a POSIX path.

    Mirrors ``fnmatch.fnmatchcase`` but treats ``**`` as "any number of
    segments" the way ``.gitignore`` does. Useful for ad-hoc allow/deny
    checks outside an IgnoreSet.

    Example:
        >>> glob_matches("src/**/*.py", "src/a/b/c.py")
        True
        >>> glob_matches("*.py", "src/a.py")
        False
    """

    rule = _compile_pattern(pattern)
    if rule is None:
        return False
    # When the caller uses ``**`` patterns we want full-path matching;
    # otherwise fnmatch is enough for short-circuiting.
    if "**" in pattern or "/" in pattern:
        return bool(rule.regex.match(path))
    return fnmatch.fnmatchcase(path, pattern)
