"""Mention parser with verb-specific fetchers (Cline pattern).

Cline introduced an ``@<thing>`` syntax in user prompts that the harness
expands into structured context snippets before the LLM ever sees the
message. Supported verbs:

* ``@file:<path>`` — inline the contents of a workspace file
* ``@http://…`` / ``@https://…`` — fetch a URL and inline the result
* ``@problems`` — current diagnostic problems (linter / type-checker)
* ``@terminal`` — recent terminal output
* ``@git-changes`` — current ``git status`` / ``git diff`` summary
* ``@<sha>`` — a specific git commit

For the v0 cut, only **parsing** is fully functional; fetchers are
intentionally stubs so callers can swap real implementations in. The pattern
is a small registry keyed by verb — replacing the stubs is one assignment
each, not a redesign.

Example:
    >>> mentions = parse_mentions("Look at @file:src/a.py and @git-changes please")
    >>> [m.verb for m in mentions]
    ['file', 'git-changes']
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

# A mention starts with ``@`` and ends at the first whitespace character.
# We deliberately let the verb capture the body too; the parser splits on
# the first ``:`` (when present) to separate verb from argument.
_MENTION_RE: Final[re.Pattern[str]] = re.compile(r"(?<!\w)@([^\s]+)")
"""Match any @token preceded by a non-word boundary. Excludes email-style
addresses where ``@`` is preceded by a word character (e.g. ``user@host``)."""


# Known verbs that take no argument (the verb *is* the entire token).
_BARE_VERBS: Final[frozenset[str]] = frozenset({"problems", "terminal", "git-changes"})


@dataclass(frozen=True, slots=True)
class Mention:
    """One parsed mention.

    ``verb`` is the canonical name (``file``, ``url``, ``problems``, ``sha``,
    …). ``argument`` is the payload — a path for ``file``, a URL for ``url``,
    a short git SHA for ``sha``, or ``None`` for the bare verbs. ``raw`` is
    the original substring, including the leading ``@``, so callers can
    splice the resolved content back into the prompt.

    Example:
        >>> Mention(verb="file", argument="src/a.py", raw="@file:src/a.py").verb
        'file'
    """

    verb: str
    argument: str | None
    raw: str


@dataclass(frozen=True, slots=True)
class ResolvedMention:
    """A mention plus the fetched content snippet.

    ``content`` is what the fetcher returned (markdown text by convention).
    ``error`` is non-None if the fetcher failed; in that case ``content`` is
    a short placeholder describing the failure so the model still sees
    *something*.

    ``is_stub`` is True when the resolution came from one of the built-in
    placeholder fetchers. Callers wiring real integrations (the loop, an
    operator UI) should treat ``is_stub=True`` as "this content is a
    development placeholder — do NOT route it to the model as real
    context"; the loop is expected to either skip the snippet or surface
    a UX warning. The content is still self-describing
    (``[stub: @verb not yet wired]``) for cases where a placeholder is
    acceptable.

    Example:
        >>> r = ResolvedMention(
        ...     mention=Mention(verb="file", argument="x", raw="@file:x"),
        ...     content="hello",
        ...     error=None,
        ... )
        >>> r.content, r.is_stub
        ('hello', False)
    """

    mention: Mention
    content: str
    error: str | None = None
    is_stub: bool = False


# A fetcher is an async-or-sync callable that takes the parsed Mention and
# returns the snippet text. We keep them sync for the v0 stub registry; the
# real implementations can be async (wrap with ``asyncio.to_thread`` etc).
Fetcher = Callable[[Mention], str]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_mentions(text: str) -> list[Mention]:
    """Extract every mention from ``text``, in order of occurrence.

    Tokens that look like mentions but are obviously something else (a bare
    ``@`` with no body, an email address embedded mid-word) are skipped.

    Example:
        >>> [m.verb for m in parse_mentions("see @file:a.py @problems and @nothing-real")]
        ['file', 'problems', 'nothing-real']
    """

    out: list[Mention] = []
    for match in _MENTION_RE.finditer(text):
        body = match.group(1)
        raw = match.group(0)
        verb, argument = _split_verb_arg(body)
        if verb is None:
            continue
        out.append(Mention(verb=verb, argument=argument, raw=raw))
    return out


def _split_verb_arg(body: str) -> tuple[str | None, str | None]:
    """Split the body of a mention into (verb, argument).

    Rules:

    * Empty body → ``(None, None)`` (caller skips).
    * ``verb:rest`` → ``(verb, rest)``.
    * URL-shaped body (``http://`` / ``https://``) → ``("url", body)``.
    * 7-40 hex chars → ``("sha", body)``.
    * One of the bare verbs → ``(verb, None)``.
    * Anything else → ``(body, None)`` so unknown verbs still parse cleanly
      and the registry can return a "not supported" error snippet.
    """

    if not body:
        return None, None

    # Explicit ``verb:argument`` form takes precedence.
    if ":" in body:
        verb, _, argument = body.partition(":")
        if verb:
            # URLs come through as ``http:`` / ``https:`` → normalize.
            if verb in {"http", "https"}:
                return "url", body
            return verb, argument or None

    if body in _BARE_VERBS:
        return body, None

    # Bare hex token of 7-40 chars → treat as a git sha.
    if 7 <= len(body) <= 40 and all(c in "0123456789abcdef" for c in body.lower()):
        return "sha", body

    # Unknown verb — keep it so callers can decide how to surface the error.
    return body, None


# ---------------------------------------------------------------------------
# Fetcher registry
# ---------------------------------------------------------------------------


class MentionResolver:
    """Pluggable verb→fetcher registry.

    Construct with the defaults (which are clearly-labeled stubs) and
    register real fetchers for the verbs your runtime supports. The stubs
    return a self-describing placeholder and set ``ResolvedMention.is_stub
    = True`` so callers can detect "no real fetcher registered" — the loop
    is expected to either skip stub content or surface a UX warning rather
    than treating it as authentic file / URL / git context.

    Real fetchers MUST be registered by the loop integration (Wave 4) for
    each verb the deployment supports.

    Example:
        >>> resolver = MentionResolver()
        >>> resolver.register("file", lambda m: f"<contents of {m.argument}>")
        >>> resolved = resolver.resolve_all("see @file:notes.md")
        >>> resolved[0].content, resolved[0].is_stub
        ('<contents of notes.md>', False)
    """

    __slots__ = ("_fetchers", "_stub_verbs")

    # Verbs for which the resolver ships a development placeholder. A real
    # fetcher registered via :meth:`register` removes the verb from the
    # stub set, so ``is_stub`` becomes False on subsequent resolutions.
    _DEFAULT_STUB_VERBS: tuple[str, ...] = (
        "file",
        "url",
        "problems",
        "terminal",
        "git-changes",
        "sha",
    )

    def __init__(self) -> None:
        self._fetchers: dict[str, Fetcher] = {}
        self._stub_verbs: set[str] = set()
        # Register clearly-labelled stubs so callers get a useful placeholder
        # in development; ``is_stub=True`` on the returned ResolvedMention
        # tells the loop to skip / warn rather than treat it as real context.
        for verb in self._DEFAULT_STUB_VERBS:
            self._fetchers[verb] = _stub_fetcher(verb)
            self._stub_verbs.add(verb)

    def register(self, verb: str, fetcher: Fetcher) -> None:
        """Register / replace a fetcher for ``verb``.

        Calling :meth:`register` with a real fetcher removes the verb from
        the stub set, so subsequent ``ResolvedMention``s carry
        ``is_stub=False``.

        Example:
            >>> MentionResolver().register("file", lambda m: "ok")  # no error
        """

        self._fetchers[verb] = fetcher
        self._stub_verbs.discard(verb)

    def resolve(self, mention: Mention) -> ResolvedMention:
        """Fetch the snippet for a single mention.

        Errors raised by the fetcher are caught and surfaced as a
        ``ResolvedMention`` with non-None ``error``; the loop must keep
        running even if one mention fails.
        """

        fetcher = self._fetchers.get(mention.verb)
        if fetcher is None:
            return ResolvedMention(
                mention=mention,
                content=f"[mention @{mention.verb}: unsupported]",
                error=f"unsupported verb: {mention.verb}",
                is_stub=False,
            )
        try:
            content = fetcher(mention)
        except Exception as exc:
            return ResolvedMention(
                mention=mention,
                content=f"[mention {mention.raw}: error]",
                error=str(exc),
                is_stub=False,
            )
        return ResolvedMention(
            mention=mention,
            content=content,
            error=None,
            is_stub=mention.verb in self._stub_verbs,
        )

    def resolve_all(self, text: str) -> list[ResolvedMention]:
        """Parse ``text`` and resolve every mention found.

        Example:
            >>> MentionResolver().resolve_all("nothing here")
            []
        """

        return [self.resolve(m) for m in parse_mentions(text)]


def _stub_fetcher(verb: str) -> Fetcher:
    """Build a placeholder fetcher for a verb (v0).

    The returned string is self-describing — it begins with ``[stub: @<verb>``
    and says "not yet wired" — so even if a caller forgets to check
    :attr:`ResolvedMention.is_stub`, the model sees an obviously-placeholder
    snippet rather than a confidently-wrong "real" result.
    """

    def _fetch(mention: Mention) -> str:
        target = f"@{verb}:{mention.argument}" if mention.argument else f"@{verb}"
        return f"[stub: {target} not yet wired — register a real fetcher]"

    return _fetch
