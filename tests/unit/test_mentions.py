"""Unit tests for :mod:`agent_harness.extras.mentions`."""

from __future__ import annotations

from agent_harness.extras.mentions import (
    Mention,
    MentionResolver,
    ResolvedMention,
    parse_mentions,
)

# ---------------------------------------------------------------------------
# parse_mentions: verb-specific parsing
# ---------------------------------------------------------------------------


def test_parse_mentions_empty_string_returns_empty_list() -> None:
    assert parse_mentions("") == []


def test_parse_mentions_no_mentions_in_text() -> None:
    assert parse_mentions("just a regular sentence with no mentions") == []


def test_parse_file_mention() -> None:
    mentions = parse_mentions("Look at @file:src/main.py please")
    assert len(mentions) == 1
    assert mentions[0].verb == "file"
    assert mentions[0].argument == "src/main.py"
    assert mentions[0].raw == "@file:src/main.py"


def test_parse_url_mention_http() -> None:
    mentions = parse_mentions("see @http://example.com/page for details")
    assert len(mentions) == 1
    assert mentions[0].verb == "url"
    assert mentions[0].argument == "http://example.com/page"


def test_parse_url_mention_https() -> None:
    mentions = parse_mentions("see @https://example.com")
    assert mentions[0].verb == "url"
    assert mentions[0].argument == "https://example.com"


def test_parse_bare_problems() -> None:
    mentions = parse_mentions("Fix @problems now")
    assert len(mentions) == 1
    assert mentions[0].verb == "problems"
    assert mentions[0].argument is None


def test_parse_bare_terminal() -> None:
    mentions = parse_mentions("@terminal")
    assert mentions[0].verb == "terminal"
    assert mentions[0].argument is None


def test_parse_bare_git_changes() -> None:
    mentions = parse_mentions("review @git-changes")
    assert mentions[0].verb == "git-changes"
    assert mentions[0].argument is None


def test_parse_sha_short() -> None:
    mentions = parse_mentions("check commit @abc1234 first")
    assert mentions[0].verb == "sha"
    assert mentions[0].argument == "abc1234"


def test_parse_sha_long() -> None:
    full = "a" * 40
    mentions = parse_mentions(f"see @{full}")
    assert mentions[0].verb == "sha"
    assert mentions[0].argument == full


def test_parse_multiple_mentions_in_order() -> None:
    mentions = parse_mentions("a @file:x.py b @problems c @git-changes")
    verbs = [m.verb for m in mentions]
    assert verbs == ["file", "problems", "git-changes"]


def test_parse_ignores_email_addresses() -> None:
    """``user@host`` is not a mention — the ``@`` is preceded by a word
    character."""

    mentions = parse_mentions("contact alice@example.com about it")
    assert mentions == []


def test_parse_unknown_verb_still_returns_mention() -> None:
    """Unknown verbs parse cleanly so the resolver can return an error
    snippet rather than the harness raising."""

    mentions = parse_mentions("@unknown-thing here")
    assert len(mentions) == 1
    assert mentions[0].verb == "unknown-thing"
    assert mentions[0].argument is None


def test_parse_file_with_path_containing_special_chars() -> None:
    mentions = parse_mentions("@file:src/utils/helpers.py")
    assert mentions[0].argument == "src/utils/helpers.py"


def test_parse_mention_at_start_of_string() -> None:
    mentions = parse_mentions("@file:a.py is interesting")
    assert mentions[0].verb == "file"


def test_parse_mention_at_end_of_string() -> None:
    mentions = parse_mentions("look at @file:a.py")
    assert mentions[0].verb == "file"


# ---------------------------------------------------------------------------
# MentionResolver: stubs + registration
# ---------------------------------------------------------------------------


def test_resolver_with_stubs_returns_placeholder() -> None:
    resolver = MentionResolver()
    mention = Mention(verb="file", argument="src/a.py", raw="@file:src/a.py")
    resolved = resolver.resolve(mention)
    assert resolved.error is None
    assert "src/a.py" in resolved.content
    assert "stub" in resolved.content.lower()
    # is_stub=True signals the loop that this is a placeholder, not real
    # content — callers must skip or warn rather than route to the model.
    assert resolved.is_stub is True


def test_resolver_stub_content_is_self_describing() -> None:
    """Stub content must obviously declare itself as not-yet-wired so a model
    that sees it (e.g. a caller that forgot the ``is_stub`` check) gets an
    obviously-placeholder snippet rather than confidently-wrong context."""

    resolver = MentionResolver()
    for verb in ("file", "url", "problems", "terminal", "git-changes", "sha"):
        mention = Mention(verb=verb, argument=None, raw=f"@{verb}")
        resolved = resolver.resolve(mention)
        assert resolved.is_stub is True
        assert "stub" in resolved.content.lower()
        assert "not yet wired" in resolved.content.lower()


def test_resolver_register_overrides_stub() -> None:
    resolver = MentionResolver()
    resolver.register("file", lambda m: f"REAL:{m.argument}")
    mention = Mention(verb="file", argument="x.py", raw="@file:x.py")
    resolved = resolver.resolve(mention)
    assert resolved.content == "REAL:x.py"
    assert resolved.error is None
    # Registering a real fetcher must clear the stub flag.
    assert resolved.is_stub is False


def test_resolver_unsupported_verb_surfaces_error() -> None:
    """Even unknown verbs that the parser returned must resolve to a
    ``ResolvedMention`` so the loop never raises."""

    resolver = MentionResolver()
    mention = Mention(verb="totally-fake", argument=None, raw="@totally-fake")
    resolved = resolver.resolve(mention)
    assert resolved.error is not None
    assert "unsupported" in resolved.error
    assert resolved.content


def test_resolver_fetcher_exception_is_caught() -> None:
    """If a fetcher raises, the resolver should surface the error as a
    ``ResolvedMention`` with non-None ``error`` rather than propagating."""

    resolver = MentionResolver()

    def explode(_: Mention) -> str:
        raise RuntimeError("network down")

    resolver.register("file", explode)
    mention = Mention(verb="file", argument="x.py", raw="@file:x.py")
    resolved = resolver.resolve(mention)
    assert resolved.error == "network down"
    assert resolved.content.startswith("[mention @file:x.py: error]")


def test_resolver_resolve_all_parses_and_resolves() -> None:
    resolver = MentionResolver()
    resolver.register("file", lambda m: f"FILE:{m.argument}")
    resolver.register("problems", lambda m: "P")
    resolved = resolver.resolve_all("see @file:a.py and @problems")
    assert len(resolved) == 2
    assert resolved[0].content == "FILE:a.py"
    assert resolved[1].content == "P"


def test_resolver_resolve_all_on_empty_text() -> None:
    assert MentionResolver().resolve_all("nothing here") == []


def test_resolved_mention_is_frozen() -> None:
    """``ResolvedMention`` is a frozen dataclass — callers can rely on
    immutability for caching."""

    import dataclasses

    rm = ResolvedMention(
        mention=Mention(verb="file", argument="a", raw="@file:a"),
        content="ok",
        error=None,
    )
    try:
        rm.content = "tampered"  # type: ignore[misc]
    except (dataclasses.FrozenInstanceError, AttributeError):
        return
    raise AssertionError("ResolvedMention must be frozen")
