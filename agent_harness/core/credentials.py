"""Per-run credentials and their resolution — no global-key fallback.

A :data:`Credential` names the provider it authenticates and carries the
secret (an API key or an OAuth token). A run is handed one of two ways:

- a concrete :data:`Credential` (the caller already has it), or
- a :data:`CredentialResolver` — a ``() -> Credential`` thunk the harness
  calls at run time (per-user / per-tenant lookup, short-lived token minting).

:func:`resolve_credential` is the single choke point that turns
``(credential, resolver)`` into a ``Credential`` and — crucially — raises
:class:`~agent_harness.core.errors.NoCredentialError` when neither is present
rather than reaching for a process-wide environment key. Providers build their
client from the resolved credential (see :meth:`SupportsCredential`), so the
secret used for a run is always the one the caller supplied for *that* run.

Example:
    >>> cred = ApiKeyCredential(provider="anthropic", key="sk-abc")
    >>> resolve_credential(credential=cred).key
    'sk-abc'
    >>> resolve_credential(credential_resolver=lambda: cred).provider
    'anthropic'
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .errors import ConfigError, NoCredentialError, NotSupportedError

# --- Credential types --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ApiKeyCredential:
    """A provider API key. ``provider`` guards against wiring, say, an OpenAI
    key into the Anthropic client.

    Example:
        >>> ApiKeyCredential(provider="openai", key="sk-1").provider
        'openai'
    """

    provider: str
    key: str


@dataclass(frozen=True, slots=True)
class OAuthCredential:
    """An OAuth bearer token (plus optional refresh metadata) for a provider.

    Carried through the same resolution path as :class:`ApiKeyCredential`;
    API-key-only providers reject it (see :func:`api_key_from_credential`).

    Example:
        >>> OAuthCredential(provider="anthropic", access_token="tok").access_token
        'tok'
    """

    provider: str
    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None


Credential = ApiKeyCredential | OAuthCredential
"""What authenticates a run: an API key or an OAuth token."""

CredentialResolver = Callable[[], Credential]
"""A ``() -> Credential`` thunk resolved at run time (per-user lookup, etc.)."""


# --- Resolution --------------------------------------------------------------


def resolve_credential(
    *,
    credential: Credential | None = None,
    credential_resolver: CredentialResolver | None = None,
) -> Credential:
    """Resolve the credential for a run — explicit ``credential`` first, then
    ``credential_resolver()``. Raise :class:`NoCredentialError` if neither is
    supplied; there is deliberately **no** global-key fallback.

    Example:
        >>> resolve_credential()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        agent_harness.core.errors.NoCredentialError: ...
    """
    if credential is not None:
        return credential
    if credential_resolver is not None:
        return credential_resolver()
    raise NoCredentialError(
        "no credential supplied: pass a Credential or a credential_resolver "
        "(no global environment-key fallback)"
    )


def api_key_from_credential(credential: Credential, *, expected_provider: str) -> str:
    """Extract the API key from ``credential`` for an API-key provider.

    Rejects a provider mismatch (a credential minted for a different provider)
    and an :class:`OAuthCredential` (the API-key providers have no bearer-token
    path yet), so a wrong-shaped credential fails loudly at client-build time.

    Example:
        >>> api_key_from_credential(
        ...     ApiKeyCredential(provider="anthropic", key="sk-x"),
        ...     expected_provider="anthropic",
        ... )
        'sk-x'
    """
    if credential.provider != expected_provider:
        raise ConfigError(
            f"credential is for provider {credential.provider!r}, not {expected_provider!r}",
            context={"expected": expected_provider, "got": credential.provider},
        )
    if isinstance(credential, OAuthCredential):
        raise NotSupportedError(
            f"{expected_provider} provider requires an API key; "
            "OAuth credentials are not yet supported"
        )
    return credential.key


# --- Provider seam -----------------------------------------------------------


@runtime_checkable
class SupportsCredential(Protocol):
    """A provider that can (re)build its client from a :data:`Credential` at
    run time.

    A provider implements ``use_credential`` to swap in the secret an
    :class:`~agent_harness.core.agent.Agent` resolves for a run. Structural
    (``runtime_checkable``) so the agent can ``isinstance``-gate on the
    capability without providers importing core — matching the ``Provider`` /
    ``Model`` Protocol idiom.

    Example:
        >>> class _P:
        ...     def use_credential(self, credential: Credential) -> None: ...
        >>> isinstance(_P(), SupportsCredential)
        True
    """

    def use_credential(self, credential: Credential) -> None: ...
