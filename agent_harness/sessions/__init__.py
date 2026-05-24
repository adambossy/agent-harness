"""Session backends (Layer 1+).

`InMemorySession` is the default; persistent backends (SQLite, Redis) land in Wave 3.
"""

from .inmemory import InMemorySession

__all__ = ["InMemorySession"]
