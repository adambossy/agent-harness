"""Tracing subscribers (Layer 1).

Console subscriber is fully implemented; OTEL subscriber is a skeleton
per open-questions.md decision #6.
"""

from .console import ConsoleSubscriber
from .otel import OTELSubscriber

__all__ = ["ConsoleSubscriber", "OTELSubscriber"]
