from __future__ import annotations


class ProviderError(RuntimeError):
    """Base error for AI provider failures."""


class TemporaryProviderError(ProviderError):
    """Raised when a provider reports a transient outage or overload."""

