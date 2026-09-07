"""Diagnostic helpers for turning raw exceptions into actionable guidance."""

from __future__ import annotations

from . import constants


def looks_like_auth_error(exc: BaseException) -> bool:
    """
    Heuristic check for an API-key/authentication failure anywhere in the exception chain.

    Different providers (OpenAI, Azure OpenAI) and SDK versions wrap auth
    failures in different exception types, so this checks for well-known
    markers in the stringified exception chain rather than matching specific
    classes.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__} {current}".lower()
        if any(marker in text for marker in constants.AUTH_ERROR_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False
