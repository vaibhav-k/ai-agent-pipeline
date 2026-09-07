"""Unit tests for agent_pipeline.errors.looks_like_auth_error."""

from agent_pipeline.errors import looks_like_auth_error


def test_detects_marker_in_direct_exception_message() -> None:
    assert looks_like_auth_error(Exception("Error code: 401 - invalid_api_key")) is True


def test_detects_marker_via_explicit_cause_chain() -> None:
    try:
        try:
            # Mirrors the real OpenAI error shape, which embeds 'code': 'invalid_api_key'
            # alongside the human-readable message.
            raise ValueError("Incorrect API key provided (code: invalid_api_key)")
        except ValueError as cause:
            raise RuntimeError("service failed to complete the prompt") from cause
    except RuntimeError as exc:
        assert looks_like_auth_error(exc) is True


def test_detects_marker_via_implicit_context_chain() -> None:
    try:
        try:
            raise ValueError("AuthenticationError: bad token")
        except ValueError:
            raise RuntimeError(
                "wrapped failure"
            )  # no 'from' -> __context__, not __cause__
    except RuntimeError as exc:
        assert looks_like_auth_error(exc) is True


def test_returns_false_for_unrelated_error() -> None:
    assert looks_like_auth_error(ValueError("outputs list was empty")) is False


def test_does_not_hang_on_self_referential_cause_chain() -> None:
    exc = RuntimeError("boom")
    exc.__cause__ = exc  # pathological, but must not infinite-loop
    assert looks_like_auth_error(exc) is False
