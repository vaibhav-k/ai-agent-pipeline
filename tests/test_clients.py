"""
Unit tests for agent_pipeline.clients: provider auto-selection and validation.

No real network access or credentials are used: constructing an OpenAIChatClient
doesn't itself make a network call, so these tests only need environment
variables to be set correctly.
"""

from __future__ import annotations

import logging

import pytest
from agent_framework.openai import OpenAIChatClient

from agent_pipeline import constants
from agent_pipeline.clients import build_chat_client


def test_raises_clear_error_when_no_provider_is_configured() -> None:
    with pytest.raises(RuntimeError, match=constants.ENV_OPENAI_API_KEY):
        build_chat_client()


def test_builds_openai_client_from_valid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(constants.ENV_OPENAI_API_KEY, "sk-test-key")
    client = build_chat_client()
    assert isinstance(client, OpenAIChatClient)


def test_warns_when_openai_key_has_unexpected_format(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(constants.ENV_OPENAI_API_KEY, "not-a-real-key-format")
    with caplog.at_level(logging.WARNING):
        build_chat_client()
    assert "doesn't start with" in caplog.text


def test_does_not_warn_for_well_formed_openai_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(constants.ENV_OPENAI_API_KEY, "sk-well-formed")
    with caplog.at_level(logging.WARNING):
        build_chat_client()
    assert "doesn't start with" not in caplog.text


def test_azure_endpoint_takes_precedence_over_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(constants.ENV_OPENAI_API_KEY, "sk-should-be-ignored")
    monkeypatch.setenv(
        constants.ENV_AZURE_OPENAI_ENDPOINT, "https://example.openai.azure.com"
    )
    monkeypatch.setenv(constants.ENV_AZURE_OPENAI_CHAT_MODEL, "my-deployment")
    monkeypatch.setenv(constants.ENV_AZURE_OPENAI_API_KEY, "azure-test-key")

    client = build_chat_client()

    assert isinstance(client, OpenAIChatClient)
    assert client.azure_endpoint == "https://example.openai.azure.com"


def test_raises_clear_error_when_azure_deployment_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        constants.ENV_AZURE_OPENAI_ENDPOINT, "https://example.openai.azure.com"
    )
    monkeypatch.setenv(constants.ENV_AZURE_OPENAI_API_KEY, "azure-test-key")

    with pytest.raises(RuntimeError, match=constants.ENV_AZURE_OPENAI_CHAT_MODEL):
        build_chat_client()


def test_azure_with_key_does_not_touch_azure_ad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        constants.ENV_AZURE_OPENAI_ENDPOINT, "https://example.openai.azure.com"
    )
    monkeypatch.setenv(constants.ENV_AZURE_OPENAI_CHAT_MODEL, "my-deployment")
    monkeypatch.setenv(constants.ENV_AZURE_OPENAI_API_KEY, "azure-test-key")

    def _must_not_be_called(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "DefaultAzureCredential should not be constructed when an API key is set"
        )

    monkeypatch.setattr("azure.identity.DefaultAzureCredential", _must_not_be_called)

    client = build_chat_client()

    assert isinstance(client, OpenAIChatClient)


def test_azure_falls_back_to_azure_ad_when_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        constants.ENV_AZURE_OPENAI_ENDPOINT, "https://example.openai.azure.com"
    )
    monkeypatch.setenv(constants.ENV_AZURE_OPENAI_CHAT_MODEL, "my-deployment")

    class _FakeCredential:
        """Minimal stand-in satisfying azure.core.credentials.TokenCredential structurally."""

        def get_token(self, *scopes: str, **kwargs: object) -> object:
            raise AssertionError("token should never actually be fetched in this test")

    monkeypatch.setattr("azure.identity.DefaultAzureCredential", _FakeCredential)

    client = build_chat_client()

    assert isinstance(client, OpenAIChatClient)


def test_azure_ad_fallback_requires_azure_identity_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        constants.ENV_AZURE_OPENAI_ENDPOINT, "https://example.openai.azure.com"
    )
    monkeypatch.setenv(constants.ENV_AZURE_OPENAI_CHAT_MODEL, "my-deployment")
    monkeypatch.setattr(
        "agent_pipeline.clients.DefaultAzureCredential", None, raising=False
    )
    # Simulate azure-identity not being installed by making the import fail.
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "azure.identity":
            raise ImportError("No module named 'azure.identity'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(RuntimeError, match="azure-identity"):
        build_chat_client()
