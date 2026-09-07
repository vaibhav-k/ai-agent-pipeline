"""
Shared pytest fixtures for the agent_pipeline test suite.

All tests run offline: OpenAIChatClient.get_response is monkeypatched so no
real network call or API key is ever needed, and provider environment
variables are cleared before every test so nothing leaks from the host shell
or between tests.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from agent_framework import ChatResponse, Message
from agent_framework.openai import OpenAIChatClient

from agent_pipeline import constants

_PROVIDER_ENV_VARS = (
    constants.ENV_OPENAI_API_KEY,
    constants.ENV_OPENAI_CHAT_MODEL,
    constants.ENV_AZURE_OPENAI_ENDPOINT,
    constants.ENV_AZURE_OPENAI_CHAT_MODEL,
    constants.ENV_AZURE_OPENAI_API_KEY,
    constants.ENV_AZURE_OPENAI_API_VERSION,
)


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear provider env vars before every test so tests can't see each other's state."""
    for var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def no_real_dotenv_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Stop main.main_async()'s load_dotenv() call from reading a real .env file.

    Uses python-dotenv's own PYTHON_DOTENV_DISABLED switch rather than patching
    the imported name: load_dotenv() with no arguments finds its .env file via
    find_dotenv(), which searches starting from the *caller's* source file
    location (main.py's directory) - not the test's current working directory -
    so simply chdir'ing during a test would not stop it from finding a real
    .env sitting next to main.py.

    Without this, running the suite from this project's own directory - where
    a real .env with working credentials is expected to exist for actual use -
    lets load_dotenv() re-populate os.environ from that file mid-test,
    silently undoing clean_provider_env above. That's exactly what happened
    when this suite was first run outside the sandbox it was written in:
    test_main_async_returns_one_on_configuration_error passed here (no .env
    present) but failed on a real dev machine with a working .env, because
    the "no provider configured" scenario it's testing no longer held true
    once load_dotenv() ran.
    """
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")


async def _fake_get_response(
    self,
    messages: list[Message],
    *,
    stream: bool = False,
    options: dict | None = None,
    **kwargs,
):
    """Stand-in for OpenAIChatClient.get_response: echoes the last message, no network call."""
    last_text = messages[-1].text if messages else ""
    reply = f"[reply to: {last_text[:60]!r}]"
    return ChatResponse(messages=Message("assistant", [reply]), model="fake-model")


@pytest.fixture
def fake_chat_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Patch OpenAIChatClient.get_response so pipeline runs never hit the network."""
    monkeypatch.setattr(OpenAIChatClient, "get_response", _fake_get_response)
    yield
