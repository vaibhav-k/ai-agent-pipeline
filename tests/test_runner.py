"""
Unit tests for agent_pipeline.runner.run_pipeline.

Uses the fake_chat_client fixture (see conftest.py) so no real network call
or API key is ever made/needed.
"""

from __future__ import annotations

import pytest

from agent_pipeline import constants
from agent_pipeline.runner import run_pipeline


@pytest.mark.asyncio
async def test_run_pipeline_returns_final_agent_text(
    monkeypatch: pytest.MonkeyPatch, fake_chat_client: None
) -> None:
    monkeypatch.setenv(constants.ENV_OPENAI_API_KEY, "sk-test-key")

    result = await run_pipeline("unit test topic")

    assert isinstance(result, str)
    assert result
    # Each of the three agents wraps the previous message in "[reply to: ...]",
    # so three occurrences confirms Researcher -> Writer -> Editor all ran.
    assert result.count("reply to:") == 3


@pytest.mark.asyncio
async def test_run_pipeline_raises_when_no_provider_configured() -> None:
    with pytest.raises(RuntimeError, match=constants.ENV_OPENAI_API_KEY):
        await run_pipeline("unit test topic")


@pytest.mark.asyncio
async def test_run_pipeline_raises_when_workflow_produces_no_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(constants.ENV_OPENAI_API_KEY, "sk-test-key")

    class _EmptyResult:
        def get_outputs(self) -> list[object]:
            return []

    class _FakeWorkflow:
        async def run(self, topic: str) -> _EmptyResult:
            return _EmptyResult()

    monkeypatch.setattr(
        "agent_pipeline.runner.build_pipeline", lambda client: _FakeWorkflow()
    )

    with pytest.raises(RuntimeError, match="no output"):
        await run_pipeline("unit test topic")
