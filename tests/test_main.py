"""Unit tests for main.py: CLI argument parsing and exit-code mapping."""

from __future__ import annotations

import logging
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest

import main
from agent_pipeline import constants


def test_parse_args_leaves_topic_none_when_not_given() -> None:
    # Random selection happens in main_async, not here, so parse_args stays a
    # pure, deterministic function of its arguments.
    args = main.parse_args([])
    assert args.topic is None


def test_parse_args_uses_provided_topic() -> None:
    args = main.parse_args(["my custom topic"])
    assert args.topic == "my custom topic"


@pytest.mark.asyncio
async def test_main_async_returns_zero_and_prints_output_on_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def _fake_run_pipeline(topic: str) -> str:
        return "the final answer"

    monkeypatch.setattr(main, "run_pipeline", _fake_run_pipeline)

    exit_code = await main.main_async(["some topic"])

    assert exit_code == 0
    assert "the final answer" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_main_async_picks_a_random_topic_when_none_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    async def _fake_run_pipeline(topic: str) -> str:
        captured["topic"] = topic
        return "the final answer"

    monkeypatch.setattr(main, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(main.random, "choice", lambda pool: pool[0])  # type: ignore[arg-type]

    exit_code = await main.main_async([])

    assert exit_code == 0
    assert captured["topic"] == constants.DEFAULT_TOPICS[0]


@pytest.mark.asyncio
async def test_main_async_does_not_randomize_when_topic_is_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_run_pipeline(topic: str) -> str:
        return "the final answer"

    def _must_not_be_called(pool: object) -> None:
        raise AssertionError("random.choice should not be called when a topic is given")

    monkeypatch.setattr(main, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(main.random, "choice", _must_not_be_called)

    exit_code = await main.main_async(["an explicit topic"])

    assert exit_code == 0


@pytest.mark.asyncio
async def test_main_async_returns_one_on_configuration_error() -> None:
    # No provider env vars are set (autouse fixture clears them), so this hits
    # the real RuntimeError from build_chat_client via run_pipeline.
    exit_code = await main.main_async(["some topic"])
    assert exit_code == 1


def test_dotenv_loading_is_disabled_during_tests(tmp_path: Path) -> None:
    """Regression test: a real .env file must never leak into a test's os.environ.

    Uses the actual dotenv.load_dotenv() with an explicit dotenv_path (rather
    than going through main_async(), which relies on find_dotenv()'s
    frame-based search) so this test's result doesn't depend on where .env
    happens to live - only on whether the PYTHON_DOTENV_DISABLED switch that
    no_real_dotenv_loading (see conftest.py) sets is actually respected.
    """
    from dotenv import load_dotenv as real_load_dotenv

    fake_env_file = tmp_path / ".env"
    fake_env_file.write_text(
        f"{constants.ENV_OPENAI_API_KEY}=sk-should-never-be-loaded\n"
    )

    loaded_something = real_load_dotenv(dotenv_path=fake_env_file)

    assert loaded_something is False
    assert constants.ENV_OPENAI_API_KEY not in os.environ


@pytest.mark.asyncio
async def test_main_async_logs_friendly_hint_on_auth_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _fake_run_pipeline(topic: str) -> str:
        raise Exception("401 Unauthorized")  # noqa: TRY002

    monkeypatch.setattr(main, "run_pipeline", _fake_run_pipeline)

    with caplog.at_level(logging.ERROR):
        exit_code = await main.main_async(["some topic"])

    assert exit_code == 1
    assert "authentication problem" in caplog.text


def test_main_exits_130_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_keyboard_interrupt(coro: Coroutine[Any, Any, Any]) -> None:
        coro.close()  # avoid a "coroutine was never awaited" warning
        raise KeyboardInterrupt

    monkeypatch.setattr(main.asyncio, "run", _raise_keyboard_interrupt)

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 130
