"""
Unit tests for agent_pipeline.agents: workflow wiring.

These check structure (agent names, instructions, execution order, shared
client) without running the workflow — see test_runner.py for execution.
"""

from __future__ import annotations

from typing import Any

from agent_framework import Workflow
from agent_framework.openai import OpenAIChatClient

from agent_pipeline import constants
from agent_pipeline.agents import build_pipeline


def _agent_executors(workflow: Workflow) -> list[Any]:
    """
    Executors that wrap an Agent participant (as opposed to e.g. the input-normalizer).

    Untyped (Any) deliberately: `.agent` isn't part of Executor's public type
    surface, only AgentExecutor's, which agent_framework doesn't expose as a
    public import path to check isinstance against here.

    Args:
        workflow: The workflow to inspect.

    Returns:
        A list of executors that wrap an Agent participant.
    """
    return [ex for ex in workflow.get_executors_list() if hasattr(ex, "agent")]


def test_build_pipeline_returns_a_workflow() -> None:
    client = OpenAIChatClient(model="test-model", api_key="sk-test")
    workflow = build_pipeline(client)
    assert isinstance(workflow, Workflow)


def test_agents_are_wired_in_researcher_writer_editor_order() -> None:
    client = OpenAIChatClient(model="test-model", api_key="sk-test")
    workflow = build_pipeline(client)

    names = [ex.agent.name for ex in _agent_executors(workflow)]

    assert names == [
        constants.RESEARCHER_NAME,
        constants.WRITER_NAME,
        constants.EDITOR_NAME,
    ]


def test_agents_receive_their_configured_instructions() -> None:
    client = OpenAIChatClient(model="test-model", api_key="sk-test")
    workflow = build_pipeline(client)

    instructions = [
        ex.agent.default_options["instructions"] for ex in _agent_executors(workflow)
    ]

    assert instructions == [
        constants.RESEARCHER_INSTRUCTIONS,
        constants.WRITER_INSTRUCTIONS,
        constants.EDITOR_INSTRUCTIONS,
    ]


def test_all_agents_share_the_same_chat_client() -> None:
    client = OpenAIChatClient(model="test-model", api_key="sk-test")
    workflow = build_pipeline(client)

    assert all(ex.agent.client is client for ex in _agent_executors(workflow))
