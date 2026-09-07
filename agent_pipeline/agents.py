"""Agent and workflow wiring: Researcher -> Writer -> Editor."""

from __future__ import annotations

from agent_framework import Agent, Workflow
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import SequentialBuilder

from . import constants


def build_pipeline(chat_client: OpenAIChatClient) -> Workflow:
    """
    Wire up the Researcher -> Writer -> Editor sequential workflow.

    All three agents share one chat client; SequentialBuilder passes the
    growing conversation from one agent to the next and surfaces the last
    agent's response as the workflow's output.
    """
    researcher = Agent(
        chat_client,
        name=constants.RESEARCHER_NAME,
        instructions=constants.RESEARCHER_INSTRUCTIONS,
    )
    writer = Agent(
        chat_client,
        name=constants.WRITER_NAME,
        instructions=constants.WRITER_INSTRUCTIONS,
    )
    editor = Agent(
        chat_client,
        name=constants.EDITOR_NAME,
        instructions=constants.EDITOR_INSTRUCTIONS,
    )

    return SequentialBuilder(participants=[researcher, writer, editor]).build()
