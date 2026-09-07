"""Pipeline execution: runs the workflow and extracts the final text output."""

from __future__ import annotations

import logging

from .agents import build_pipeline
from .clients import build_chat_client

logger = logging.getLogger(__name__)


async def run_pipeline(topic: str) -> str:
    """
    Run the Researcher -> Writer -> Editor pipeline on a topic.

    Returns the final (Editor) agent's text output.

    Args:
        topic: The subject matter to be researched, written about, and edited.

    Raises:
        RuntimeError: If the pipeline finishes but produces no output.

    Returns:
        The final (Editor) agent's text output.
    """
    chat_client = build_chat_client()
    workflow = build_pipeline(chat_client)

    logger.info("Running pipeline for topic: %s", topic)
    result = await workflow.run(topic)

    outputs = result.get_outputs()
    if not outputs:
        raise RuntimeError("Pipeline finished but produced no output.")

    return outputs[-1].text
