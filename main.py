"""CLI entry point for the sequential agent pipeline.

Usage:
    python main.py "Research the current trends in solar energy."
    python main.py                       # picks a random sample topic

The actual pipeline logic lives in the agent_pipeline package: constants,
chat client construction, agent/workflow wiring, execution, and error
diagnostics are each in their own module there.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys

from dotenv import load_dotenv

from agent_pipeline import constants
from agent_pipeline.errors import looks_like_auth_error
from agent_pipeline.runner import run_pipeline

logging.basicConfig(level=logging.INFO, format=constants.LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sequential agent pipeline on a topic."
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="Topic to run through the pipeline. If omitted, a random sample topic is used.",
    )
    return parser.parse_args(argv)


async def main_async(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    topic = args.topic
    if topic is None:
        topic = random.choice(constants.DEFAULT_TOPICS)
        logger.info("No topic given; picked a random sample topic: %s", topic)

    try:
        final_text = await run_pipeline(topic)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("Pipeline run failed.")
        if looks_like_auth_error(exc):
            logger.error(
                "This looks like an authentication problem. If you're using OpenAI, verify %s "
                "in your .env file is a current, valid secret key from "
                "https://platform.openai.com/account/api-keys. If you're using Azure OpenAI, "
                "verify %s/%s (or your Azure AD sign-in) are correct for that resource.",
                constants.ENV_OPENAI_API_KEY,
                constants.ENV_AZURE_OPENAI_ENDPOINT,
                constants.ENV_AZURE_OPENAI_API_KEY,
            )
        return 1

    print("\n--- Final Output ---")
    print(final_text)
    return 0


def main() -> None:
    try:
        exit_code = asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
