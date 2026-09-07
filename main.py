"""
Sequential multi-agent content pipeline built on Microsoft Agent Framework.

Three specialized agents run one after another, each building on the previous
agent's output:

    Researcher -> Writer -> Editor

Usage:
    python main.py "Research the current trends in solar energy."
    python main.py                       # uses the default topic below
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from agent_framework import Agent, Workflow
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import SequentialBuilder
from dotenv import load_dotenv

DEFAULT_TOPIC = "Research the current trends in solar energy."
DEFAULT_MODEL = "gpt-4o-mini"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def build_chat_client() -> OpenAIChatClient:
    """Create the shared chat client used by every agent in the pipeline.

    Uses Azure OpenAI when AZURE_OPENAI_ENDPOINT is set; otherwise falls back to
    plain OpenAI. Both providers are served by the same OpenAIChatClient class.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        return _build_azure_chat_client(azure_endpoint)
    return _build_openai_chat_client()


def _build_openai_chat_client() -> OpenAIChatClient:
    """Build a plain OpenAI chat client from OPENAI_API_KEY / OPENAI_CHAT_MODEL."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or export it in your shell before running this script. "
            "(To use Azure OpenAI instead, set AZURE_OPENAI_ENDPOINT.)"
        )
    if not api_key.startswith("sk-"):
        logger.warning(
            "OPENAI_API_KEY doesn't start with 'sk-', which is how OpenAI secret keys are "
            "formatted. Double-check you copied the right value from "
            "https://platform.openai.com/account/api-keys."
        )

    model = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_MODEL)
    return OpenAIChatClient(model=model)


def _build_azure_chat_client(azure_endpoint: str) -> OpenAIChatClient:
    """Build an Azure OpenAI chat client.

    Auth: uses AZURE_OPENAI_API_KEY if set; otherwise falls back to Azure AD via
    DefaultAzureCredential (the recommended approach for production - no long-lived
    key to leak or rotate). The latter requires the 'azure-identity' package and
    a signed-in identity (e.g. `az login`, or a managed identity in Azure).
    """
    deployment = os.getenv("AZURE_OPENAI_CHAT_MODEL")
    if not deployment:
        raise RuntimeError(
            "AZURE_OPENAI_CHAT_MODEL is not set. Set it to the name of your Azure OpenAI "
            "*deployment* (not the underlying model name, e.g. 'gpt-4o-mini-deployment')."
        )

    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if api_key:
        return OpenAIChatClient(
            model=deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    try:
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        raise RuntimeError(
            "No AZURE_OPENAI_API_KEY is set, so this needs Azure AD auth, which requires the "
            "'azure-identity' package (pip install azure-identity). Alternatively, set "
            "AZURE_OPENAI_API_KEY in .env to authenticate with a key instead."
        ) from exc

    logger.info(
        "AZURE_OPENAI_API_KEY not set; authenticating via Azure AD (DefaultAzureCredential)."
    )
    return OpenAIChatClient(
        model=deployment,
        azure_endpoint=azure_endpoint,
        credential=DefaultAzureCredential(),
        api_version=api_version,
    )


def build_pipeline(chat_client: OpenAIChatClient) -> Workflow:
    """Wire up the Researcher -> Writer -> Editor sequential workflow."""
    researcher = Agent(
        chat_client,
        name="Researcher",
        instructions="Gather comprehensive raw data, statistics, and current facts on the topic.",
    )
    writer = Agent(
        chat_client,
        name="Writer",
        instructions="Turn the provided research into a clear, concise summary.",
    )
    editor = Agent(
        chat_client,
        name="Editor",
        instructions="Refine the summary for professional tone, flow, and grammar. "
        "Return only the final polished text.",
    )

    return SequentialBuilder(participants=[researcher, writer, editor]).build()


async def run_pipeline(topic: str) -> str:
    """Run the pipeline on a topic and return the final agent's text output."""
    chat_client = build_chat_client()
    workflow = build_pipeline(chat_client)

    logger.info("Running pipeline for topic: %s", topic)
    result = await workflow.run(topic)

    outputs = result.get_outputs()
    if not outputs:
        raise RuntimeError("Pipeline finished but produced no output.")

    return outputs[-1].text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sequential agent pipeline on a topic."
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=DEFAULT_TOPIC,
        help=f"Topic to run through the pipeline (default: {DEFAULT_TOPIC!r}).",
    )
    return parser.parse_args(argv)


def _looks_like_auth_error(exc: BaseException) -> bool:
    """Heuristic check for an API-key/authentication failure anywhere in the exception chain."""
    markers = ("401", "unauthorized", "invalid_api_key", "authenticationerror")
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__} {current}".lower()
        if any(marker in text for marker in markers):
            return True
        current = current.__cause__ or current.__context__
    return False


async def main_async(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    try:
        final_text = await run_pipeline(args.topic)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("Pipeline run failed.")
        if _looks_like_auth_error(exc):
            logger.error(
                "This looks like an authentication problem. If you're using OpenAI, verify "
                "OPENAI_API_KEY in your .env file is a current, valid secret key from "
                "https://platform.openai.com/account/api-keys. If you're using Azure OpenAI, "
                "verify AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY (or your Azure AD sign-in) "
                "are correct for that resource."
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
