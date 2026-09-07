"""
Chat client construction: OpenAI and Azure OpenAI, auto-selected by environment.

Provider selection: if AZURE_OPENAI_ENDPOINT is set, Azure OpenAI is used;
otherwise the pipeline falls back to plain OpenAI. Both are served by the same
OpenAIChatClient class from agent_framework.
"""

from __future__ import annotations

import logging
import os

from agent_framework.openai import OpenAIChatClient

from . import constants

logger = logging.getLogger(__name__)


def build_chat_client() -> OpenAIChatClient:
    """Create the shared chat client used by every agent in the pipeline."""
    azure_endpoint = os.getenv(constants.ENV_AZURE_OPENAI_ENDPOINT)
    if azure_endpoint:
        return _build_azure_chat_client(azure_endpoint)
    return _build_openai_chat_client()


def _build_openai_chat_client() -> OpenAIChatClient:
    """Build a plain OpenAI chat client from OPENAI_API_KEY / OPENAI_CHAT_MODEL."""
    api_key = os.getenv(constants.ENV_OPENAI_API_KEY)
    if not api_key:
        raise RuntimeError(
            f"{constants.ENV_OPENAI_API_KEY} is not set. Copy .env.example to .env and add your "
            "key, or export it in your shell before running this script. "
            f"(To use Azure OpenAI instead, set {constants.ENV_AZURE_OPENAI_ENDPOINT}.)"
        )
    if not api_key.startswith(constants.OPENAI_KEY_PREFIX):
        logger.warning(
            "%s doesn't start with '%s', which is how OpenAI secret keys are formatted. "
            "Double-check you copied the right value from "
            "https://platform.openai.com/account/api-keys.",
            constants.ENV_OPENAI_API_KEY,
            constants.OPENAI_KEY_PREFIX,
        )

    model = os.getenv(constants.ENV_OPENAI_CHAT_MODEL, constants.DEFAULT_OPENAI_MODEL)
    return OpenAIChatClient(model=model)


def _build_azure_chat_client(azure_endpoint: str) -> OpenAIChatClient:
    """
    Build an Azure OpenAI chat client.

    Auth: uses AZURE_OPENAI_API_KEY if set; otherwise falls back to Azure AD via
    DefaultAzureCredential (the recommended approach for production - no long-lived
    key to leak or rotate). The latter requires the 'azure-identity' package and a
    signed-in identity (e.g. `az login`, or a managed identity in Azure).
    """
    deployment = os.getenv(constants.ENV_AZURE_OPENAI_CHAT_MODEL)
    if not deployment:
        raise RuntimeError(
            f"{constants.ENV_AZURE_OPENAI_CHAT_MODEL} is not set. Set it to the name of your "
            "Azure OpenAI *deployment* (not the underlying model name, e.g. "
            "'gpt-4o-mini-deployment')."
        )

    api_version = os.getenv(constants.ENV_AZURE_OPENAI_API_VERSION)
    api_key = os.getenv(constants.ENV_AZURE_OPENAI_API_KEY)

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
            f"No {constants.ENV_AZURE_OPENAI_API_KEY} is set, so this needs Azure AD auth, which "
            "requires the 'azure-identity' package (pip install azure-identity). Alternatively, "
            f"set {constants.ENV_AZURE_OPENAI_API_KEY} in .env to authenticate with a key instead."
        ) from exc

    logger.info(
        "%s not set; authenticating via Azure AD (DefaultAzureCredential).",
        constants.ENV_AZURE_OPENAI_API_KEY,
    )
    return OpenAIChatClient(
        model=deployment,
        azure_endpoint=azure_endpoint,
        credential=DefaultAzureCredential(),
        api_version=api_version,
    )
