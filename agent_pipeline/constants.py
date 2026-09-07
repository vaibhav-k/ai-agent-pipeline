"""
Centralized constants for the agent pipeline.

Keeping these in one place makes it possible to retune defaults (models,
topics, agent prompts, environment variable names) without hunting through
the orchestration, client, or CLI code.
"""

from __future__ import annotations

# --- Pipeline defaults ---
# Sample topics used when the CLI is run with no topic argument; one is picked
# at random (see main.py) so repeated runs aren't stuck showcasing the same
# example every time. A plain tuple, not a dict: there's no key/value
# relationship here, just a pool to choose from.
DEFAULT_TOPICS: tuple[str, ...] = (
    "Research the current trends in solar energy.",
    "Research the current trends in quantum computing.",
    "Research the current trends in electric vehicle battery technology.",
    "Research the current trends in personalized medicine.",
    "Research the current trends in sustainable agriculture.",
    "Research the current trends in space exploration.",
)
DEFAULT_OPENAI_MODEL = "gpt-5.4"

# --- Logging ---
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# --- Agent identities & instructions ---
RESEARCHER_NAME = "Researcher"
RESEARCHER_INSTRUCTIONS = (
    "Gather comprehensive raw data, statistics, and current facts on the topic."
)

WRITER_NAME = "Writer"
WRITER_INSTRUCTIONS = "Turn the provided research into a clear, concise summary."

EDITOR_NAME = "Editor"
EDITOR_INSTRUCTIONS = "Refine the summary for professional tone, flow, and grammar. Return only the final polished text."

# --- OpenAI environment variables ---
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_CHAT_MODEL = "OPENAI_CHAT_MODEL"
OPENAI_KEY_PREFIX = "sk-"

# --- Azure OpenAI environment variables ---
ENV_AZURE_OPENAI_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
ENV_AZURE_OPENAI_CHAT_MODEL = "AZURE_OPENAI_CHAT_MODEL"
ENV_AZURE_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"
ENV_AZURE_OPENAI_API_VERSION = "AZURE_OPENAI_API_VERSION"

# --- Diagnostics ---
# Case-insensitive substrings that indicate an exception is an auth/API-key failure.
# Checked across the whole __cause__/__context__ chain in agent_pipeline.errors.
AUTH_ERROR_MARKERS = ("401", "unauthorized", "invalid_api_key", "authenticationerror")
