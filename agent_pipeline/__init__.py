"""Sequential multi-agent content pipeline built on Microsoft Agent Framework.

Three specialized agents run one after another, each building on the previous
agent's output: Researcher -> Writer -> Editor.

Modules:
    constants - all tunable constants (defaults, prompts, env var names)
    clients   - chat client construction (OpenAI / Azure OpenAI)
    agents    - agent definitions and workflow wiring
    runner    - executes the workflow and extracts the final output
    errors    - diagnostic helpers (e.g. auth-error detection)
"""

from __future__ import annotations

from .runner import run_pipeline

__all__ = ["run_pipeline"]
