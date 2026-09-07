# AI Agent Pipeline

A lightweight, asynchronous multi-agent orchestration pipeline built with Python and [Microsoft Agent Framework](https://github.com/microsoft/agent-framework). Three specialized agents run in sequence, each handing its output to the next, to research, write, and edit content on a given topic.

## Pipeline Architecture

```
topic -> [Researcher] -> [Writer] -> [Editor] -> final text
```

1. **Researcher agent** — gathers raw data, statistics, and foundational context.
2. **Writer agent** — synthesizes the raw data into a concise summary.
3. **Editor agent** — refines the summary for professional tone, grammar, and clarity.

Under the hood this uses Agent Framework's `SequentialBuilder`, which passes the growing conversation from one agent to the next and returns the last agent's response as the pipeline's output.

The code is split into a thin CLI (`main.py`) and an `agent_pipeline` package that does the actual work:

- `constants.py` — every tunable value in one place: default topic/model, agent names and prompts, environment variable names, and the auth-error markers used for diagnostics.
- `clients.py` — builds the chat client, auto-selecting OpenAI or Azure OpenAI.
- `agents.py` — defines the three agents and wires them into a `SequentialBuilder` workflow.
- `runner.py` — runs the workflow for a given topic and extracts the final text.
- `errors.py` — turns a raw exception into "does this look like an auth problem?" for friendlier CLI messages.

`main.py` only handles argument parsing, logging setup, and translating pipeline errors into exit codes — it has no orchestration logic of its own.

## Prerequisites

- Python 3.10 or higher
- Either an OpenAI API key, or an Azure OpenAI resource with a deployed chat model — see [Configuring a provider](#configuring-a-provider)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/vaibhav-k/ai-agent-pipeline.git
   cd ai-agent-pipeline
   ```

2. (Recommended) create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure your environment variables:

   ```bash
   cp .env.example .env
   ```

   Open `.env` and fill in one of the two provider sections — see below.

## Configuring a provider

The pipeline auto-detects which provider to use: if `AZURE_OPENAI_ENDPOINT` is set, it uses Azure OpenAI; otherwise it uses plain OpenAI. Both are handled by the same `OpenAIChatClient` under the hood, so switching providers is just an environment-variable change, not a code change.

**OpenAI** — set in `.env`:

- `OPENAI_API_KEY` (required)
- `OPENAI_CHAT_MODEL` (optional, defaults to `gpt-4o-mini`)

**Azure OpenAI** — set in `.env`:

- `AZURE_OPENAI_ENDPOINT` (required) — your resource endpoint, e.g. `https://<resource-name>.openai.azure.com`
- `AZURE_OPENAI_CHAT_MODEL` (required) — the *deployment* name from Azure AI Foundry / Azure OpenAI Studio, not the underlying model name
- `AZURE_OPENAI_API_KEY` (optional) — if set, authenticates with a key. If left unset, the pipeline authenticates with Azure AD instead via `DefaultAzureCredential` (e.g. a local `az login`, or a managed identity when running in Azure) — this is the recommended approach for production since there's no long-lived key to leak or rotate. Azure AD auth needs the `azure-identity` package, already in `requirements.txt`.
- `AZURE_OPENAI_API_VERSION` (optional) — leave unset to use the library's current default

## Usage

Run with no arguments to get a random sample topic (see `DEFAULT_TOPICS` in `agent_pipeline/constants.py` for the pool it picks from):

```bash
python main.py
```

Or pass your own topic:

```bash
python main.py "Summarize recent breakthroughs in battery storage."
```

The final, edited output is printed to stdout. Non-zero exit codes indicate a configuration problem (e.g. a missing API key) or a failure during the run; see the log output for details.

## Testing

The test suite runs entirely offline — it monkeypatches the OpenAI client so no network call, API key, or Azure resource is ever needed, and no test relies on the others' state. It's also isolated from this project's own `.env`: provider env vars are cleared before every test, and python-dotenv's loading is disabled for the duration of the suite (via `PYTHON_DOTENV_DISABLED`), so a real, working `.env` sitting right here can't leak credentials into a test that's specifically checking what happens when none are configured.

Install the extra test dependencies (on top of `requirements.txt`) and run:

```bash
pip install -r requirements-dev.txt
pytest
```

Coverage by module:

- `test_constants.py` — sanity checks on defaults and, especially, the environment variable names (a typo there would silently break provider auto-detection).
- `test_clients.py` — provider selection (OpenAI vs. Azure OpenAI, Azure takes precedence when both are configured), the missing-key/missing-deployment error messages, the malformed-key warning, and both Azure auth paths (API key and the `DefaultAzureCredential` fallback, including the case where `azure-identity` isn't installed).
- `test_agents.py` — the workflow is wired Researcher → Writer → Editor, each with its configured instructions, all sharing one chat client.
- `test_runner.py` — a full mocked pipeline run returns the final agent's text, plus the "no output produced" error path.
- `test_errors.py` — the auth-error heuristic across direct messages, explicit (`raise ... from`) and implicit exception chains, unrelated errors, and a pathological self-referential chain (must not hang).
- `test_main.py` — CLI argument defaults/overrides, the random-topic pick, exit codes for success/config-error/auth-error, `Ctrl+C` handling, and the `.env`-isolation regression test described above.

## Project Structure

```
ai-agent-pipeline/
├── .env.example          # Template for required environment variables
├── .gitignore
├── LICENSE
├── README.md
├── main.py               # CLI entry point (argument parsing, exit codes)
├── requirements.txt
├── requirements-dev.txt  # requirements.txt + pytest, for running the test suite
├── pytest.ini
├── agent_pipeline/
│   ├── __init__.py       # Public API: run_pipeline
│   ├── constants.py      # Defaults, prompts, env var names, error markers
│   ├── clients.py        # OpenAI / Azure OpenAI chat client construction
│   ├── agents.py         # Agent definitions and SequentialBuilder wiring
│   ├── runner.py         # Executes the workflow, extracts the final output
│   └── errors.py         # Exception -> friendly-diagnostic heuristics
└── tests/
    ├── conftest.py        # Shared fixtures: offline chat client, clean/isolated env
    ├── test_constants.py
    ├── test_clients.py
    ├── test_agents.py
    ├── test_runner.py
    ├── test_errors.py
    └── test_main.py
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
