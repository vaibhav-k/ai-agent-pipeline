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

Run the pipeline with the default topic:

```bash
python main.py
```

Or pass your own topic:

```bash
python main.py "Summarize recent breakthroughs in battery storage."
```

The final, edited output is printed to stdout. Non-zero exit codes indicate a configuration problem (e.g. a missing API key) or a failure during the run; see the log output for details.

## Project Structure

```
ai-agent-pipeline/
├── .env.example      # Template for required environment variables
├── .gitignore
├── LICENSE
├── README.md
├── main.py           # Pipeline definition and CLI entry point
└── requirements.txt
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
