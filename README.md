# abmcts-explorer

AB-MCTS based exploration utilities for research workflows.

This repository contains:

- `abmcts_explorer`: a Python package wrapping TreeQuest AB-MCTS backends
- `abmcts_engine.py`: standalone search-engine CLI
- `deep_research_cli.py`: high-budget DeepResearch runner CLI
- SearXNG web search integration
- llama-server integration through an OpenAI-compatible API
- structured logging for LLM context, timeline logs, and human-readable logs

## Install

```powershell
$env:PYTHONUTF8='1'
$env:UV_LINK_MODE='copy'
uv pip install --link-mode=copy -e ".[dev]"
```

## Search Engine CLI

```powershell
uv run python .\abmcts_engine.py `
  --task "search topic" `
  --profile auto `
  --budget 12 `
  --epoch-budget 4 `
  --batch-size 2 `
  --best-k 2 `
  --output .\engine_auto_result.json
```

## DeepResearch CLI

```powershell
$env:SEARXNG_URL='http://127.0.0.1:4866'
$env:SEARXNG_ENGINE='google'
$env:SEARXNG_LANGUAGE='ja'
$env:LLAMA_SERVER_BASE_URL='http://127.0.0.1:1067'

uv run python .\deep_research_cli.py `
  --topic "research topic" `
  --model "Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL" `
  --total-budget 1000 `
  --epoch-budget 25 `
  --output-dir .\runs\example
```

## Test

```powershell
uv run pytest tests -q
```
