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

The output JSON includes `node_logs` by default. Each row records the generated
node id, parent id, score, action, profile, algorithm, depth, and state repr.

To also save node logs as JSONL:

```powershell
uv run python .\abmcts_engine.py `
  --task "search topic" `
  --profile auto `
  --budget 40 `
  --epoch-budget 8 `
  --batch-size 4 `
  --output .\engine_auto_result.json `
  --node-log .\engine_nodes.jsonl
```

## Live Tree WebUI

Use `--ui` to start a local HTML/SSE visualizer while the search runs. Nodes are
added in real time, and hovering a node shows the action, score, profile,
algorithm, depth, and state summary.

```powershell
uv run python .\abmcts_engine.py `
  --task "search topic" `
  --profile auto `
  --budget 80 `
  --epoch-budget 10 `
  --batch-size 5 `
  --ui `
  --ui-hold-seconds -1 `
  --output .\engine_auto_result.json `
  --node-log .\engine_nodes.jsonl
```

`--ui-hold-seconds -1` keeps the UI server open until `Ctrl+C`. Use
`--no-open-browser` when running in a headless shell.

## GO_DEEP Parent Selection

`ABMCTSExplorer` supports seeded deep expansion through
`ExplorerConfig.metadata["action_parent_states"]`. When this list is present
and a generated trial has no parent from the backend tree, each batch item uses
one seed parent in round-robin order.

`DeepResearchRunner` uses this as a rule: when switching from `GO_WIDE` to
`GO_DEEP`, all candidates tied for the highest score become deep parent
candidates. If two or more depth-1 candidates have the same highest score, all
of them are explored by `GO_DEEP`; they are not collapsed to a single winner.

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
