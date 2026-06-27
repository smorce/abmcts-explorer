# DeepResearch Quickstart

このクイックスタートでは、ローカル `llama-server` と SearXNG を使って、AB-MCTS による DeepResearch 実験を実行します。

## 前提

- OS: Windows / PowerShell
- ローカルLLM: `llama-server` が `http://127.0.0.1:1067` で起動済み
- Web検索: SearXNG が `http://127.0.0.1:4866` で起動済み
- Python実行: `uv`

## 今回のテーマ

既定の設定ファイルには、次のテーマを設定済みです。

```text
EUのプライバシー対応の現状と、日本の各企業が実施している主要な対応
```

設定ファイル:

```text
experiments/arc2/configs/config.yaml
```

## 環境変数

通常は `config.yaml` の既定値が `run.py` 起動時に適用されるため、明示設定は不要です。上書きしたい場合は PowerShell で以下を設定します。

```powershell
$env:LLAMA_SERVER_BASE_URL='http://127.0.0.1:1067'
$env:LLM_MODEL='unsloth/Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL'
$env:LLAMA_SERVER_ENABLE_THINKING='false'
$env:LLAMA_SERVER_TEMPERATURE='0.3'
$env:LLAMA_SERVER_MAX_TOKENS='12000'

$env:SEARXNG_URL='http://127.0.0.1:4866'
$env:SEARXNG_ENGINE='google,bing,brave,yandex'
$env:SEARXNG_LANGUAGE='ja'
```

## 接続確認

SearXNG の簡易確認:

```powershell
Invoke-RestMethod 'http://127.0.0.1:4866/search?q=GDPR&format=json&pageno=1&engines=google&language=ja'
```

`llama-server` は OpenAI Responses API 互換で起動している前提です。

## クイック実行

まずは探索数を少なくして動作確認します。

```powershell
$env:PYTHONUTF8='1'
$env:UV_LINK_MODE='copy'
uv run --link-mode=copy experiments/arc2/run.py max_num_nodes=2 top_k=2 algo.class_name=ABMCTSA
```

`ABMCTSM` のバッチ実行を試す場合:

```powershell
$env:PYTHONUTF8='1'
$env:UV_LINK_MODE='copy'
uv run --link-mode=copy experiments/arc2/run.py max_num_nodes=10 top_k=5 algo.class_name=ABMCTSM algo.batch_size=5
```

`uv run --link-mode=copy` が `treequest` のGit固定コミットで失敗する場合は、依存を明示してテスト/実行環境を作る方法を使います。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='src;experiments/arc2'
uv run --no-project --with hydra-core --with omegaconf --with python-dotenv --with tqdm --with httpx --with openai --with 'treequest[abmcts-m,vis]' experiments/arc2/run.py max_num_nodes=2 top_k=2 algo.class_name=ABMCTSA
```

## 出力先

実行結果は Hydra により以下へ保存されます。

```text
outputs/deepresearch/<ALGO>_<timestamp>/
```

主な出力:

- `final_report.md`: 最終レポート
- `final_review.json`: 最終レポートの自動レビュー
- `logs/events.jsonl`: 時系列イベントログ
- `logs/research.log`: 人間向けログ
- `logs/progress.md`: 探索進行のMarkdownログ
- `llm_io/llm_calls.jsonl`: LLM投入用ログ
- `tree/nodes.jsonl`: ノード情報
- `tree/edges.jsonl`: エッジ情報
- `tree/tree.html`: 探索木のHTML可視化

## テスト

DeepResearch 実験部分のテスト:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='src;experiments/arc2'
uv run --no-project --with pytest --with hydra-core --with omegaconf --with python-dotenv --with tqdm --with httpx --with openai --with 'treequest[abmcts-m,vis]' pytest experiments/arc2
```

期待結果:

```text
4 passed
```

## テーマを変える

一時的にテーマを変える場合は、Hydra override を使います。

```powershell
uv run --link-mode=copy experiments/arc2/run.py research_topic='日本企業のGDPR対応とCookie同意管理の最新動向'
```

恒久的に変える場合は、`experiments/arc2/configs/config.yaml` の `research_topic` を編集します。
