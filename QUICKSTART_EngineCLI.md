# 探索エンジン CLI クイックスタート

DeepResearch から切り離して、`ABMCTSExplorer` の探索エンジンだけを CLI 風に実行できます。

## デモ実行

```powershell
$env:PYTHONUTF8='1'
$env:UV_LINK_MODE='copy'
uv run python .\ABMCTSExplorer\abmcts_engine.py `
  --task "探索エンジンCLIテスト" `
  --profile go_wide `
  --budget 8 `
  --batch-size 2 `
  --best-k 2 `
  --output .\ABMCTSExplorer\engine_result.json
```

`--profile go_wide` は `ABMCTSA` と `ASYNC_ASK_TELL` を使います。

`--profile go_deep` は `ABMCTSM` と `ASK_TELL` を使います。

`--profile auto` は epoch ごとに `SearchDiagnostics` を作り、`RuleBasedProfileDecider` で次の profile を選びます。

```powershell
uv run python .\ABMCTSExplorer\abmcts_engine.py `
  --task "探索テーマ" `
  --profile auto `
  --budget 12 `
  --epoch-budget 4 `
  --batch-size 2 `
  --best-k 2 `
  --output .\ABMCTSExplorer\engine_auto_result.json
```

`auto` の出力には `profile_decisions`, `diagnostics`, `last_executed_profile`, `next_profile_recommendation` が含まれます。

## JSONL 入力

候補を外部から渡す場合は JSONL を使えます。

```jsonl
{"text":"候補A","score":0.4}
{"text":"候補B","score":0.9}
```

```powershell
uv run python .\ABMCTSExplorer\abmcts_engine.py `
  --task "JSONL候補探索" `
  --profile go_wide `
  --budget 8 `
  --batch-size 2 `
  --best-k 2 `
  --input-jsonl .\states.jsonl `
  --output .\engine_result.json
```

## 出力

出力 JSON には次を含みます。

- `best`: 上位候補、score、state
- `history`: `ask_tell_batch_async` などの探索イベント
- `profile`, `algorithm`, `budget`, `batch_size`

この CLI は LLM や Web検索に依存しません。探索器単体の smoke test、外部生成候補のランキング、`GO_WIDE` / `GO_DEEP` の挙動確認に使います。
