# DeepResearch 向けクイックスタート

`ABMCTSExplorer` は、DeepResearch の候補仮説・調査メモ・統合回答を探索木の `state` として扱えます。

基本方針は次の通りです。

- `GO_WIDE`: 初期調査で観点や仮説を広げる
- `GO_DEEP`: 有望な候補を検証・補強・統合する
- `ActionSpec`: 調査、反証、要約、統合などの探索アクションを表す
- `score`: 根拠の具体性、網羅性、一貫性などを 0.0 から 1.0 に正規化する

## 依存関係

PowerShell で実行します。

```powershell
$env:PYTHONUTF8='1'
$env:UV_LINK_MODE='copy'
uv pip install --link-mode=copy "treequest[abmcts-m]"
```

llama-server を使う場合は、必要に応じて環境変数を設定します。

```powershell
$env:LLAMA_SERVER_BASE_URL='http://127.0.0.1:1067'
$env:LLM_MODEL='unsloth/Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL'
$env:LLAMA_SERVER_ENABLE_THINKING='false'
$env:LLAMA_SERVER_TEMPERATURE='0.3'
$env:LLAMA_SERVER_MAX_TOKENS='1200'
```

SearXNG を使う場合は、提示された `curl` と同じ設定を環境変数で渡せます。

```powershell
$env:SEARXNG_URL='http://127.0.0.1:4866'
$env:SEARXNG_ENGINE='google'
$env:SEARXNG_LANGUAGE='ja'
```

## 最小例

以下は、外部検索 API などをまだ接続せず、LLM だけで DeepResearch 風の探索を行う最小例です。

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from abmcts_explorer import (
    ABMCTSExplorer,
    ActionSpec,
    ExplorerContext,
    ExplorerPreset,
)


@dataclass(frozen=True)
class ResearchState:
    question: str
    draft: str
    evidence_notes: tuple[str, ...]
    open_questions: tuple[str, ...]
    depth: int = 0


def build_research_prompt(
    parent_state: ResearchState | None,
    context: ExplorerContext,
) -> list[dict[str, str]]:
    if parent_state is None:
        user_content = f"""
調査テーマ:
{context.task}

DeepResearch の初期候補を作ってください。
出力には、暫定回答、根拠メモ、未解決論点を含めてください。
"""
    else:
        user_content = f"""
調査テーマ:
{context.task}

現在の暫定回答:
{parent_state.draft}

根拠メモ:
{chr(10).join(parent_state.evidence_notes)}

未解決論点:
{chr(10).join(parent_state.open_questions)}

この候補を改善してください。
不足している根拠、反証可能性、次に調べるべき論点を補ってください。
"""

    return [
        {
            "role": "system",
            "content": (
                "You are a rigorous DeepResearch assistant. "
                "Separate claims, evidence, uncertainty, and next questions."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def parse_research_state(
    raw: str,
    parent_state: ResearchState | None,
    context: ExplorerContext,
) -> ResearchState:
    text = raw.strip()
    return ResearchState(
        question=context.task,
        draft=text,
        evidence_notes=tuple(
            line.strip("- ").strip()
            for line in text.splitlines()
            if "根拠" in line or "evidence" in line.lower()
        ),
        open_questions=tuple(
            line.strip("- ").strip()
            for line in text.splitlines()
            if "未解決" in line or "question" in line.lower()
        ),
        depth=0 if parent_state is None else parent_state.depth + 1,
    )


def score_research_state(
    state: ResearchState,
    parent_state: ResearchState | None,
    context: ExplorerContext,
) -> float:
    text = state.draft
    if not text:
        return 0.0

    length_score = min(len(text) / 3000, 1.0)
    evidence_score = min(len(state.evidence_notes) / 4, 1.0)
    question_score = min(len(state.open_questions) / 3, 1.0)
    structure_score = 1.0 if ("根拠" in text and "未解決" in text) else 0.5

    score = (
        length_score * 0.30
        + evidence_score * 0.30
        + question_score * 0.20
        + structure_score * 0.20
    )
    return max(0.0, min(score, 1.0))


actions = [
    ActionSpec[ResearchState](
        name="broaden_hypotheses",
        model="llama-research-fast",
        prompt_builder=build_research_prompt,
        parser=parse_research_state,
        scorer=score_research_state,
        temperature=0.7,
        max_tokens=1600,
    ),
    ActionSpec[ResearchState](
        name="verify_and_refine",
        model="llama-research-precise",
        prompt_builder=build_research_prompt,
        parser=parse_research_state,
        scorer=score_research_state,
        temperature=0.2,
        max_tokens=2200,
    ),
]


async def main() -> None:
    explorer = ABMCTSExplorer[ResearchState](
        task="生成AIエージェントの評価指標として、タスク成功率だけでは不足する理由を調査する",
        actions=actions,
        config=ExplorerPreset.go_wide(
            budget=24,
            batch_size=6,
        ),
    )

    wide_best = await explorer.run_async()

    explorer.switch_config(
        ExplorerPreset.go_deep(
            budget=10,
            batch_size=5,
        ),
        preserve_tree=False,
    )

    deep_best = explorer.run()

    for rank, (state, score) in enumerate(deep_best or wide_best, start=1):
        print(f"## Candidate {rank}: score={score:.3f}")
        print(state.draft)
        print()


if __name__ == "__main__":
    asyncio.run(main())
```

## 実運用で差し替える箇所

`build_research_prompt` には、検索済みスニペット、引用候補、社内文書の抜粋などを `context.metadata` 経由で渡せます。

SearXNG を使う場合は、`SearXNGSearchClient` で検索結果を取得し、`ExplorerConfig.metadata` に入れてから探索を開始できます。

```python
from abmcts_explorer import (
    ExecutionMode,
    ExplorerConfig,
    SearchProfile,
    SearXNGSearchClient,
)


search_client = SearXNGSearchClient()
search_response = search_client.search(
    "AI agent evaluation metrics task success rate limitations",
    limit=5,
)

web_results = [
    {
        "title": result.title,
        "url": result.url,
        "description": result.description,
    }
    for result in search_response.results
]

config = ExplorerConfig(
    profile=SearchProfile.GO_WIDE,
    execution_mode=ExecutionMode.ASYNC_ASK_TELL,
    budget=24,
    batch_size=6,
    metadata={"web_results": web_results},
)
```

`build_research_prompt` 側では、次のように検索結果を prompt に含めます。

```python
web_results = context.metadata.get("web_results", [])
web_context = "\n".join(
    f"- {item['title']}: {item['description']} ({item['url']})"
    for item in web_results
)
```

`parse_research_state` は、本番では JSON 出力を要求して `json.loads` とスキーマ検証に置き換えるのが安全です。

`score_research_state` は、次の評価軸に置き換えると DeepResearch 向けになります。

- 根拠の一次情報性
- 出典の多様性
- 主張と根拠の対応
- 反証・不確実性の明示
- 最終回答としての統合度

## 推奨プロファイル

初期探索は `ExplorerPreset.go_wide(budget=100, batch_size=16)` から始め、候補が収束したら `ExplorerPreset.go_deep(budget=30, batch_size=5)` に切り替えます。

TreeQuest の `ABMCTSM` は重くなりやすいため、`go_deep` の `batch_size` は 5 前後から始めてください。

## Web調査での切り替え方針

テーマを受け取ったら、まず `SearXNGSearchClient.search(topic, limit=5)` で初期検索結果を集め、`ExplorerConfig.metadata["web_results"]` に渡します。

`GO_WIDE` では、検索語を広げる action、別観点を出す action、反証候補を探す action を複数用意します。候補の種類が少ない、未解決論点が多い、検索結果が薄い場合は `GO_WIDE` を継続します。

`GO_DEEP` では、上位候補に絞って根拠の対応付け、矛盾確認、最終回答への統合を行います。上位候補の score が高く、候補間の差が小さくなり、残り budget が少なくなったら `GO_DEEP` に切り替えます。

実装上は、epoch ごとに次の順で回します。

```python
search_response = search_client.search(topic, limit=5)
explorer.config.metadata["web_results"] = [
    {
        "title": result.title,
        "url": result.url,
        "description": result.description,
    }
    for result in search_response.results
]

if should_go_deep:
    explorer.switch_config(ExplorerPreset.go_deep(budget=30, batch_size=5))
else:
    explorer.switch_config(ExplorerPreset.go_wide(budget=100, batch_size=16))
```

`should_go_deep` は `RuleBasedProfileDecider` に `SearchDiagnostics` を渡して決めるか、最初は「初期 2 epoch は `GO_WIDE`、以降は上位 score と未解決論点数で `GO_DEEP`」という単純なルールから始めるのが実装しやすいです。

## 自動 DeepResearch Runner

`DeepResearchRunner` を使うと、テーマを渡すだけで次のループを実行します。

1. SearXNG でテーマを検索
2. 検索結果を `ExplorerConfig.metadata["web_results"]` に保存
3. `GO_WIDE` で観点・反証・追加論点を広げる
4. 探索候補から `SearchDiagnostics` を作る
5. `RuleBasedProfileDecider` が `GO_WIDE` / `GO_DEEP` を判断する
6. `GO_DEEP` で根拠確認・矛盾確認・統合を厚くする
7. 最終レポートを Markdown で返す

```python
from __future__ import annotations

import asyncio

from abmcts_explorer import DeepResearchRunner, DeepResearchRunnerConfig


async def main() -> None:
    runner = DeepResearchRunner(
        config=DeepResearchRunnerConfig(
            total_budget=80,
            epoch_budget=10,
            search_limit=5,
            min_wide_epochs=2,
            wide_batch_size=8,
            deep_batch_size=5,
            report_model="llama-research-precise",
        )
    )

    result = await runner.run(
        "AIエージェントの評価指標としてタスク成功率だけでは不足する理由"
    )

    if result.logger:
        result.logger.save("deep_research_logs")

    print(result.report)
    print([decision.profile.value for decision in result.profile_history])


if __name__ == "__main__":
    asyncio.run(main())
```

`report_model` を `None` にすると、LLM による最終統合を行わず、探索候補から決定的に Markdown を組み立てます。CI や疎通確認では `None`、本番レポートでは llama-server のモデル名を指定してください。

`result.logger.save("deep_research_logs")` は次の 3 種類のログを保存します。

- `llm_context_log.json`: 次の LLM 呼び出しに渡しやすい構造化ログ
- `timeline_log.jsonl`: 時系列で追えるイベントログ
- `human_log.md`: 人間がレビューしやすい Markdown ログ

## 高 budget で一気に回す

コストを気にしない場合は、専用 CLI で `total-budget` を大きくします。

```powershell
$env:SEARXNG_URL='http://127.0.0.1:4866'
$env:SEARXNG_ENGINE='google'
$env:SEARXNG_LANGUAGE='ja'
$env:LLAMA_SERVER_BASE_URL='http://127.0.0.1:1067'

uv run python .\ABMCTSExplorer\deep_research_cli.py `
  --topic "調査テーマ" `
  --model "Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL" `
  --total-budget 1000 `
  --epoch-budget 25 `
  --wide-batch-size 16 `
  --deep-batch-size 5 `
  --best-k 8 `
  --search-limit 8 `
  --output-dir .\ABMCTSExplorer\runs\expert_distillation
```

この設定では、序盤は `GO_WIDE` で論点・仮説・反証候補を大量に広げ、診断値が深掘り寄りになったら `GO_DEEP` へ自動で切り替えます。出力先には `report.md`, `profile_history.md`, `sources.md`, `logs/` が保存されます。

## WebUI で探索木を確認する

探索中の木をリアルタイムで確認したい場合は、DeepResearch CLI に `--ui` を付けます。
`--ui-hold-seconds -1` を指定すると、探索完了後も WebUI サーバーを開いたままにできます。
ブラウザを自動で開きたくない場合は `--no-open-browser` を付けてください。

```powershell
$env:SEARXNG_URL='http://127.0.0.1:4866'
$env:SEARXNG_ENGINE='google'
$env:SEARXNG_LANGUAGE='ja'
$env:LLAMA_SERVER_BASE_URL='http://127.0.0.1:1067'

uv run python .\ABMCTSExplorer\deep_research_cli.py `
  --topic "EUのプライバシー対応の現状と日本の各企業がどんな主要な対応をしているのか？" `
  --model "Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL" `
  --total-budget 1000 `
  --epoch-budget 25 `
  --wide-batch-size 16 `
  --deep-batch-size 5 `
  --best-k 8 `
  --search-limit 8 `
  --output-dir .\ABMCTSExplorer\runs\expert_distillation `
  --ui `
  --ui-port 8770 `
  --no-open-browser `
  --ui-hold-seconds -1
```

起動後は次の URL をブラウザで開きます。

```text
http://127.0.0.1:8770/
```

UI は 5 秒ごとに `/snapshot` をポーリングしてノードを自動更新します。
現在探索中のノードはパルス表示され、ノードをクリックすると tooltip 相当の詳細を固定パネルで確認できます。
探索は `GO_WIDE -> GO_DEEP -> GO_WIDE` のサイクルで進み、同点最高スコアのノードが複数ある場合は全件が次の深掘りまたは横展開の親候補になります。
