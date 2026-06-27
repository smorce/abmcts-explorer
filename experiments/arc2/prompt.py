from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from utils import NodeState, ResearchAction


ACTION_DESCRIPTIONS: dict[ResearchAction, str] = {
    "new_angle": "未探索の観点から新しい仮説・論点を立てる。",
    "deepen": "親ノードの仮説について、検索結果を根拠に深掘りして裏取りする。",
    "criticize": "親ノードの仮説に対する反証、弱点、矛盾、古い情報を探す。",
    "revise": "親ノードの自動レビュー指摘を解消するため、再検索と主張修正を行う。",
}


ACTION_QUERY_INSTRUCTIONS: dict[ResearchAction, str] = {
    "new_angle": "親ノードの主張に引きずられず、指定観点から未探索の角度を広げる。",
    "deepen": "親ノードの open_questions を具体的に検証できる検索にする。",
    "criticize": "親ノードの主張の反証、弱点、矛盾、古い情報を探す。",
    "revise": "親ノードの findings を解消する追加根拠や具体事例を探す。",
}


def query_planner_system_prompt() -> str:
    return """
あなたはDeepResearchの検索クエリ設計者です。
次の探索アクションで使うWeb検索クエリを、日本語の短いキーワード列として設計してください。

守ること:
- 必ずJSONオブジェクトだけを返す。
- 形式は {"queries": ["単語 単語 単語", "..."]}。
- クエリは自然文や質問文にしない。
- 各クエリは空白区切りで最大語数以内にする。
- 必要な本数だけ返し、最大本数を無理に埋めない。
- 同じ意味のクエリを重複させない。
""".strip()


def researcher_system_prompt(action: ResearchAction, perspective: str) -> str:
    action_description = ACTION_DESCRIPTIONS.get(action, ACTION_DESCRIPTIONS["deepen"])
    return f"""
あなたはDeepResearchチームの調査担当です。
現在の役割は「{perspective}」観点のリサーチャーです。
今回の探索アクションは「{action}」です: {action_description}

守ること:
- 検索結果に含まれる情報だけを根拠として扱う。
- 不確かな主張は不確かと明記する。
- 根拠、反証、残った不確実性を分けて書く。
- 最終回答ではなく、この探索ノードで得られた調査メモを作る。
- 日本語で簡潔に出力する。
""".strip()


def reviewer_system_prompt() -> str:
    return """
あなたはDeepResearchの厳格な自動レビュアーです。
調査メモに対して、根拠不足、古い情報、矛盾、出典の弱さ、結論の飛躍を探してください。

必ず次のJSONオブジェクトだけを返してください。
{
  "score": 0.0から1.0の数値,
  "summary": "短い総評",
  "findings": ["修正または再調査すべき指摘", "..."]
}

採点基準:
- 1.0: 強い根拠が複数あり、矛盾と限界も扱えている
- 0.7: 実用的だが、補強すべき点が残る
- 0.4: 重要な根拠不足または飛躍がある
- 0.0: 調査として使えない
""".strip()


def final_report_system_prompt() -> str:
    return """
あなたはDeepResearchの編集長です。
探索済みノードのうち有望な論点だけを統合し、根拠と不確実性が分かる最終レポートを作成してください。

構成:
1. Executive Summary
2. 主要論点
3. 根拠と出典
4. 反証・矛盾・限界
5. 結論
6. 次に調べるべきこと

日本語で、出典URLを本文中に含めてください。
""".strip()


def build_research_prompt(
    *,
    topic: str,
    action: ResearchAction,
    perspective: str,
    search_queries: list[str],
    sources: list[dict[str, Any]],
    parent_state: NodeState | None,
) -> str:
    parent_summary = "親ノードはありません。今回が初回の論点設計です。"
    if parent_state is not None:
        parent_summary = json.dumps(
            {
                "action": parent_state.action,
                "perspective": parent_state.perspective,
                "text": parent_state.text,
                "open_questions": parent_state.open_questions,
                "findings": parent_state.findings,
                "search_queries": parent_state.search_queries,
                "score": parent_state.score,
            },
            ensure_ascii=False,
            indent=2,
        )

    source_block = json.dumps(sources, ensure_ascii=False, indent=2)
    return f"""
# 調査テーマ
{topic}

# アクション
{action}: {ACTION_DESCRIPTIONS.get(action, "")}

# 観点
{perspective}

# 親ノード
{parent_summary}

# 検索クエリ
{json.dumps(search_queries, ensure_ascii=False, indent=2)}

# 検索結果（重複削除済み）
{source_block}

# 出力してほしい内容
- このノードで扱う仮説または論点
- 検索結果から得られる根拠
- 反証・矛盾・限界
- 次に深掘りすべき問い
""".strip()


def build_query_planner_prompt(
    *,
    topic: str,
    action: ResearchAction,
    perspective: str,
    parent_state: NodeState | None,
    max_queries: int,
    max_words: int,
) -> str:
    if parent_state is None:
        parent_summary = "親ノードはありません。初回探索として、概要を素早く掴むための広めのキーワード群を作ってください。"
    else:
        parent_summary = json.dumps(
            {
                "action": parent_state.action,
                "perspective": parent_state.perspective,
                "text": parent_state.text,
                "open_questions": parent_state.open_questions,
                "findings": parent_state.findings,
                "search_queries": parent_state.search_queries,
                "score": parent_state.score,
            },
            ensure_ascii=False,
            indent=2,
        )

    return f"""
# 調査テーマ
{topic}

# 次の探索アクション
{action}: {ACTION_QUERY_INSTRUCTIONS.get(action, "")}

# 観点
{perspective}

# 親ノード
{parent_summary}

# 制約
- クエリ本数: 1〜{max_queries}本
- 1クエリの最大語数: {max_words}語
- 自然文は禁止。例: "GDPR 日本企業 DPO 導入" のような単語列にする。
- root(親なし)では、概要把握・最新動向・主要プレイヤーなどを必要な範囲で分ける。

# 出力JSON
{{"queries": ["単語 単語 単語"]}}
""".strip()


def build_review_prompt(
    *,
    topic: str,
    action: ResearchAction,
    perspective: str,
    text: str,
    sources: list[dict[str, Any]],
) -> str:
    return f"""
# 調査テーマ
{topic}

# アクション
{action}

# 観点
{perspective}

# 出典
{json.dumps(sources, ensure_ascii=False, indent=2)}

# レビュー対象の調査メモ
{text}
""".strip()


def build_final_report_prompt(
    topic: str,
    top_states: list[tuple[NodeState, float]],
) -> str:
    payload = [
        {
            **asdict(state),
            "node_score": node_score,
        }
        for state, node_score in top_states
    ]
    return f"""
# 調査テーマ
{topic}

# 探索で得た上位ノード
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}

上位ノードの重複を整理し、矛盾がある場合は矛盾として扱ってください。
""".strip()
