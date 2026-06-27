from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from typing import Any

from utils import NodeState, ResearchAction


def format_reference_date(reference_date: date | None = None) -> str:
    d = reference_date or date.today()
    weekdays = "月火水木金土日"
    weekday = weekdays[d.weekday()]
    return f"{d.year}年{d.month}月{d.day}日（{weekday}）"


def reference_date_block(reference_date: date | None = None) -> str:
    d = reference_date or date.today()
    formatted = format_reference_date(d)
    return (
        f"# 参照日\n"
        f"{formatted}（ISO: {d.isoformat()}）\n"
        f"- 出典の日付・新旧・未来/過去の判断は、この参照日を基準にしてください。\n"
        f"- 参照日以前の日付は過去または当日の情報であり、「未来の日付」ではありません。"
    )


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


def query_planner_system_prompt(*, reference_date: date | None = None) -> str:
    ref = format_reference_date(reference_date)
    return f"""
あなたはDeepResearchの検索クエリ設計者です。
次の探索アクションで使うWeb検索クエリを、短いキーワード列として設計してください。

参照日（調査実行日）: {ref}
- 「最新」「直近」などの判断はこの日付を基準にしてください。

守ること:
- 必ずJSONオブジェクトだけを返す。
- 形式は {{"queries": ["単語 単語 単語", "..."]}}。
- クエリは自然文や質問文にしない。
- 各クエリは空白区切りで最大語数以内にする。
- 必要な本数だけ返し、最大本数を無理に埋めない。
- 同じ意味のクエリを重複させない。
- 「避けるクエリ」と同一または酷似したクエリを返さない。
- 日本語に限定しない。企業名、英語の規制名、Privacy Enhancing Technologies、PETs、ZKP、Federated Learning など、検索精度が上がる固有名詞や英語キーワードを積極的に使ってよい。
""".strip()


def topic_decomposition_system_prompt(*, reference_date: date | None = None) -> str:
    ref = format_reference_date(reference_date)
    return f"""
あなたはDeepResearchの調査設計者です。
調査テーマを、探索で網羅すべき独立したファセットに分解してください。

参照日（調査実行日）: {ref}

守ること:
- 必ずJSONオブジェクトだけを返す。
- 形式は {{"facets": ["ファセット", "..."]}}。
- 各ファセットは短く、検索観点として使える粒度にする。
- テーマの前半・後半のどちらかに偏らず、主要な論点を網羅する。
""".strip()


def researcher_system_prompt(
    action: ResearchAction,
    perspective: str,
    *,
    reference_date: date | None = None,
) -> str:
    action_description = ACTION_DESCRIPTIONS.get(action, ACTION_DESCRIPTIONS["deepen"])
    ref = format_reference_date(reference_date)
    return f"""
あなたはDeepResearchチームの調査担当です。
現在の役割は「{perspective}」観点のリサーチャーです。
今回の探索アクションは「{action}」です: {action_description}

参照日（調査実行日）: {ref}
- 出典の日付・新旧を判断するときはこの日付を基準にしてください。

守ること:
- 検索結果に含まれる情報だけを根拠として扱う。
- 不確かな主張は不確かと明記する。
- 根拠、反証、残った不確実性を分けて書く。
- 最終回答ではなく、この探索ノードで得られた調査メモを作る。
- 日本語で簡潔に出力する。
""".strip()


def reviewer_system_prompt(*, reference_date: date | None = None) -> str:
    ref = format_reference_date(reference_date)
    return f"""
あなたはDeepResearchの厳格な自動レビュアーです。
調査メモに対して、根拠不足、古い情報、矛盾、出典の弱さ、結論の飛躍を探してください。

参照日（調査実行日）: {ref}
- 出典の日付が参照日以前であれば、それは過去または当日の情報です。「未来の日付」や「重大な誤記」として指摘しないでください。
- 「未来の日付」とは参照日より後の日付のみを指します。

必ず次のJSONオブジェクトだけを返してください。
{{
  "score": 0.0から1.0の数値,
  "summary": "短い総評",
  "findings": ["修正または再調査すべき指摘", "..."]
}}

採点基準:
- 1.0: 強い根拠が複数あり、矛盾と限界も扱えている
- 0.7: 実用的だが、補強すべき点が残る
- 0.4: 重要な根拠不足または飛躍がある
- 0.0: 調査として使えない
""".strip()


def final_report_system_prompt(*, reference_date: date | None = None) -> str:
    ref = format_reference_date(reference_date)
    return f"""
あなたはDeepResearchの編集長です。
探索済みノードのうち有望な論点だけを統合し、根拠と不確実性が分かる最終レポートを作成してください。

参照日（調査実行日）: {ref}
- 出典の日付・新旧を判断するときはこの日付を基準にしてください。

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
    facet: str | None,
    search_queries: list[str],
    sources: list[dict[str, Any]],
    parent_state: NodeState | None,
    reference_date: date | None = None,
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
{reference_date_block(reference_date)}

# 調査テーマ
{topic}

# アクション
{action}: {ACTION_DESCRIPTIONS.get(action, "")}

# 観点
{perspective}

# ファセット
{facet or "未指定"}

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
    facet: str | None,
    parent_state: NodeState | None,
    avoid_queries: list[str],
    focus_question: str | None,
    max_queries: int,
    max_words: int,
    reference_date: date | None = None,
) -> str:
    if parent_state is None:
        parent_summary = "親ノードはありません。初回探索として、概要を素早く掴むための広めのキーワード群を作ってください。"
        parent_open_questions: list[str] = []
        parent_findings: list[str] = []
    else:
        parent_summary = json.dumps(
            {
                "action": parent_state.action,
                "perspective": parent_state.perspective,
                "facet": parent_state.facet,
                "text": parent_state.text,
                "search_queries": parent_state.search_queries,
                "score": parent_state.score,
            },
            ensure_ascii=False,
            indent=2,
        )
        parent_open_questions = parent_state.open_questions
        parent_findings = parent_state.findings

    avoid_block = json.dumps(avoid_queries, ensure_ascii=False, indent=2)
    open_questions_block = json.dumps(parent_open_questions, ensure_ascii=False, indent=2)
    findings_block = json.dumps(parent_findings, ensure_ascii=False, indent=2)

    return f"""
{reference_date_block(reference_date)}

# 調査テーマ
{topic}

# 次の探索アクション
{action}: {ACTION_QUERY_INSTRUCTIONS.get(action, "")}

# 観点
{perspective}

# ファセット
{facet or "未指定"}

# このアクションで最優先する問い
{focus_question or "未指定"}

# 親ノードの open_questions
{open_questions_block}

# 親ノードの reviewer findings
{findings_block}

# 避けるクエリ
{avoid_block}

# 親ノード
{parent_summary}

# 制約
- クエリ本数: 1〜{max_queries}本
- 1クエリの最大語数: {max_words}語
- 自然文は禁止。例: "単語 単語 単語 単語" のような単語列にする。
- root(親なし)または new_angle では、指定ファセットに沿って未探索の角度を広げる。
- deepen では「このアクションで最優先する問い」を具体的に検証できる検索語へ分解する。
- revise では reviewer findings を解消する追加根拠・具体例・企業名を探す。
- criticize では親ノードの主張の反証、弱点、古い情報、未検証の前提を探す。
- 「避けるクエリ」と同一・類似のクエリは禁止。

# 出力JSON
{{"queries": ["単語 単語 単語"]}}
""".strip()


def build_topic_decomposition_prompt(
    topic: str,
    *,
    num_facets: int,
    reference_date: date | None = None,
) -> str:
    return f"""
{reference_date_block(reference_date)}

# 調査テーマ
{topic}

# 制約
- ファセット数: 最大 {num_facets} 個
- 各ファセットは短い名詞句にする。
- EU側の規制現状と、日本企業の具体的対応の両方を含める。

# 出力JSON
{{"facets": ["EU規制の現状", "日本企業の具体的対応"]}}
""".strip()


def build_review_prompt(
    *,
    topic: str,
    action: ResearchAction,
    perspective: str,
    text: str,
    sources: list[dict[str, Any]],
    reference_date: date | None = None,
) -> str:
    return f"""
{reference_date_block(reference_date)}

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


def build_final_review_prompt(
    *,
    topic: str,
    report: str,
    reference_date: date | None = None,
) -> str:
    return f"""
{reference_date_block(reference_date)}

# 調査テーマ
{topic}

# 最終レポート
{report}
""".strip()


def build_final_report_prompt(
    topic: str,
    top_states: list[tuple[NodeState, float]],
    *,
    reference_date: date | None = None,
) -> str:
    payload = [
        {
            **asdict(state),
            "node_score": node_score,
        }
        for state, node_score in top_states
    ]
    return f"""
{reference_date_block(reference_date)}

# 調査テーマ
{topic}

# 探索で得た上位ノード
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}

上位ノードの重複を整理し、矛盾がある場合は矛盾として扱ってください。
""".strip()
