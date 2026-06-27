from __future__ import annotations

import json
import re
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


# モード(action)ごとに「どうクエリを作るか」という生成戦略そのものを変える。
# AB-MCTS-M から見ると各 action は独立した探索アクションであり、
# wider(new_angle) と deeper(deepen) で渡すコンテキストと方針を明確に分ける。
QUERY_STRATEGIES: dict[ResearchAction, str] = {
    "new_angle": (
        "目的: 既存ノードと重複しない未探索の角度を開拓する（幅を広げる）。\n"
        "- 「既出の観点」「既出のファセット」「避けるクエリ」を確認し、それらと語が重複しない新しいクエリを作る。\n"
        "- 指定ファセット・観点に沿って、まだ調べていない側面（別の主体・地域・時間軸・指標）を狙う。\n"
        "- 親ノードの主張は引き継がず、独立した切り口にする。"
    ),
    "deepen": (
        "目的: 親ノードの有望な論点を裏取りして深掘りする（深さを伸ばす）。\n"
        "- 「このアクションで最優先する問い」を、検証可能な具体的サブクエリへ分解する。\n"
        "- 親ノードの検索クエリと同じ語の繰り返しを避け、別の語で踏み込む。\n"
        "- 抽象語ではなく、固有名詞・年・指標名・統計・一次情報源など特定性の高い語を使う。"
    ),
    "criticize": (
        "目的: 親ノードの主要主張の妥当性を疑う（反証を探す）。\n"
        "- 親ノードの調査メモから主要な主張を1〜2個特定し、その反証・矛盾・古い情報・未検証の前提を探す語にする。\n"
        "- 「批判」一般ではなく、特定の主張をピンポイントで崩せる検索にする。"
    ),
    "revise": (
        "目的: 親ノードの自動レビュー指摘(findings)を解消する。\n"
        "- 各 finding を解消する具体的な追加根拠・具体事例・企業名・一次情報を狙う。\n"
        "- 指摘と無関係な一般クエリにしない。"
    ),
}


def query_planner_system_prompt(*, reference_date: date | None = None) -> str:
    ref = format_reference_date(reference_date)
    return f"""
あなたはDeepResearchの検索クエリ設計者です。
次の探索アクションで使うWeb検索クエリを、日本語または英語の短いキーワード列として設計してください。欲しい情報に応じて言語は選択してください。

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


# モード(action)ごとに調査メモで重視する成果を変える。
ACTION_RESEARCH_FOCUS: dict[ResearchAction, str] = {
    "new_angle": "既存の論点とは別の切り口で、新しい仮説・論点を立てることを最優先にする。",
    "deepen": "親ノードの論点を一次情報・具体的数値・事例で裏取りし、結論の確度を一段引き上げる。",
    "criticize": "親ノードの主張の反証・矛盾・古さ・未検証の前提を具体的に指摘する。賛同するだけの記述にしない。",
    "revise": "親ノードのレビュー指摘(findings)を一つずつ取り上げ、追加根拠でどう解消したかを明示する。",
}


def researcher_system_prompt(
    action: ResearchAction,
    perspective: str,
    *,
    reference_date: date | None = None,
) -> str:
    action_description = ACTION_DESCRIPTIONS.get(action, ACTION_DESCRIPTIONS["deepen"])
    action_focus = ACTION_RESEARCH_FOCUS.get(action, ACTION_RESEARCH_FOCUS["deepen"])
    ref = format_reference_date(reference_date)
    return f"""
あなたはDeepResearchチームの調査担当です。
現在の役割は「{perspective}」観点のリサーチャーです。
今回の探索アクションは「{action}」です: {action_description}

このアクションで重視すること:
{action_focus}

参照日（調査実行日）: {ref}
- 出典の日付・新旧を判断するときはこの日付を基準にしてください。

守ること:
- 検索結果に含まれる情報だけを根拠として扱う。
- 不確かな主張は不確かと明記する。
- 根拠、反証、残った不確実性を分けて書く。
- 最終回答ではなく、この探索ノードで得られた調査メモを作る。
- 日本語で簡潔に出力する。
""".strip()


# 採点スケールは全モード共通の [0,1]。観点だけをモード別に補足し、
# 甘い採点器に偏らないよう「成功の意味」は揃える（docs/AB-MCTS-M_SPEC.md 5.3 参照）。
ACTION_REVIEW_FOCUS: dict[ResearchAction, str] = {
    "new_angle": "既存ノードと重複しない新しい論点を実際に開拓できているか。",
    "deepen": "一次情報・具体的数値・事例で論点を実際に裏取りできているか。",
    "criticize": "本物の反証・矛盾・古い情報を具体的に指摘できているか（言い換えだけは不可）。",
    "revise": "親ノードのレビュー指摘を実際に解消できているか。",
}


def reviewer_system_prompt(
    action: ResearchAction | None = None,
    *,
    reference_date: date | None = None,
) -> str:
    ref = format_reference_date(reference_date)
    focus = ACTION_REVIEW_FOCUS.get(action or "", "")
    focus_block = (
        f"\nこのアクションで特に確認する観点（採点スケールは変えない）:\n- {focus}\n"
        if focus
        else ""
    )
    return f"""
あなたはDeepResearchの厳格な自動レビュアーです。
調査メモに対して、根拠不足、古い情報、矛盾、出典の弱さ、結論の飛躍を探してください。

参照日（調査実行日）: {ref}
- 出典の日付が参照日以前であれば、それは過去または当日の情報です。「未来の日付」や「重大な誤記」として指摘しないでください。
- 「未来の日付」とは参照日より後の日付のみを指します。
{focus_block}
必ず次のJSONオブジェクトだけを返してください。
{{
  "score": 0.0から1.0の数値,
  "summary": "短い総評",
  "findings": ["修正または再調査すべき指摘", "..."]
}}

採点基準（全アクション共通。アクションによってスコアの意味を変えない）:
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


def _text_excerpt(text: str, *, max_chars: int = 600) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "…"


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
    covered_facets: list[str] | None = None,
    covered_perspectives: list[str] | None = None,
    reference_date: date | None = None,
) -> str:
    strategy = QUERY_STRATEGIES.get(action, QUERY_STRATEGIES["deepen"])
    sections: list[str] = [
        reference_date_block(reference_date),
        f"# 調査テーマ\n{topic}",
        f"# 次の探索アクション\n{action}",
        f"# このアクションのクエリ生成戦略\n{strategy}",
        f"# 観点\n{perspective}",
        f"# ファセット\n{facet or '未指定'}",
    ]

    # モードごとに渡すコンテキストを変える。
    if action == "new_angle" or parent_state is None:
        sections.append(
            "# 既出の観点（重複回避用）\n"
            + json.dumps(covered_perspectives or [], ensure_ascii=False)
        )
        sections.append(
            "# 既出のファセット（重複回避用）\n"
            + json.dumps(covered_facets or [], ensure_ascii=False)
        )
    if action == "deepen":
        sections.append(f"# このアクションで最優先する問い\n{focus_question or '未指定'}")
        sections.append(
            "# 親ノードの open_questions\n"
            + json.dumps(
                parent_state.open_questions if parent_state else [],
                ensure_ascii=False,
                indent=2,
            )
        )
    if action in ("deepen", "criticize") and parent_state is not None:
        sections.append(
            "# 親ノードの調査メモ（抜粋）\n" + _text_excerpt(parent_state.text)
        )
    if action == "revise":
        sections.append(
            "# 親ノードの reviewer findings\n"
            + json.dumps(
                parent_state.findings if parent_state else [],
                ensure_ascii=False,
                indent=2,
            )
        )

    if parent_state is not None:
        parent_summary = json.dumps(
            {
                "action": parent_state.action,
                "perspective": parent_state.perspective,
                "facet": parent_state.facet,
                "search_queries": parent_state.search_queries,
                "score": parent_state.score,
            },
            ensure_ascii=False,
            indent=2,
        )
    else:
        parent_summary = "親ノードはありません。初回探索として、概要を素早く掴むための広めのキーワード群を作ってください。"
    sections.append(f"# 親ノード\n{parent_summary}")

    sections.append(
        "# 避けるクエリ（これらと同一・酷似のクエリは禁止）\n"
        + json.dumps(avoid_queries, ensure_ascii=False, indent=2)
    )
    sections.append(
        "# 制約\n"
        f"- クエリ本数: 1〜{max_queries}本\n"
        f"- 1クエリの最大語数: {max_words}語\n"
        '- 自然文は禁止。例: "単語 単語 単語 単語" のような単語列にする。\n'
        "- 上記「クエリ生成戦略」に厳密に従う。\n"
        "- 「避けるクエリ」と同一・類似のクエリは禁止。語の並べ替えだけの言い換えも禁止。"
    )
    sections.append('# 出力JSON\n{"queries": ["単語 単語 単語"]}')

    return "\n\n".join(sections).strip()


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
