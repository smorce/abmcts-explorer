from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logging_utils import ResearchLogger
from prompt import (
    build_review_prompt,
    format_reference_date,
    reference_date_block,
    reviewer_system_prompt,
)
from run import generate_fn
from utils import (
    NodeState,
    dummy_web_search,
    extract_open_questions,
    merge_and_dedupe_sources,
    parse_query_plan,
    parse_review_payload,
    score_review,
)


def test_dummy_web_search_returns_structured_sources() -> None:
    results = dummy_web_search("ローカルLLM 調査", limit=2)

    assert len(results) == 2
    assert results[0]["title"]
    assert results[0]["url"].startswith("https://example.invalid/")
    assert results[0]["provider"] == "dummy"


def test_parse_review_payload_and_score_penalty() -> None:
    review = '{"score": 0.8, "summary": "概ね良い", "findings": ["根拠不足", "矛盾あり"]}'

    base_score, findings, summary = parse_review_payload(review)
    score = score_review(base_score, findings, num_sources=3, search_success=True)

    assert summary == "概ね良い"
    assert findings == ["根拠不足", "矛盾あり"]
    assert score == 0.72


def test_parse_query_plan_limits_words_and_queries() -> None:
    payload = json.dumps(
        {
            "queries": [
                "GDPR 日本企業 DPO 導入 事例 追加語",
                "GDPR 日本企業 DPO 導入 事例 追加語",
                "CRA 日本企業 対応",
            ]
        },
        ensure_ascii=False,
    )

    queries = parse_query_plan(payload, max_queries=2, max_words=5)

    assert queries == ["GDPR 日本企業 DPO 導入 事例", "CRA 日本企業 対応"]


def test_extract_open_questions_from_markdown() -> None:
    text = """
## 反証・矛盾・限界
本文

## 次に深掘りすべき問い
1. 日本企業のDPO設置状況は？
- CMP導入率は？

## 別見出し
除外
"""

    questions = extract_open_questions(text)

    assert questions == ["日本企業のDPO設置状況は？", "CMP導入率は？"]


def test_merge_and_dedupe_sources_removes_url_and_similar_items() -> None:
    per_query_results = [
        [
            {
                "title": "GDPR対応 日本企業 事例",
                "url": "https://example.com/a",
                "snippet": "日本企業のGDPR対応事例を紹介する記事です。",
                "provider": "fake",
                "query": "GDPR 日本企業 事例",
                "query_index": 1,
            },
            {
                "title": "EU AI法 対応",
                "url": "https://example.com/b",
                "snippet": "EU AI法への対応を解説します。",
                "provider": "fake",
                "query": "EU AI法 日本企業",
                "query_index": 1,
            },
        ],
        [
            {
                "title": "GDPR対応 日本企業 事例",
                "url": "https://example.com/a",
                "snippet": "日本企業のGDPR対応事例を紹介する記事です。",
                "provider": "fake",
                "query": "DPO 日本企業",
                "query_index": 2,
            },
            {
                "title": "GDPR対応 日本企業 事例集",
                "url": "https://example.com/c",
                "snippet": "日本企業のGDPR対応事例を紹介する記事です。",
                "provider": "fake",
                "query": "GDPR 事例集",
                "query_index": 2,
            },
        ],
    ]

    sources = merge_and_dedupe_sources(per_query_results)

    assert [source["url"] for source in sources] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert sources[0]["matched_queries"] == [
        "GDPR 日本企業 事例",
        "DPO 日本企業",
        "GDPR 事例集",
    ]
    assert [source["position"] for source in sources] == [1, 2]


def test_research_logger_writes_jsonl(tmp_path: Path) -> None:
    logger = ResearchLogger(tmp_path)
    state = NodeState(
        topic="テスト",
        action="new_angle",
        perspective="技術",
        text="調査メモ",
        sources=[],
        findings=[],
        eval_results=[],
        score=0.5,
    )

    logger.log_event("node_generated", node_id=state.node_id, score=state.score)
    logger.log_node(state)
    logger.log_edge(
        parent_id=None,
        child_id=state.node_id,
        action=state.action,
        score_delta=None,
    )
    logger.log_llm_call(
        node_id=state.node_id,
        action=state.action,
        perspective=state.perspective,
        role="researcher",
        system="system",
        user="user",
        response="response",
    )

    event = json.loads((tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    node = json.loads((tmp_path / "tree" / "nodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    edge = json.loads((tmp_path / "tree" / "edges.jsonl").read_text(encoding="utf-8").splitlines()[0])
    llm_call = json.loads((tmp_path / "llm_io" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert event["event"] == "node_generated"
    assert node["node_id"] == state.node_id
    assert edge["child_id"] == state.node_id
    assert llm_call["response"] == "response"


def test_generate_fn_uses_mocked_llm_and_search(monkeypatch, tmp_path: Path) -> None:
    logger = ResearchLogger(tmp_path)

    def fake_web_search(query: str, max_results: int):
        return [
            {
                "title": "出典",
                "url": "https://example.com",
                "snippet": "根拠",
                "provider": "fake",
            }
        ], True

    def fake_llm(**kwargs):
        if kwargs["role"] == "planner":
            return '{"queries": ["GDPR 日本企業 DPO", "CMP 導入 事例"]}'
        if kwargs["role"] == "reviewer":
            return '{"score": 0.9, "summary": "良い", "findings": []}'
        return """
検索結果に基づく調査メモ

## 次に深掘りすべき問い
1. 日本企業のDPO設置状況は？
""".strip()

    monkeypatch.setattr("run.web_search", fake_web_search)
    monkeypatch.setattr("run.call_local_llm", fake_llm)

    state, score = generate_fn(
        None,
        action="new_angle",
        topic="ローカルLLMの活用",
        temperature=0.3,
        max_tokens=1200,
        max_results=3,
        max_queries=3,
        max_query_words=6,
        research_logger=logger,
    )

    assert state.action == "new_angle"
    assert "検索結果に基づく調査メモ" in state.text
    assert state.search_queries == ["GDPR 日本企業 DPO", "CMP 導入 事例"]
    assert state.open_questions == ["日本企業のDPO設置状況は？"]
    assert score == 0.9
    assert state.score == score


def test_prompts_include_reference_date() -> None:
    ref = date(2026, 5, 25)

    assert format_reference_date(ref) == "2026年5月25日（月）"
    assert "2026-05-25" in reference_date_block(ref)
    assert "未来の日付" in reference_date_block(ref)

    review_user = build_review_prompt(
        topic="テスト",
        action="deepen",
        perspective="技術",
        text="調査メモ",
        sources=[],
        reference_date=ref,
    )
    assert "2026年5月25日（月）" in review_user

    review_system = reviewer_system_prompt(reference_date=ref)
    assert "2026年5月25日（月）" in review_system
    assert "参照日より後" in review_system
