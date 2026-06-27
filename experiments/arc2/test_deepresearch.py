from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logging_utils import ResearchLogger
from run import generate_fn
from utils import (
    NodeState,
    dummy_web_search,
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
        if kwargs["role"] == "reviewer":
            return '{"score": 0.9, "summary": "良い", "findings": []}'
        return "検索結果に基づく調査メモ"

    monkeypatch.setattr("run.web_search", fake_web_search)
    monkeypatch.setattr("run.call_local_llm", fake_llm)

    state, score = generate_fn(
        None,
        action="new_angle",
        topic="ローカルLLMの活用",
        temperature=0.3,
        max_tokens=1200,
        max_results=3,
        research_logger=logger,
    )

    assert state.action == "new_angle"
    assert state.text == "検索結果に基づく調査メモ"
    assert score == 0.9
    assert state.score == score
