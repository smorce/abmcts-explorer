from __future__ import annotations

import html
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import treequest as tq

from ab_mcts_arc2.eval_result import EvalResultWithScore
from ab_mcts_arc2.web_search import SearXNGSearchClient


ResearchAction = str
DEFAULT_ACTIONS: tuple[ResearchAction, ...] = (
    "new_angle",
    "deepen",
    "criticize",
    "revise",
)
PERSPECTIVES: tuple[str, ...] = ("市場", "技術", "戦略", "リスク", "実装")


@dataclass
class NodeState:
    topic: str
    action: ResearchAction
    perspective: str
    text: str
    sources: list[dict[str, Any]]
    findings: list[str]
    eval_results: list[EvalResultWithScore]
    score: float
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    depth: int = 1
    search_query: str | None = None
    review_text: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def choose_perspective(parent_state: NodeState | None, action: ResearchAction) -> str:
    if parent_state is None or action == "new_angle":
        depth = 0 if parent_state is None else parent_state.depth
        return PERSPECTIVES[depth % len(PERSPECTIVES)]
    return parent_state.perspective


def build_search_query(
    topic: str,
    action: ResearchAction,
    parent_state: NodeState | None,
    perspective: str,
) -> str:
    if parent_state is None:
        return f"{topic} {perspective} 最新 動向 根拠"
    if action == "criticize":
        return f"{topic} {parent_state.text[:80]} 反証 課題 矛盾"
    if action == "revise" and parent_state.findings:
        return f"{topic} {' '.join(parent_state.findings[:2])} 追加根拠 検証"
    if action == "new_angle":
        return f"{topic} {perspective} 別観点 比較"
    return f"{topic} {parent_state.text[:80]} 深掘り 根拠"


def dummy_web_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index in range(max(1, limit)):
        results.append(
            {
                "title": f"Dummy source {index + 1}: {query}",
                "url": f"https://example.invalid/search/{index + 1}",
                "snippet": (
                    "SearXNG が利用できない場合のダミー検索結果です。"
                    "実運用では SEARXNG_URL を設定して実検索に置き換えます。"
                ),
                "position": index + 1,
                "provider": "dummy",
            }
        )
    return results


def web_search(query: str, max_results: int = 5) -> tuple[list[dict[str, Any]], bool]:
    client = SearXNGSearchClient()
    if not client.is_available():
        return dummy_web_search(query, max_results), False

    response = client.search(query, limit=max_results)
    if not response.success or not response.results:
        return dummy_web_search(query, max_results), False

    return [
        {
            "title": result.title,
            "url": result.url,
            "snippet": result.description,
            "position": result.position,
            "provider": client.name,
        }
        for result in response.results
    ], True


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_review_payload(text: str) -> tuple[float, list[str], str]:
    data = parse_json_object(text)
    if data is None:
        return 0.45, ["自動レビューがJSONとして解釈できませんでした。"], text

    raw_score = data.get("score", 0.5)
    try:
        base_score = float(raw_score)
    except (TypeError, ValueError):
        base_score = 0.5

    raw_findings = data.get("findings", [])
    if isinstance(raw_findings, list):
        findings = [str(item) for item in raw_findings if str(item).strip()]
    elif raw_findings:
        findings = [str(raw_findings)]
    else:
        findings = []

    summary = str(data.get("summary", text))
    return clamp01(base_score), findings, summary


def score_review(
    base_score: float,
    findings: list[str],
    *,
    num_sources: int,
    search_success: bool,
) -> float:
    penalty = min(0.30, 0.04 * len(findings))
    if num_sources == 0:
        penalty += 0.20
    if not search_success:
        penalty += 0.05
    return round(clamp01(base_score - penalty), 6)


def make_eval_results(score: float, findings: list[str]) -> list[EvalResultWithScore]:
    reason = "\n".join(findings) if findings else "重大な指摘はありません。"
    return [EvalResultWithScore(score=score, reason=reason)]


def get_top_k(state: Any, algorithm: Any, k: int) -> list[tuple[NodeState, float]]:
    if hasattr(state, "tree"):
        nodes = state.tree.get_nodes()
        nodes.sort(key=lambda node: node.score, reverse=True)
        return [(node.state, node.score) for node in nodes[:k]]
    return tq.top_k(state, algorithm, k=k)


def state_formatter_html(state: NodeState) -> str:
    findings = "<br>".join(html.escape(item) for item in state.findings[:5])
    return (
        f"<b>action:</b> {html.escape(state.action)}<br>"
        f"<b>perspective:</b> {html.escape(state.perspective)}<br>"
        f"<b>score:</b> {state.score:.3f}<br>"
        f"<b>depth:</b> {state.depth}<br>"
        f"<b>query:</b> {html.escape(state.search_query or '')}<br>"
        f"<b>text:</b> {html.escape(state.text[:500])}<br>"
        f"<b>findings:</b><br>{findings}"
    )
