from __future__ import annotations

import html
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
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
    facet: str | None = None
    search_query: str | None = None
    search_queries: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    review_text: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def choose_perspective(
    parent_state: NodeState | None,
    action: ResearchAction,
    perspective_counts: dict[str, int] | None = None,
) -> str:
    # wider(new_angle)/root では最小使用回数の観点を選び、観点が偏らないようにする。
    # deepen/criticize/revise は親観点を継承し、論点の連続性を保つ。
    if parent_state is None or action == "new_angle":
        counts = perspective_counts or {}
        return min(
            PERSPECTIVES,
            key=lambda perspective: (counts.get(perspective, 0), PERSPECTIVES.index(perspective)),
        )
    return parent_state.perspective


def _trim_query_words(query: str, max_words: int) -> str:
    return " ".join(query.strip().split()[:max_words])


def parse_query_plan(
    text: str,
    *,
    max_queries: int,
    max_words: int,
    existing_queries: list[str] | None = None,
    similarity_threshold: float = 1.0,
) -> list[str]:
    data = parse_json_object(text)
    if data is None:
        return []

    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list):
        return []

    existing_token_sets = [_query_tokens(query) for query in (existing_queries or [])]
    queries: list[str] = []
    seen: set[str] = set()
    accepted_token_sets: list[frozenset[str]] = []
    for item in raw_queries:
        query = _trim_query_words(str(item), max_words=max_words)
        if not query or query in seen:
            continue
        tokens = _query_tokens(query)
        # 既存クエリや既に採用済みのクエリと酷似する場合は除外し、近似重複を防ぐ。
        if similarity_threshold < 1.0 and tokens:
            candidates = existing_token_sets + accepted_token_sets
            if max_query_jaccard(tokens, candidates) >= similarity_threshold:
                continue
        seen.add(query)
        queries.append(query)
        accepted_token_sets.append(tokens)
        if len(queries) >= max_queries:
            break
    return queries


def parse_facets(text: str, *, num_facets: int) -> list[str]:
    data = parse_json_object(text)
    if data is None:
        return []

    raw_facets = data.get("facets", [])
    if not isinstance(raw_facets, list):
        return []

    facets: list[str] = []
    seen: set[str] = set()
    for item in raw_facets:
        facet = str(item).strip()
        if not facet or facet in seen:
            continue
        facets.append(facet)
        seen.add(facet)
        if len(facets) >= max(1, num_facets):
            break
    return facets


def choose_facet(facets: list[str], facet_counts: dict[str, int]) -> str | None:
    if not facets:
        return None
    return min(facets, key=lambda facet: (facet_counts.get(facet, 0), facets.index(facet)))


def _short_topic_words(topic: str, *, max_words: int) -> str:
    compact = re.sub(r"[、。・/（）()「」『』,.;:]+", " ", topic)
    return _trim_query_words(compact, max_words=max_words)


def build_fallback_queries(
    topic: str,
    action: ResearchAction,
    parent_state: NodeState | None,
    perspective: str,
    *,
    max_queries: int,
    max_words: int,
) -> list[str]:
    topic_words = _short_topic_words(topic, max_words=max(1, max_words - 2))
    if parent_state is None:
        candidates = [
            f"{topic_words} {perspective}",
            f"{topic_words} 最新 動向",
            f"{topic_words} 主要 企業",
        ]
    elif action == "criticize":
        candidates = [f"{topic_words} 反証 課題 矛盾"]
    elif action == "revise" and parent_state.findings:
        finding_words = _short_topic_words(
            " ".join(parent_state.findings[:2]),
            max_words=max(1, max_words - 2),
        )
        candidates = [f"{topic_words} {finding_words} 検証"]
    elif action == "new_angle":
        candidates = [f"{topic_words} {perspective} 比較"]
    elif parent_state.open_questions:
        question_words = _short_topic_words(
            parent_state.open_questions[0],
            max_words=max(1, max_words - 2),
        )
        candidates = [f"{topic_words} {question_words}"]
    else:
        candidates = [f"{topic_words} 深掘り 根拠"]

    queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        query = _trim_query_words(candidate, max_words=max_words)
        if query and query not in seen:
            queries.append(query)
            seen.add(query)
        if len(queries) >= max_queries:
            break
    return queries


def extract_open_questions(text: str) -> list[str]:
    lines = text.splitlines()
    in_section = False
    questions: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#{2,}\s*(?:\d+\.\s*)?次に(?:深掘り|調べる)すべき問い", stripped):
            in_section = True
            continue
        if in_section and stripped.startswith("##"):
            break
        if not in_section or not stripped:
            continue
        question = re.sub(r"^(?:[-*]\s*|\d+[.)]\s*)", "", stripped).strip()
        if question:
            questions.append(question)
    return questions


def _source_similarity_text(source: dict[str, Any]) -> str:
    text = f"{source.get('title', '')} {source.get('snippet', '')}".lower()
    return re.sub(r"\s+", " ", text).strip()


def _append_matched_query(source: dict[str, Any], query: str | None) -> None:
    if not query:
        return
    matched = source.setdefault("matched_queries", [])
    if isinstance(matched, list) and query not in matched:
        matched.append(query)


def merge_and_dedupe_sources(
    per_query_results: list[list[dict[str, Any]]],
    *,
    similarity_threshold: float = 0.9,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_urls: dict[str, dict[str, Any]] = {}

    for results in per_query_results:
        for raw_source in results:
            source = dict(raw_source)
            query = str(source.get("query", "")).strip() or None
            url = str(source.get("url", "")).strip()
            if url and url in seen_urls:
                _append_matched_query(seen_urls[url], query)
                continue

            text = _source_similarity_text(source)
            similar_source = next(
                (
                    existing
                    for existing in merged
                    if text
                    and SequenceMatcher(
                        None,
                        text,
                        _source_similarity_text(existing),
                    ).ratio()
                    > similarity_threshold
                ),
                None,
            )
            if similar_source is not None:
                _append_matched_query(similar_source, query)
                continue

            if query:
                source["matched_queries"] = [query]
            source.pop("query_index", None)
            source.pop("query", None)
            merged.append(source)
            if url:
                seen_urls[url] = source

    for index, source in enumerate(merged, start=1):
        source["position"] = index
    return merged


def _signature_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _query_tokens(query: str) -> frozenset[str]:
    normalized = re.sub(r"[、。・/（）()「」『』,.;:]+", " ", query.lower())
    return frozenset(token for token in normalized.split() if token)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def query_token_signature(queries: list[str]) -> frozenset[str]:
    tokens: set[str] = set()
    for query in queries:
        tokens |= _query_tokens(query)
    return frozenset(tokens)


def max_query_jaccard(
    signature: frozenset[str], seen_signatures: list[frozenset[str]]
) -> float:
    return max(
        (_jaccard(signature, existing) for existing in seen_signatures),
        default=0.0,
    )


def query_set_signature(queries: list[str]) -> frozenset[str]:
    return frozenset(_signature_text(query) for query in queries if query.strip())


def source_url_signature(sources: list[dict[str, Any]]) -> frozenset[str]:
    urls: set[str] = set()
    for source in sources:
        url = str(source.get("url", "")).strip().rstrip("/")
        if url:
            urls.add(url)
    return frozenset(urls)


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
    repetition_penalty: float = 0.0,
    coverage_bonus: float = 0.0,
) -> float:
    # findings ペナルティは上限を抑え、良質ノードのスコア差が 0.2 付近に潰れないようにする。
    penalty = min(0.24, 0.03 * len(findings))
    # 検索品質の減点は一本化し、検索失敗(dummy)時に 0-source と !success を二重に課さない。
    if not search_success:
        penalty += 0.10
    elif num_sources == 0:
        penalty += 0.20
    return round(clamp01(base_score - penalty - repetition_penalty + coverage_bonus), 6)


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
    queries = "<br>".join(html.escape(item) for item in state.search_queries[:5])
    questions = "<br>".join(html.escape(item) for item in state.open_questions[:5])
    return (
        f"<b>action:</b> {html.escape(state.action)}<br>"
        f"<b>perspective:</b> {html.escape(state.perspective)}<br>"
        f"<b>facet:</b> {html.escape(state.facet or '')}<br>"
        f"<b>score:</b> {state.score:.3f}<br>"
        f"<b>depth:</b> {state.depth}<br>"
        f"<b>queries:</b><br>{queries}<br>"
        f"<b>open_questions:</b><br>{questions}<br>"
        f"<b>text:</b> {html.escape(state.text[:500])}<br>"
        f"<b>findings:</b><br>{findings}"
    )
