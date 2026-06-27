from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import treequest as tq
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ab_mcts_arc2.llama_server import (
    LlamaServerEnvConfig,
    messages_to_prompt,
    run_llama_server_completion_sync,
)

sys.path.insert(0, str(Path(__file__).parent))
from logging_utils import ResearchLogger
from prompt import (
    build_final_report_prompt,
    build_final_review_prompt,
    build_query_planner_prompt,
    build_research_prompt,
    build_review_prompt,
    build_topic_decomposition_prompt,
    final_report_system_prompt,
    query_planner_system_prompt,
    researcher_system_prompt,
    topic_decomposition_system_prompt,
    reviewer_system_prompt,
)
from utils import (
    DEFAULT_ACTIONS,
    NodeState,
    build_fallback_queries,
    choose_perspective,
    choose_facet,
    clamp01,
    extract_open_questions,
    get_top_k,
    make_eval_results,
    merge_and_dedupe_sources,
    parse_facets,
    parse_query_plan,
    parse_review_payload,
    query_set_signature,
    score_review,
    source_url_signature,
    state_formatter_html,
    web_search,
)


logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def apply_env_defaults(cfg: DictConfig) -> None:
    llm_cfg = cfg.get("llm", {})
    search_cfg = cfg.get("search", {})
    defaults = {
        "LLAMA_SERVER_BASE_URL": _cfg_get(llm_cfg, "base_url"),
        "LLM_MODEL": _cfg_get(llm_cfg, "model"),
        "LLAMA_SERVER_ENABLE_THINKING": str(
            _cfg_get(llm_cfg, "enable_thinking", False)
        ).lower(),
        "LLAMA_SERVER_TEMPERATURE": str(_cfg_get(llm_cfg, "temperature", 0.3)),
        "LLAMA_SERVER_MAX_TOKENS": str(_cfg_get(llm_cfg, "max_tokens", 1200)),
        "SEARXNG_URL": _cfg_get(search_cfg, "url"),
        "SEARXNG_ENGINE": _cfg_get(search_cfg, "engines"),
        "SEARXNG_LANGUAGE": _cfg_get(search_cfg, "language"),
    }
    for key, value in defaults.items():
        if value is not None:
            os.environ.setdefault(key, str(value))


def _fallback_facets(topic: str, *, num_facets: int) -> list[str]:
    candidates = [
        part.strip()
        for part in topic.replace("、", "と").replace("および", "と").split("と")
        if part.strip()
    ]
    if len(candidates) < 2:
        candidates.extend(["EU規制の現状", "日本企業の具体的対応"])

    facets: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        facets.append(candidate)
        seen.add(candidate)
        if len(facets) >= max(1, num_facets):
            break
    return facets


def _select_facet(
    *,
    parent_state: NodeState | None,
    action: str,
    facets: list[str],
    facet_counts: dict[str, int],
) -> str | None:
    if parent_state is not None and action != "new_angle" and parent_state.facet:
        return parent_state.facet
    return choose_facet(facets, facet_counts)


def _select_focus_question(
    parent_state: NodeState | None,
    used_open_questions: set[str],
) -> str | None:
    if parent_state is None:
        return None
    for question in parent_state.open_questions:
        if question not in used_open_questions:
            return question
    return parent_state.open_questions[0] if parent_state.open_questions else None


def _jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _is_repeated_sources(
    url_sig: frozenset[str],
    seen_url_sigs: list[frozenset[str]],
    *,
    threshold: float = 0.8,
) -> bool:
    return any(_jaccard_similarity(url_sig, seen) >= threshold for seen in seen_url_sigs)


def call_local_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    research_logger: ResearchLogger | None,
    node_id: str | None,
    action: str,
    perspective: str,
    role: str,
) -> str:
    config = LlamaServerEnvConfig.from_env().with_overrides(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    prompt = messages_to_prompt(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    response = run_llama_server_completion_sync(config, prompt)
    if research_logger is not None:
        research_logger.log_llm_call(
            node_id=node_id,
            action=action,
            perspective=perspective,
            role=role,
            system=system_prompt,
            user=user_prompt,
            response=response,
        )
    return response


def generate_fn(
    parent_state: NodeState | None,
    *,
    action: str,
    topic: str,
    temperature: float,
    planner_temperature: float,
    max_tokens: int,
    max_results: int,
    max_queries: int,
    max_query_words: int,
    facets: list[str],
    exploration_state: dict[str, Any],
    novelty_penalty_weight: float,
    coverage_bonus: float,
    research_logger: ResearchLogger | None = None,
) -> tuple[NodeState, float]:
    start_time = time.time()
    max_queries = max(1, max_queries)
    max_query_words = max(1, max_query_words)
    perspective = choose_perspective(parent_state, action)
    parent_id = parent_state.node_id if parent_state is not None else None
    depth = 1 if parent_state is None else parent_state.depth + 1
    facet_counts = exploration_state.setdefault("facet_counts", {})
    used_queries = exploration_state.setdefault("used_queries", set())
    used_open_questions = exploration_state.setdefault("used_open_questions", set())
    seen_query_sigs = exploration_state.setdefault("seen_query_sigs", set())
    seen_url_sigs = exploration_state.setdefault("seen_url_sigs", [])
    facet = _select_facet(
        parent_state=parent_state,
        action=action,
        facets=facets,
        facet_counts=facet_counts,
    )
    focus_question = (
        _select_focus_question(parent_state, used_open_questions)
        if action == "deepen"
        else None
    )
    avoid_queries = sorted(str(query) for query in used_queries)[-20:]

    planner_text = call_local_llm(
        system_prompt=query_planner_system_prompt(),
        user_prompt=build_query_planner_prompt(
            topic=topic,
            action=action,
            perspective=perspective,
            facet=facet,
            parent_state=parent_state,
            avoid_queries=avoid_queries,
            focus_question=focus_question,
            max_queries=max_queries,
            max_words=max_query_words,
        ),
        temperature=planner_temperature,
        max_tokens=max_tokens,
        research_logger=research_logger,
        node_id=parent_id,
        action=action,
        perspective=perspective,
        role="planner",
    )
    search_queries = parse_query_plan(
        planner_text,
        max_queries=max_queries,
        max_words=max_query_words,
    )
    if not search_queries:
        search_queries = build_fallback_queries(
            topic,
            action,
            parent_state,
            perspective,
            max_queries=max_queries,
            max_words=max_query_words,
        )

    query_sig = query_set_signature(search_queries)
    per_query_results: list[list[dict[str, Any]]] = []
    search_successes: list[bool] = []
    for query_index, search_query in enumerate(search_queries, start=1):
        query_sources, query_success = web_search(
            search_query,
            max_results=max_results,
        )
        annotated_sources = [
            {
                **source,
                "query": search_query,
                "query_index": query_index,
            }
            for source in query_sources
        ]
        per_query_results.append(annotated_sources)
        search_successes.append(query_success)

        if research_logger is not None:
            research_logger.log_event(
                "search",
                parent_id=parent_id,
                action=action,
                perspective=perspective,
                query=search_query,
                query_index=query_index,
                num_sources=len(query_sources),
                search_success=query_success,
                results=annotated_sources,
            )

    sources = merge_and_dedupe_sources(per_query_results)
    url_sig = source_url_signature(sources)
    repetition_penalty = (
        novelty_penalty_weight
        if query_sig in seen_query_sigs or _is_repeated_sources(url_sig, seen_url_sigs)
        else 0.0
    )
    applied_coverage_bonus = (
        coverage_bonus if facet is not None and facet_counts.get(facet, 0) == 0 else 0.0
    )
    search_success = any(search_successes)
    search_query = " | ".join(search_queries)

    if research_logger is not None:
        research_logger.log_event(
            "search_merged",
            parent_id=parent_id,
            action=action,
            perspective=perspective,
            queries=search_queries,
            facet=facet,
            focus_question=focus_question,
            num_sources=len(sources),
            search_success=search_success,
            results=sources,
        )

    system_prompt = researcher_system_prompt(action, perspective)
    user_prompt = build_research_prompt(
        topic=topic,
        action=action,
        perspective=perspective,
        facet=facet,
        search_queries=search_queries,
        sources=sources,
        parent_state=parent_state,
    )
    text = call_local_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        research_logger=research_logger,
        node_id=parent_id,
        action=action,
        perspective=perspective,
        role="researcher",
    )
    open_questions = extract_open_questions(text)

    review_system = reviewer_system_prompt()
    review_user = build_review_prompt(
        topic=topic,
        action=action,
        perspective=perspective,
        text=text,
        sources=sources,
    )
    review_text = call_local_llm(
        system_prompt=review_system,
        user_prompt=review_user,
        temperature=0.0,
        max_tokens=max_tokens,
        research_logger=research_logger,
        node_id=parent_id,
        action=action,
        perspective=perspective,
        role="reviewer",
    )

    base_score, findings, review_summary = parse_review_payload(review_text)
    score = score_review(
        base_score,
        findings,
        num_sources=len(sources),
        search_success=search_success,
        repetition_penalty=repetition_penalty,
        coverage_bonus=applied_coverage_bonus,
    )
    eval_results = make_eval_results(score, findings)
    state = NodeState(
        topic=topic,
        action=action,
        perspective=perspective,
        text=text,
        sources=sources,
        findings=findings,
        eval_results=eval_results,
        score=score,
        parent_id=parent_id,
        depth=depth,
        facet=facet,
        search_query=search_query,
        search_queries=search_queries,
        open_questions=open_questions,
        review_text=review_summary,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)
    if research_logger is not None:
        score_delta = None if parent_state is None else score - parent_state.score
        research_logger.log_node(state)
        research_logger.log_edge(
            parent_id=parent_id,
            child_id=state.node_id,
            action=action,
            score_delta=score_delta,
        )
        research_logger.log_event(
            "node_generated",
            node_id=state.node_id,
            parent_id=parent_id,
            action=action,
            perspective=perspective,
            facet=facet,
            score=score,
            query=search_query,
            queries=search_queries,
            num_sources=len(sources),
            findings=findings,
            open_questions=open_questions,
            focus_question=focus_question,
            repetition_penalty=repetition_penalty,
            coverage_bonus=applied_coverage_bonus,
            elapsed_ms=elapsed_ms,
        )
        research_logger.log_human(
            "node_generated",
            node_id=state.node_id,
            action=action,
            perspective=perspective,
            facet=facet,
            score=f"{score:.3f}",
            sources=len(sources),
            findings=len(findings),
        )
        research_logger.log_progress(
            f"Node {state.node_id[:8]}",
            (
                f"- action: `{action}`\n"
                f"- perspective: `{perspective}`\n"
                f"- facet: `{facet or ''}`\n"
                f"- score: `{score:.3f}`\n"
                f"- repetition_penalty: `{repetition_penalty:.3f}`\n"
                f"- coverage_bonus: `{applied_coverage_bonus:.3f}`\n"
                f"- focus_question: `{focus_question or ''}`\n"
                f"- search_queries: `{json.dumps(search_queries, ensure_ascii=False)}`\n"
                f"- sources: `{len(sources)}`\n"
                f"- findings: `{len(findings)}`\n\n"
                f"### 次に渡す問い\n\n"
                f"{json.dumps(open_questions, ensure_ascii=False, indent=2)}\n\n"
                f"{text}"
            ),
        )

    if facet is not None:
        facet_counts[facet] = facet_counts.get(facet, 0) + 1
    used_queries.update(search_queries)
    if focus_question is not None:
        used_open_questions.add(focus_question)
    seen_query_sigs.add(query_sig)
    if url_sig:
        seen_url_sigs.append(url_sig)

    return state, score


def build_actions(include_revise: bool) -> list[str]:
    actions = list(DEFAULT_ACTIONS)
    if not include_revise:
        actions.remove("revise")
    return actions


def build_algorithm(algo_name: str, params: dict[str, Any], batch_size: int) -> tq.Algorithm:
    algo_cls = getattr(tq, algo_name)
    try:
        return algo_cls(**params)
    except TypeError:
        if algo_name == "ABMCTSM":
            return algo_cls(max_process_workers=batch_size)
        raise


def count_states(search_tree: Any, algo: Any) -> int:
    if hasattr(algo, "get_state_score_pairs"):
        return len(algo.get_state_score_pairs(search_tree))
    try:
        return len(tq.top_k(search_tree, algo, k=10_000))
    except Exception:
        return 0


def save_checkpoint(search_tree: Any, save_dir: Path, n_states: int) -> None:
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    with (checkpoint_dir / f"checkpoint_n_states_{n_states}.pkl").open("wb") as file:
        pickle.dump(search_tree, file)
    with (checkpoint_dir / "checkpoint_latest.pkl").open("wb") as file:
        pickle.dump(search_tree, file)


def save_time_summary(
    save_dir: Path,
    *,
    total_time: float,
    node_times: list[float],
) -> None:
    costs_dir = save_dir / "costs"
    costs_dir.mkdir(exist_ok=True)
    time_summary = {
        "total_cost": 0.0,
        "cost_by_model": {},
        "total_time": total_time,
        "total_time_minutes": total_time / 60,
        "total_time_hours": total_time / 3600,
        "node_times": node_times,
        "avg_node_time": sum(node_times) / len(node_times) if node_times else 0,
    }
    (costs_dir / "time_summary.json").write_text(
        json.dumps(time_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (save_dir / "time_summary.json").write_text(
        json.dumps(time_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_final_report(
    *,
    topic: str,
    search_tree: Any,
    algo: Any,
    top_k: int,
    temperature: float,
    max_tokens: int,
    research_logger: ResearchLogger,
) -> None:
    top_states = get_top_k(search_tree, algo, k=top_k)
    if not top_states:
        research_logger.write_final_artifact(
            "final_report.md",
            "# Final Report\n\n探索ノードが生成されませんでした。\n",
        )
        return

    report = call_local_llm(
        system_prompt=final_report_system_prompt(),
        user_prompt=build_final_report_prompt(topic, top_states),
        temperature=temperature,
        max_tokens=max_tokens,
        research_logger=research_logger,
        node_id=None,
        action="final_report",
        perspective="編集長",
        role="editor",
    )
    review = call_local_llm(
        system_prompt=reviewer_system_prompt(),
        user_prompt=build_final_review_prompt(topic=topic, report=report),
        temperature=0.0,
        max_tokens=max_tokens,
        research_logger=research_logger,
        node_id=None,
        action="final_review",
        perspective="レビュアー",
        role="reviewer",
    )
    base_score, findings, summary = parse_review_payload(review)
    final_score = clamp01(base_score - min(0.30, 0.04 * len(findings)))

    research_logger.write_final_artifact("final_report.md", report)
    research_logger.write_final_artifact(
        "final_review.json",
        json.dumps(
            {
                "score": final_score,
                "summary": summary,
                "findings": findings,
                "raw": review,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    research_logger.log_event(
        "final_report",
        score=final_score,
        findings=findings,
        num_top_states=len(top_states),
    )
    research_logger.log_progress(
        "Final Report",
        f"- final_score: `{final_score:.3f}`\n- findings: `{len(findings)}`\n",
    )


def decompose_topic(
    *,
    topic: str,
    num_facets: int,
    temperature: float,
    max_tokens: int,
    research_logger: ResearchLogger,
) -> list[str]:
    planner_text = call_local_llm(
        system_prompt=topic_decomposition_system_prompt(),
        user_prompt=build_topic_decomposition_prompt(topic, num_facets=num_facets),
        temperature=temperature,
        max_tokens=max_tokens,
        research_logger=research_logger,
        node_id=None,
        action="topic_decomposition",
        perspective="設計",
        role="decomposer",
    )
    facets = parse_facets(planner_text, num_facets=num_facets)
    if not facets:
        facets = _fallback_facets(topic, num_facets=num_facets)
    research_logger.log_event("facets", facets=facets)
    return facets


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_env_defaults(cfg)
    start_time = time.time()
    node_times: list[float] = []

    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    for subdir in ["checkpoints", "costs", "llm_logs"]:
        (save_dir / subdir).mkdir(parents=True, exist_ok=True)
    research_logger = ResearchLogger(save_dir)
    research_logger.log_event(
        "run_started",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    topic = str(cfg["research_topic"])
    llm_cfg = cfg.get("llm", {})
    search_cfg = cfg.get("search", {})
    exploration_cfg = cfg.get("exploration", {})
    algo_cfg = cfg["algo"]
    algo_name = str(algo_cfg["class_name"])
    algo_params = dict(algo_cfg.get("params", {}))
    batch_size = int(algo_cfg.get("batch_size", 5))
    max_num_nodes = int(cfg["max_num_nodes"])
    top_k = int(cfg.get("top_k", 5))
    actions = build_actions(bool(cfg.get("include_revise", True)))
    planner_temperature = float(_cfg_get(exploration_cfg, "planner_temperature", 0.7))
    num_facets = int(_cfg_get(exploration_cfg, "num_facets", 4))
    novelty_penalty_weight = float(
        _cfg_get(exploration_cfg, "novelty_penalty_weight", 0.15)
    )
    coverage_bonus = float(_cfg_get(exploration_cfg, "coverage_bonus", 0.05))
    facets = decompose_topic(
        topic=topic,
        num_facets=num_facets,
        temperature=planner_temperature,
        max_tokens=int(_cfg_get(llm_cfg, "max_tokens", 1200)),
        research_logger=research_logger,
    )
    exploration_state: dict[str, Any] = {
        "facet_counts": {},
        "used_queries": set(),
        "used_open_questions": set(),
        "seen_query_sigs": set(),
        "seen_url_sigs": [],
    }

    algo = build_algorithm(algo_name, algo_params, batch_size)
    generate_fns = {
        action: partial(
            generate_fn,
            action=action,
            topic=topic,
            temperature=float(_cfg_get(llm_cfg, "temperature", 0.3)),
            planner_temperature=planner_temperature,
            max_tokens=int(_cfg_get(llm_cfg, "max_tokens", 30000)),
            max_results=int(_cfg_get(search_cfg, "max_results", 5)),
            max_queries=int(_cfg_get(search_cfg, "max_queries", 3)),
            max_query_words=int(_cfg_get(search_cfg, "max_query_words", 10)),
            facets=facets,
            exploration_state=exploration_state,
            novelty_penalty_weight=novelty_penalty_weight,
            coverage_bonus=coverage_bonus,
            research_logger=research_logger,
        )
        for action in actions
    }

    checkpoint_path = cfg.get("checkpoint_path")
    if checkpoint_path:
        with Path(checkpoint_path).open("rb") as file:
            search_tree = pickle.load(file)
        logger.info("Loaded checkpoint from %s", checkpoint_path)
    else:
        search_tree = algo.init_tree()
        logger.info("Initialized search tree")

    if algo_name == "ABMCTSM":
        num_steps = max(1, (max_num_nodes + batch_size - 1) // batch_size)
        for step in tqdm(range(num_steps)):
            node_start = time.time()
            search_tree, trials = algo.ask_batch(search_tree, batch_size, actions)
            for trial in trials:
                result = generate_fns[trial.action](trial.parent_state)
                search_tree = algo.tell(search_tree, trial.trial_id, result)
            node_times.append(time.time() - node_start)
            n_states = count_states(search_tree, algo)
            research_logger.log_event("batch_completed", step=step + 1, n_states=n_states)
            save_checkpoint(search_tree, save_dir, n_states)
    else:
        initial_states = count_states(search_tree, algo)
        for step in tqdm(range(max(0, max_num_nodes - initial_states))):
            node_start = time.time()
            search_tree = algo.step(search_tree, generate_fns)
            node_times.append(time.time() - node_start)
            n_states = count_states(search_tree, algo)
            research_logger.log_event("step_completed", step=step + 1, n_states=n_states)
            if n_states % 10 == 0 or n_states in (1, 2, 4, 8):
                save_checkpoint(search_tree, save_dir, n_states)

    try:
        tq.render(
            search_tree,
            output_basename=save_dir / "tree" / "tree",
            format="html",
            state_formatter=state_formatter_html,
        )
    except Exception as exc:
        research_logger.log_event("tree_render_failed", error=str(exc))

    write_final_report(
        topic=topic,
        search_tree=search_tree,
        algo=algo,
        top_k=top_k,
        temperature=float(_cfg_get(llm_cfg, "temperature", 0.3)),
        max_tokens=int(_cfg_get(llm_cfg, "max_tokens", 1200)),
        research_logger=research_logger,
    )
    total_time = time.time() - start_time
    save_time_summary(save_dir, total_time=total_time, node_times=node_times)
    research_logger.log_event("run_completed", total_time=total_time)


if __name__ == "__main__":
    load_dotenv(Path(__file__).parents[1].resolve())
    main()
