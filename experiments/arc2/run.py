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
    build_research_prompt,
    build_review_prompt,
    final_report_system_prompt,
    researcher_system_prompt,
    reviewer_system_prompt,
)
from utils import (
    DEFAULT_ACTIONS,
    NodeState,
    build_search_query,
    choose_perspective,
    clamp01,
    get_top_k,
    make_eval_results,
    parse_review_payload,
    score_review,
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
    max_tokens: int,
    max_results: int,
    research_logger: ResearchLogger | None = None,
) -> tuple[NodeState, float]:
    start_time = time.time()
    perspective = choose_perspective(parent_state, action)
    search_query = build_search_query(topic, action, parent_state, perspective)

    sources, search_success = web_search(search_query, max_results=max_results)
    parent_id = parent_state.node_id if parent_state is not None else None
    depth = 1 if parent_state is None else parent_state.depth + 1

    if research_logger is not None:
        research_logger.log_event(
            "search",
            parent_id=parent_id,
            action=action,
            perspective=perspective,
            query=search_query,
            num_sources=len(sources),
            search_success=search_success,
        )

    system_prompt = researcher_system_prompt(action, perspective)
    user_prompt = build_research_prompt(
        topic=topic,
        action=action,
        perspective=perspective,
        search_query=search_query,
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
        search_query=search_query,
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
            score=score,
            query=search_query,
            num_sources=len(sources),
            findings=findings,
            elapsed_ms=elapsed_ms,
        )
        research_logger.log_human(
            "node_generated",
            node_id=state.node_id,
            action=action,
            perspective=perspective,
            score=f"{score:.3f}",
            sources=len(sources),
            findings=len(findings),
        )
        research_logger.log_progress(
            f"Node {state.node_id[:8]}",
            (
                f"- action: `{action}`\n"
                f"- perspective: `{perspective}`\n"
                f"- score: `{score:.3f}`\n"
                f"- sources: `{len(sources)}`\n"
                f"- findings: `{len(findings)}`\n\n"
                f"{text[:1000]}"
            ),
        )

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
        user_prompt=f"# 調査テーマ\n{topic}\n\n# 最終レポート\n{report}",
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
    algo_cfg = cfg["algo"]
    algo_name = str(algo_cfg["class_name"])
    algo_params = dict(algo_cfg.get("params", {}))
    batch_size = int(algo_cfg.get("batch_size", 5))
    max_num_nodes = int(cfg["max_num_nodes"])
    top_k = int(cfg.get("top_k", 5))
    actions = build_actions(bool(cfg.get("include_revise", True)))

    algo = build_algorithm(algo_name, algo_params, batch_size)
    generate_fns = {
        action: partial(
            generate_fn,
            action=action,
            topic=topic,
            temperature=float(_cfg_get(llm_cfg, "temperature", 0.3)),
            max_tokens=int(_cfg_get(llm_cfg, "max_tokens", 1200)),
            max_results=int(_cfg_get(search_cfg, "max_results", 5)),
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
