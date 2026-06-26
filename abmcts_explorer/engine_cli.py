from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .core import (
    ABMCTSExplorer,
    ActionSpec,
    AlgorithmKind,
    ExecutionMode,
    ExplorerConfig,
    ExplorerContext,
    GenerationResult,
    SearchProfile,
)
from .profile_decider import (
    ProfileDeciderConfig,
    ProfileDecision,
    RuleBasedProfileDecider,
    SearchDiagnostics,
)
from .tree_visualizer import TreeVisualizerObserver, TreeVisualizerServer


@dataclass(frozen=True)
class EngineCliState:
    text: str
    depth: int = 0
    metadata: dict[str, Any] | None = None


def _score_text(text: str) -> float:
    if not text.strip():
        return 0.0
    length_score = min(len(text) / 2000, 1.0)
    structure_score = 1.0 if ("\n" in text or "#" in text or "-" in text) else 0.5
    return max(0.0, min(length_score * 0.7 + structure_score * 0.3, 1.0))


def make_demo_action() -> ActionSpec[EngineCliState]:
    calls = {"count": 0}

    def generate(
        parent_state: EngineCliState | None,
        context: ExplorerContext,
    ) -> GenerationResult[EngineCliState]:
        calls["count"] += 1
        depth = 0 if parent_state is None else parent_state.depth + 1
        parent_text = parent_state.text if parent_state else context.task
        text = (
            f"# Candidate {calls['count']}\n"
            f"profile: {context.profile.value}\n"
            f"depth: {depth}\n"
            f"task: {context.task}\n"
            f"parent: {parent_text[:160]}\n"
            f"- angle: {calls['count'] % 3}\n"
            f"- refinement: {depth}\n"
        )
        return GenerationResult(
            state=EngineCliState(
                text=text,
                depth=depth,
                metadata={"call": calls["count"], "profile": context.profile.value},
            ),
            score=_score_text(text),
        )

    return ActionSpec(name="demo_expand", generator=generate)


def make_jsonl_action(path: Path) -> ActionSpec[EngineCliState]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No JSONL rows found: {path}")
    cursor = {"index": 0}

    def generate(
        parent_state: EngineCliState | None,
        context: ExplorerContext,
    ) -> GenerationResult[EngineCliState]:
        row = rows[cursor["index"] % len(rows)]
        cursor["index"] += 1
        text = str(row.get("text", row.get("state", "")))
        score = float(row.get("score", _score_text(text)))
        depth = 0 if parent_state is None else parent_state.depth + 1
        return GenerationResult(
            state=EngineCliState(
                text=text,
                depth=depth,
                metadata={
                    "row_index": cursor["index"] - 1,
                    "profile": context.profile.value,
                },
            ),
            score=max(0.0, min(score, 1.0)),
        )

    return ActionSpec(name="jsonl_expand", generator=generate)


def _write_node_log(path: Path, node_logs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, ensure_ascii=False) for item in node_logs]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_config(
    args: argparse.Namespace,
    profile: SearchProfile | None = None,
    budget: int | None = None,
) -> ExplorerConfig:
    profile = profile or SearchProfile(args.profile)
    budget = budget if budget is not None else args.budget
    if profile == SearchProfile.GO_DEEP:
        return ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSM,
            profile=profile,
            execution_mode=ExecutionMode.ASK_TELL,
            budget=budget,
            batch_size=args.batch_size,
            best_k=args.best_k,
            algo_kwargs={"max_process_workers": args.batch_size},
            metadata={"cli": True},
        )

    return ExplorerConfig(
        algorithm_kind=AlgorithmKind.ABMCTSA,
        profile=profile,
        execution_mode=ExecutionMode.ASYNC_ASK_TELL,
        budget=budget,
        batch_size=args.batch_size,
        best_k=args.best_k,
        metadata={"cli": True},
    )


async def run_engine(args: argparse.Namespace) -> dict[str, Any]:
    action = make_jsonl_action(Path(args.input_jsonl)) if args.input_jsonl else make_demo_action()
    observer = TreeVisualizerObserver(title=f"AB-MCTS: {args.task}") if args.ui else None
    server = None
    if observer is not None:
        server = TreeVisualizerServer(observer, host=args.ui_host, port=args.ui_port)
        args.ui_url = server.start(open_browser=not args.no_open_browser)
        print(f"ui={args.ui_url}")

    try:
        if args.profile == "auto":
            result = await run_engine_auto(args, action, observer=observer)
        else:
            result = await run_engine_single(args, action, observer=observer)

        if args.node_log:
            _write_node_log(Path(args.node_log), result["node_logs"])

        if observer is not None and args.ui_hold_seconds != 0:
            await _hold_ui(args.ui_hold_seconds)

        return result
    finally:
        if server is not None:
            server.stop()


async def run_engine_single(
    args: argparse.Namespace,
    action: ActionSpec[EngineCliState],
    *,
    observer: TreeVisualizerObserver | None = None,
) -> dict[str, Any]:
    if args.profile == "auto":
        raise ValueError("run_engine_single does not accept profile=auto.")

    explorer = ABMCTSExplorer[EngineCliState](
        task=args.task,
        actions=[action],
        config=build_config(args),
        observer=observer,
    )

    if explorer.config.execution_mode == ExecutionMode.ASYNC_ASK_TELL:
        best = await explorer.run_async()
    else:
        best = await asyncio.to_thread(explorer.run)

    return {
        "task": args.task,
        "profile": explorer.config.profile.value,
        "algorithm": explorer.config.algorithm_kind.value,
        "budget": args.budget,
        "batch_size": args.batch_size,
        "best": [
            {
                "rank": index + 1,
                "score": score,
                "state": asdict(state),
            }
            for index, (state, score) in enumerate(best)
        ],
        "history": explorer.history,
        "node_logs": explorer.node_logs,
    }


async def run_engine_auto(
    args: argparse.Namespace,
    action: ActionSpec[EngineCliState],
    *,
    observer: TreeVisualizerObserver | None = None,
) -> dict[str, Any]:
    decider = RuleBasedProfileDecider(
        ProfileDeciderConfig(
            min_initial_nodes=args.epoch_budget,
            min_epochs_before_switch=1,
            strong_candidate_score=0.55,
            top_k_convergence_std=0.08,
            late_budget_ratio=0.35,
            switch_confidence_threshold=0.55,
        )
    )
    profile = SearchProfile.GO_WIDE
    explorer = ABMCTSExplorer[EngineCliState](
        task=args.task,
        actions=[action],
        config=build_config(args, profile=profile, budget=args.epoch_budget),
        observer=observer,
    )

    consumed = 0
    best: list[tuple[EngineCliState, float]] = []
    decisions: list[ProfileDecision] = []
    diagnostics_history: list[SearchDiagnostics] = []
    history: list[dict[str, Any]] = []
    node_logs: list[dict[str, Any]] = []
    last_history_index = 0
    last_node_log_index = 0

    while consumed < args.budget:
        epoch_budget = min(args.epoch_budget, args.budget - consumed)

        if explorer.config.execution_mode == ExecutionMode.ASYNC_ASK_TELL:
            best = await explorer.run_async(budget=epoch_budget)
        else:
            best = await asyncio.to_thread(explorer.run, epoch_budget)

        consumed += epoch_budget
        history.extend(explorer.history[last_history_index:])
        node_logs.extend(explorer.node_logs[last_node_log_index:])
        last_history_index = len(explorer.history)
        last_node_log_index = len(explorer.node_logs)

        diagnostics = diagnose_best(
            best=best,
            consumed_budget=consumed,
            total_budget=args.budget,
        )
        diagnostics_history.append(diagnostics)
        decision = decider.decide(diagnostics, explorer.config.profile)

        if len(decisions) < args.min_wide_epochs:
            decision = ProfileDecision(
                profile=SearchProfile.GO_WIDE,
                confidence=max(decision.confidence, 0.75),
                reasons=decision.reasons,
                explanation=decision.explanation + " | forced_initial_go_wide",
                suggested_batch_size=decision.suggested_batch_size,
                suggested_budget=decision.suggested_budget,
            )

        decisions.append(decision)

        if consumed < args.budget and decision.profile != explorer.config.profile:
            explorer = ABMCTSExplorer[EngineCliState](
                task=args.task,
                actions=[action],
                config=build_config(args, profile=decision.profile, budget=args.epoch_budget),
                observer=observer,
            )
            last_history_index = 0
            last_node_log_index = 0

    return {
        "task": args.task,
        "profile": "auto",
        "last_executed_profile": explorer.config.profile.value,
        "next_profile_recommendation": (
            decisions[-1].profile.value if decisions else profile.value
        ),
        "budget": args.budget,
        "batch_size": args.batch_size,
        "epoch_budget": args.epoch_budget,
        "best": [
            {
                "rank": index + 1,
                "score": score,
                "state": asdict(state),
            }
            for index, (state, score) in enumerate(best)
        ],
        "profile_decisions": [
            {
                "profile": decision.profile.value,
                "confidence": decision.confidence,
                "reasons": [reason.value for reason in decision.reasons],
                "explanation": decision.explanation,
            }
            for decision in decisions
        ],
        "diagnostics": [
            {
                "total_nodes": item.total_nodes,
                "best_score": item.best_score,
                "top_k_mean_score": item.top_k_mean_score,
                "top_k_score_std": item.top_k_score_std,
                "diversity_score": item.diversity_score,
                "uncertainty_score": item.uncertainty_score,
                "remaining_budget_ratio": item.remaining_budget_ratio,
            }
            for item in diagnostics_history
        ],
        "history": history,
        "node_logs": node_logs,
    }


async def _hold_ui(seconds: int) -> None:
    if seconds < 0:
        while True:
            await asyncio.sleep(3600)
    await asyncio.sleep(seconds)


def diagnose_best(
    *,
    best: list[tuple[EngineCliState, float]],
    consumed_budget: int,
    total_budget: int,
) -> SearchDiagnostics:
    scores = [score for _, score in best]
    best_score = max(scores) if scores else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0
    if len(scores) >= 2:
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        score_std = variance**0.5
    else:
        score_std = 0.0

    depths = [state.depth for state, _ in best]
    unique_prefixes = {state.text[:80] for state, _ in best}
    diversity_score = len(unique_prefixes) / max(len(best), 1)
    uncertainty_score = max(0.0, 1.0 - best_score)

    return SearchDiagnostics(
        total_nodes=consumed_budget,
        max_depth=max(depths) if depths else 0,
        avg_depth=sum(depths) / len(depths) if depths else 0.0,
        best_score=best_score,
        top_k_mean_score=mean_score,
        top_k_score_std=score_std,
        score_improvement_recent=best_score - mean_score,
        diversity_score=diversity_score,
        uncertainty_score=uncertainty_score,
        judge_disagreement=0.0,
        plateau_steps=1 if score_std < 0.03 and best else 0,
        remaining_budget_ratio=max(0.0, (total_budget - consumed_budget) / max(total_budget, 1)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abmcts-engine",
        description="Run ABMCTSExplorer as a standalone search engine.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--profile",
        choices=[SearchProfile.GO_WIDE.value, SearchProfile.GO_DEEP.value, "auto"],
        default=SearchProfile.GO_WIDE.value,
    )
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--epoch-budget", type=int, default=4)
    parser.add_argument("--min-wide-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--best-k", type=int, default=2)
    parser.add_argument("--input-jsonl")
    parser.add_argument("--output")
    parser.add_argument("--node-log")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8765)
    parser.add_argument("--ui-hold-seconds", type=int, default=0)
    parser.add_argument("--no-open-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = asyncio.run(run_engine(args))
    output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
