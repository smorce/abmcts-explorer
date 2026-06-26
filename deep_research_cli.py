from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from abmcts_explorer import (
    ActionSpec,
    DeepResearchRunner,
    DeepResearchRunnerConfig,
    DeepResearchState,
    build_deep_research_prompt,
    parse_deep_research_state,
    score_deep_research_state,
)
from abmcts_explorer.tree_visualizer import TreeVisualizerObserver, TreeVisualizerServer


def build_actions(model: str, *, max_tokens: int | None = None) -> list[ActionSpec[DeepResearchState]]:
    wide_tokens = max_tokens or 2200
    reject_tokens = max_tokens or 2600
    synthesis_tokens = max_tokens or 3600
    return [
        ActionSpec[DeepResearchState](
            name="wide_hypothesis_search",
            model=model,
            prompt_builder=build_deep_research_prompt,
            parser=parse_deep_research_state,
            scorer=score_deep_research_state,
            temperature=0.75,
            max_tokens=wide_tokens,
        ),
        ActionSpec[DeepResearchState](
            name="counterevidence_and_rejection",
            model=model,
            prompt_builder=build_deep_research_prompt,
            parser=parse_deep_research_state,
            scorer=score_deep_research_state,
            temperature=0.45,
            max_tokens=reject_tokens,
        ),
        ActionSpec[DeepResearchState](
            name="deep_synthesis",
            model=model,
            prompt_builder=build_deep_research_prompt,
            parser=parse_deep_research_state,
            scorer=score_deep_research_state,
            temperature=0.2,
            max_tokens=synthesis_tokens,
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-research-runner",
        description="Run high-budget DeepResearch with AB-MCTS profile switching.",
    )
    parser.add_argument("--topic", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--total-budget", type=int, default=1000)
    parser.add_argument("--epoch-budget", type=int, default=25)
    parser.add_argument("--best-k", type=int, default=8)
    parser.add_argument("--search-limit", type=int, default=8)
    parser.add_argument("--min-wide-epochs", type=int, default=3)
    parser.add_argument("--wide-batch-size", type=int, default=16)
    parser.add_argument("--deep-batch-size", type=int, default=5)
    parser.add_argument("--action-max-tokens", type=int)
    parser.add_argument("--report-max-tokens", type=int, default=8192)
    parser.add_argument("--output-dir", default="deep_research_run")
    parser.add_argument("--no-final-llm-report", action="store_true")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8765)
    parser.add_argument("--ui-hold-seconds", type=int, default=0)
    parser.add_argument("--no-open-browser", action="store_true")
    return parser


async def run(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observer = TreeVisualizerObserver(title=f"DeepResearch: {args.topic}") if args.ui else None
    server = None
    if observer is not None:
        server = TreeVisualizerServer(observer, host=args.ui_host, port=args.ui_port)
        print(f"ui={server.start(open_browser=not args.no_open_browser)}")

    try:
        runner = DeepResearchRunner(
            actions=build_actions(args.model, max_tokens=args.action_max_tokens),
            config=DeepResearchRunnerConfig(
                total_budget=args.total_budget,
                epoch_budget=args.epoch_budget,
                best_k=args.best_k,
                search_limit=args.search_limit,
                min_wide_epochs=args.min_wide_epochs,
                wide_batch_size=args.wide_batch_size,
                deep_batch_size=args.deep_batch_size,
                report_model=None if args.no_final_llm_report else args.model,
                report_max_tokens=args.report_max_tokens,
            ),
            observer=observer,
        )
        result = await runner.run(args.topic)

        report_path = output_dir / "report.md"
        profile_path = output_dir / "profile_history.md"
        sources_path = output_dir / "sources.md"
        node_log_path = output_dir / "node_logs.jsonl"

        report_path.write_text(result.report, encoding="utf-8")
        profile_path.write_text(
            "\n".join(
                f"- {index + 1}: {decision.profile.value} "
                f"confidence={decision.confidence:.2f} {decision.explanation}"
                for index, decision in enumerate(result.profile_history)
            ),
            encoding="utf-8",
        )
        sources_path.write_text(
            "\n".join(f"- [{item.title}]({item.url})" for item in result.web_results),
            encoding="utf-8",
        )
        node_log_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in result.node_logs)
            + ("\n" if result.node_logs else ""),
            encoding="utf-8",
        )
        log_paths = result.logger.save(output_dir / "logs") if result.logger else {}

        if observer is not None and args.ui_hold_seconds != 0:
            await _hold_ui(args.ui_hold_seconds)

        return {
            "report": report_path,
            "profile_history": profile_path,
            "sources": sources_path,
            "node_logs": node_log_path,
            **log_paths,
        }
    finally:
        if server is not None:
            server.stop()


async def _hold_ui(seconds: int) -> None:
    if seconds < 0:
        while True:
            await asyncio.sleep(3600)
    await asyncio.sleep(seconds)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = asyncio.run(run(args))
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
