from __future__ import annotations

import asyncio

from abmcts_explorer import (
    ABMCTSExplorer,
    ActionSpec,
    AlgorithmKind,
    ExecutionMode,
    ExplorerConfig,
    ExplorerContext,
    GenerationResult,
    SearchProfile,
)


def make_counter_action(name: str = "expand") -> ActionSpec[str]:
    calls = {"count": 0}

    def generate(
        parent_state: str | None,
        context: ExplorerContext,
    ) -> GenerationResult[str]:
        calls["count"] += 1
        depth = 0 if parent_state is None else parent_state.count("/")
        state = f"{parent_state or 'root'}/{name}-{calls['count']}"
        score = min(0.1 + (depth + calls["count"]) * 0.05, 1.0)
        return GenerationResult(state=state, score=score)

    return ActionSpec(name=name, generator=generate)


def test_abmctsa_step_runs_with_real_treequest_backend() -> None:
    explorer = ABMCTSExplorer[str](
        task="deterministic search",
        actions=[make_counter_action()],
        config=ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSA,
            profile=SearchProfile.GO_WIDE,
            execution_mode=ExecutionMode.STEP,
            budget=6,
            best_k=3,
        ),
    )

    best = explorer.run()

    assert explorer.step_index == 6
    assert len(best) == 3
    assert all(isinstance(state, str) for state, _ in best)
    assert all(0.0 <= score <= 1.0 for _, score in best)
    assert explorer.history[-1]["event_type"] == "step"


def test_abmctsm_ask_tell_runs_with_real_treequest_backend() -> None:
    explorer = ABMCTSExplorer[str](
        task="deterministic batched search",
        actions=[make_counter_action()],
        config=ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSM,
            profile=SearchProfile.GO_DEEP,
            execution_mode=ExecutionMode.ASK_TELL,
            budget=6,
            batch_size=3,
            best_k=2,
            algo_kwargs={"max_process_workers": 3},
        ),
    )

    best = explorer.run()

    assert explorer.step_index == 6
    assert len(best) == 2
    assert all(isinstance(state, str) for state, _ in best)
    assert all(0.0 <= score <= 1.0 for _, score in best)
    assert explorer.history[-1]["event_type"] == "ask_tell_batch"


def test_async_ask_tell_runs_with_real_treequest_backend() -> None:
    explorer = ABMCTSExplorer[str](
        task="deterministic async search",
        actions=[make_counter_action()],
        config=ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSA,
            profile=SearchProfile.GO_WIDE,
            execution_mode=ExecutionMode.ASYNC_ASK_TELL,
            budget=4,
            batch_size=2,
            best_k=2,
        ),
    )

    best = asyncio.run(explorer.run_async())

    assert explorer.step_index == 4
    assert len(best) == 2
    assert all(isinstance(state, str) for state, _ in best)
    assert all(0.0 <= score <= 1.0 for _, score in best)
    assert explorer.history[-1]["event_type"] == "ask_tell_batch_async"
