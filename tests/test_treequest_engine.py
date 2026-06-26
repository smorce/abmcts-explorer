from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

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


class BackendWithParent:
    name = "backend_with_parent"

    def init_tree(self) -> list[tuple[str, float]]:
        return []

    def step(self, tree: Any, generate_fns: dict[str, Any]) -> Any:
        state, score = next(iter(generate_fns.values()))("backend-parent")
        return [*tree, (state, score)]

    def ask_batch(
        self,
        tree: Any,
        batch_size: int,
        actions: list[str],
    ) -> tuple[Any, list[Any]]:
        return (
            tree,
            [
                SimpleNamespace(
                    action=actions[0],
                    parent_state="backend-parent",
                    trial_id=index,
                )
                for index in range(batch_size)
            ],
        )

    def tell(
        self,
        tree: Any,
        trial_id: Any,
        result: tuple[str, float],
    ) -> Any:
        return [*tree, result]

    def top_k(self, tree: Any, k: int) -> list[tuple[str, float]]:
        return list(tree)[-k:]

    def is_tree_compatible_with(self, other: Any) -> bool:
        return False


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


def test_best_reduces_k_when_tree_has_fewer_states() -> None:
    explorer = ABMCTSExplorer[str](
        task="small tree",
        actions=[make_counter_action()],
        config=ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSA,
            profile=SearchProfile.GO_WIDE,
            execution_mode=ExecutionMode.ASK_TELL,
            budget=1,
            batch_size=1,
            best_k=3,
        ),
    )

    explorer.ask_tell_batch(1)
    best = explorer.best(3)

    assert len(best) == 1


def test_ask_tell_uses_multiple_action_parent_states_round_robin() -> None:
    parent_states = ["seed-a", "seed-b"]
    seen_parents: list[str | None] = []

    def generate(
        parent_state: str | None,
        context: ExplorerContext,
    ) -> GenerationResult[str]:
        seen_parents.append(parent_state)
        state = f"{parent_state}/child-{len(seen_parents)}"
        return GenerationResult(state=state, score=0.5)

    explorer = ABMCTSExplorer[str](
        task="seeded search",
        actions=[ActionSpec(name="expand", generator=generate)],
        config=ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSA,
            profile=SearchProfile.GO_DEEP,
            execution_mode=ExecutionMode.ASK_TELL,
            budget=2,
            batch_size=2,
            best_k=2,
            metadata={"action_parent_states": parent_states},
        ),
    )

    explorer.ask_tell_batch(2)

    assert seen_parents == parent_states


def test_seeded_parent_states_override_backend_parent_selection() -> None:
    parent_states = ["seed-a", "seed-b"]
    seen_parents: list[str | None] = []

    def generate(
        parent_state: str | None,
        context: ExplorerContext,
    ) -> GenerationResult[str]:
        seen_parents.append(parent_state)
        return GenerationResult(state=f"{parent_state}/child", score=0.5)

    explorer = ABMCTSExplorer[str](
        task="forced seeded search",
        actions=[ActionSpec(name="expand", generator=generate)],
        config=ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSA,
            profile=SearchProfile.GO_DEEP,
            execution_mode=ExecutionMode.ASK_TELL,
            budget=2,
            batch_size=2,
            best_k=2,
            metadata={"action_parent_states": parent_states},
        ),
        backend=BackendWithParent(),
    )

    explorer.ask_tell_batch(2)

    assert seen_parents == parent_states


def test_force_root_parent_overrides_backend_parent_selection() -> None:
    seen_parents: list[str | None] = []

    def generate(
        parent_state: str | None,
        context: ExplorerContext,
    ) -> GenerationResult[str]:
        seen_parents.append(parent_state)
        return GenerationResult(state=f"child-{len(seen_parents)}", score=0.5)

    explorer = ABMCTSExplorer[str](
        task="forced root search",
        actions=[ActionSpec(name="expand", generator=generate)],
        config=ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSA,
            profile=SearchProfile.GO_WIDE,
            execution_mode=ExecutionMode.ASK_TELL,
            budget=2,
            batch_size=2,
            best_k=2,
            metadata={"force_root_parent": True},
        ),
        backend=BackendWithParent(),
    )

    explorer.ask_tell_batch(2)

    assert seen_parents == [None, None]
    assert {item["parent_id"] for item in explorer.node_logs} == {"root"}


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
