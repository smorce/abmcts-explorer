from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Generic, Optional, Protocol, TypeVar, Union

import asyncio

from .llama_server import LlamaServerEnvConfig, messages_to_prompt, run_llama_server_completion_sync


StateT = TypeVar("StateT")


class AlgorithmKind(str, Enum):
    ABMCTSM = "abmctsm"
    ABMCTSA = "abmctsa"
    STANDARD_MCTS = "standard_mcts"
    CUSTOM_LIGHT = "custom_light"


class SearchProfile(str, Enum):
    GO_DEEP = "go_deep"
    GO_WIDE = "go_wide"
    BALANCED = "balanced"
    HYBRID = "hybrid"


class ExecutionMode(str, Enum):
    STEP = "step"
    ASK_TELL = "ask_tell"
    ASYNC_ASK_TELL = "async_ask_tell"


@dataclass(frozen=True)
class GenerationResult(Generic[StateT]):
    state: StateT
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExplorerContext:
    task: str
    step_index: int
    profile: SearchProfile
    metadata: dict[str, Any] = field(default_factory=dict)


MaybeAwaitable = Union[GenerationResult[StateT], Awaitable[GenerationResult[StateT]]]

NodeGenerator = Callable[[Optional[StateT], ExplorerContext], MaybeAwaitable[StateT]]
PromptBuilder = Callable[[Optional[StateT], ExplorerContext], list[dict[str, str]]]
StateParser = Callable[[str, Optional[StateT], ExplorerContext], StateT]
StateScorer = Callable[[StateT, Optional[StateT], ExplorerContext], float]


def call_llamas_server(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    server_url: str | None = None,
    timeout_sec: int = 120,
    **kwargs: Any,
) -> str:
    config = LlamaServerEnvConfig.from_env().with_overrides(
        model=model,
        base_url_no_v1=server_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_sec,
        enable_thinking=kwargs.get("enable_thinking"),
        top_p=kwargs.get("top_p"),
        top_k=kwargs.get("top_k"),
        min_p=kwargs.get("min_p"),
    )
    return run_llama_server_completion_sync(config, messages_to_prompt(messages))


@dataclass
class ActionSpec(Generic[StateT]):
    name: str
    generator: NodeGenerator[StateT] | None = None
    model: str | None = None
    prompt_builder: PromptBuilder[StateT] | None = None
    parser: StateParser[StateT] | None = None
    scorer: StateScorer[StateT] | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    enabled: bool = True
    tags: set[str] = field(default_factory=set)
    llm_kwargs: dict[str, Any] = field(default_factory=dict)

    def run_sync(
        self,
        parent_state: StateT | None,
        context: ExplorerContext,
    ) -> GenerationResult[StateT]:
        if self.generator is not None:
            result = self.generator(parent_state, context)
            if inspect.isawaitable(result):
                raise RuntimeError(
                    f"Action {self.name} is async. Use run_async / ASYNC_ASK_TELL."
                )
            return result

        if not all([self.model, self.prompt_builder, self.parser, self.scorer]):
            raise ValueError(
                f"Action {self.name} requires either generator or "
                "model + prompt_builder + parser + scorer."
            )

        messages = self.prompt_builder(parent_state, context)
        raw = call_llamas_server(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.llm_kwargs,
        )

        state = self.parser(raw, parent_state, context)
        score = self.scorer(state, parent_state, context)

        return GenerationResult(
            state=state,
            score=score,
            metadata={
                "action": self.name,
                "model": self.model,
                "raw_response": raw,
            },
        )

    async def run_async(
        self,
        parent_state: StateT | None,
        context: ExplorerContext,
    ) -> GenerationResult[StateT]:
        if self.generator is not None:
            result = self.generator(parent_state, context)
            if inspect.isawaitable(result):
                return await result
            return result

        return await asyncio.to_thread(self.run_sync, parent_state, context)


class SearchBackend(Protocol, Generic[StateT]):
    name: str

    def init_tree(self) -> Any:
        ...

    def step(
        self,
        tree: Any,
        generate_fns: dict[str, Callable[[StateT | None], tuple[StateT, float]]],
    ) -> Any:
        ...

    def ask_batch(
        self,
        tree: Any,
        batch_size: int,
        actions: list[str],
    ) -> tuple[Any, list[Any]]:
        ...

    def tell(
        self,
        tree: Any,
        trial_id: Any,
        result: tuple[StateT, float],
    ) -> Any:
        ...

    def top_k(
        self,
        tree: Any,
        k: int,
    ) -> list[tuple[StateT, float]]:
        ...

    def is_tree_compatible_with(self, other: "SearchBackend[StateT]") -> bool:
        ...


@dataclass
class TreeQuestBackend(Generic[StateT]):
    algorithm_kind: AlgorithmKind
    algo_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import treequest as tq

        self._tq = tq
        self.algo = self._build_algo()
        self.name = self.algorithm_kind.value

    def _build_algo(self) -> Any:
        tq = self._tq

        if self.algorithm_kind == AlgorithmKind.ABMCTSA:
            return tq.ABMCTSA(**self.algo_kwargs)

        if self.algorithm_kind == AlgorithmKind.ABMCTSM:
            return tq.ABMCTSM(**self.algo_kwargs)

        if self.algorithm_kind == AlgorithmKind.STANDARD_MCTS:
            return tq.StandardMCTS(**self.algo_kwargs)

        raise ValueError(f"Unsupported TreeQuest algorithm: {self.algorithm_kind}")

    def init_tree(self) -> Any:
        return self.algo.init_tree()

    def step(
        self,
        tree: Any,
        generate_fns: dict[str, Callable[[StateT | None], tuple[StateT, float]]],
    ) -> Any:
        return self.algo.step(tree, generate_fns)

    def ask_batch(
        self,
        tree: Any,
        batch_size: int,
        actions: list[str],
    ) -> tuple[Any, list[Any]]:
        return self.algo.ask_batch(tree, batch_size, actions)

    def tell(
        self,
        tree: Any,
        trial_id: Any,
        result: tuple[StateT, float],
    ) -> Any:
        return self.algo.tell(tree, trial_id, result)

    def top_k(
        self,
        tree: Any,
        k: int,
    ) -> list[tuple[StateT, float]]:
        return self._tq.top_k(tree, self.algo, k=k)

    def is_tree_compatible_with(self, other: SearchBackend[StateT]) -> bool:
        return getattr(other, "name", None) == self.name


@dataclass(frozen=True)
class ExplorerConfig:
    algorithm_kind: AlgorithmKind = AlgorithmKind.ABMCTSA
    profile: SearchProfile = SearchProfile.BALANCED
    execution_mode: ExecutionMode = ExecutionMode.ASK_TELL
    budget: int = 50
    batch_size: int = 5
    validate_score_range: bool = True
    best_k: int = 5
    algo_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExplorerPreset:
    @staticmethod
    def go_deep(
        *,
        budget: int = 40,
        batch_size: int = 5,
    ) -> ExplorerConfig:
        return ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSM,
            profile=SearchProfile.GO_DEEP,
            execution_mode=ExecutionMode.ASK_TELL,
            budget=budget,
            batch_size=batch_size,
            algo_kwargs={"max_process_workers": batch_size},
        )

    @staticmethod
    def go_wide(
        *,
        budget: int = 200,
        batch_size: int = 32,
    ) -> ExplorerConfig:
        return ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSA,
            profile=SearchProfile.GO_WIDE,
            execution_mode=ExecutionMode.ASYNC_ASK_TELL,
            budget=budget,
            batch_size=batch_size,
        )

    @staticmethod
    def balanced(
        *,
        budget: int = 80,
        batch_size: int = 8,
    ) -> ExplorerConfig:
        return ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSA,
            profile=SearchProfile.BALANCED,
            execution_mode=ExecutionMode.ASK_TELL,
            budget=budget,
            batch_size=batch_size,
        )


class ABMCTSExplorer(Generic[StateT]):
    def __init__(
        self,
        *,
        task: str,
        actions: list[ActionSpec[StateT]],
        config: ExplorerConfig,
        backend: SearchBackend[StateT] | None = None,
    ) -> None:
        if not actions:
            raise ValueError("At least one ActionSpec is required.")

        self.task = task
        self.config = config
        self.actions: dict[str, ActionSpec[StateT]] = {
            action.name: action for action in actions
        }
        self.backend: SearchBackend[StateT] = backend or TreeQuestBackend(
            algorithm_kind=config.algorithm_kind,
            algo_kwargs=config.algo_kwargs,
        )
        self.tree = self.backend.init_tree()
        self.step_index = 0
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.tree = self.backend.init_tree()
        self.step_index = 0
        self.history.clear()

    def active_actions(self) -> dict[str, ActionSpec[StateT]]:
        return {
            name: action
            for name, action in self.actions.items()
            if action.enabled
        }

    def _context(self) -> ExplorerContext:
        return ExplorerContext(
            task=self.task,
            step_index=self.step_index,
            profile=self.config.profile,
            metadata=self.config.metadata,
        )

    def _validate_score(self, score: float) -> None:
        if not self.config.validate_score_range:
            return

        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Score must be normalized to [0, 1], got {score}.")

    def _make_generate_fns(
        self,
    ) -> dict[str, Callable[[StateT | None], tuple[StateT, float]]]:
        context = self._context()
        generate_fns: dict[str, Callable[[StateT | None], tuple[StateT, float]]] = {}

        for name, action in self.active_actions().items():

            def generate(
                parent_state: StateT | None,
                _action: ActionSpec[StateT] = action,
            ) -> tuple[StateT, float]:
                result = _action.run_sync(parent_state, context)
                self._validate_score(result.score)
                return result.state, result.score

            generate_fns[name] = generate

        return generate_fns

    def step(self, n: int = 1) -> None:
        if n <= 0:
            return

        for _ in range(n):
            generate_fns = self._make_generate_fns()
            self.tree = self.backend.step(self.tree, generate_fns)
            self.step_index += 1
            self._record_event("step", {"n": 1})

    def ask_tell_batch(self, batch_size: int | None = None) -> None:
        batch_size = batch_size or self.config.batch_size
        actions = list(self.active_actions().keys())

        if not actions:
            raise ValueError("No active actions.")

        self.tree, trials = self.backend.ask_batch(self.tree, batch_size, actions)
        context = self._context()

        for trial in trials:
            action = self.actions[trial.action]
            result = action.run_sync(trial.parent_state, context)
            self._validate_score(result.score)
            self.tree = self.backend.tell(
                self.tree,
                trial.trial_id,
                (result.state, result.score),
            )
            self.step_index += 1

        self._record_event(
            "ask_tell_batch",
            {"batch_size": batch_size, "num_trials": len(trials)},
        )

    async def ask_tell_batch_async(self, batch_size: int | None = None) -> None:
        batch_size = batch_size or self.config.batch_size
        actions = list(self.active_actions().keys())

        if not actions:
            raise ValueError("No active actions.")

        self.tree, trials = self.backend.ask_batch(self.tree, batch_size, actions)
        context = self._context()

        async def run_trial(trial: Any) -> tuple[Any, GenerationResult[StateT]]:
            action = self.actions[trial.action]
            result = await action.run_async(trial.parent_state, context)
            return trial, result

        tasks = [asyncio.create_task(run_trial(trial)) for trial in trials]

        for task in asyncio.as_completed(tasks):
            trial, result = await task
            self._validate_score(result.score)
            self.tree = self.backend.tell(
                self.tree,
                trial.trial_id,
                (result.state, result.score),
            )
            self.step_index += 1

        self._record_event(
            "ask_tell_batch_async",
            {"batch_size": batch_size, "num_trials": len(trials)},
        )

    def run(self, budget: int | None = None) -> list[tuple[StateT, float]]:
        budget = budget or self.config.budget
        consumed = 0

        while consumed < budget:
            remaining = budget - consumed

            if self.config.execution_mode == ExecutionMode.STEP:
                self.step(1)
                consumed += 1
            elif self.config.execution_mode == ExecutionMode.ASK_TELL:
                batch = min(self.config.batch_size, remaining)
                self.ask_tell_batch(batch)
                consumed += batch
            else:
                raise RuntimeError("ASYNC_ASK_TELL requires run_async().")

        return self.best(self.config.best_k)

    async def run_async(self, budget: int | None = None) -> list[tuple[StateT, float]]:
        budget = budget or self.config.budget
        consumed = 0

        while consumed < budget:
            remaining = budget - consumed

            if self.config.execution_mode == ExecutionMode.STEP:
                await asyncio.to_thread(self.step, 1)
                consumed += 1
            elif self.config.execution_mode == ExecutionMode.ASK_TELL:
                batch = min(self.config.batch_size, remaining)
                await asyncio.to_thread(self.ask_tell_batch, batch)
                consumed += batch
            elif self.config.execution_mode == ExecutionMode.ASYNC_ASK_TELL:
                batch = min(self.config.batch_size, remaining)
                await self.ask_tell_batch_async(batch)
                consumed += batch
            else:
                raise ValueError(f"Unknown execution mode: {self.config.execution_mode}")

        return self.best(self.config.best_k)

    def best(self, k: int = 1) -> list[tuple[StateT, float]]:
        return self.backend.top_k(self.tree, k=k)

    def switch_config(
        self,
        new_config: ExplorerConfig,
        *,
        preserve_tree: bool = True,
    ) -> None:
        new_backend = TreeQuestBackend(
            algorithm_kind=new_config.algorithm_kind,
            algo_kwargs=new_config.algo_kwargs,
        )

        if preserve_tree and self.backend.is_tree_compatible_with(new_backend):
            self.backend = new_backend
            self.config = new_config
            return

        self.backend = new_backend
        self.config = new_config
        self.tree = self.backend.init_tree()
        self.step_index = 0

    def _record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.history.append(
            {
                "event_type": event_type,
                "step_index": self.step_index,
                "profile": self.config.profile.value,
                "algorithm": self.config.algorithm_kind.value,
                "payload": payload,
            }
        )
