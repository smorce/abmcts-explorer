from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field, is_dataclass
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


class ExplorerObserver(Protocol, Generic[StateT]):
    def on_trial_started(
        self,
        *,
        parent_state: StateT | None,
        context: ExplorerContext,
        action_name: str,
        algorithm: str,
    ) -> None:
        ...

    def on_node_generated(
        self,
        *,
        parent_state: StateT | None,
        result: GenerationResult[StateT],
        context: ExplorerContext,
        action_name: str,
        algorithm: str,
    ) -> None:
        ...

    def on_run_event(self, event: dict[str, Any]) -> None:
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
        observer: ExplorerObserver[StateT] | None = None,
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
        self.observer = observer
        self.tree = self.backend.init_tree()
        self.step_index = 0
        self.history: list[dict[str, Any]] = []
        self.node_logs: list[dict[str, Any]] = []
        self._state_to_node_id: dict[int, str] = {}
        self._node_depths: dict[str, int] = {"root": 0}
        self._next_node_id = 1

    def reset(self) -> None:
        self.tree = self.backend.init_tree()
        self.step_index = 0
        self.history.clear()
        self.node_logs.clear()
        self._state_to_node_id.clear()
        self._node_depths = {"root": 0}
        self._next_node_id = 1

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
                _action_name: str = name,
            ) -> tuple[StateT, float]:
                action_parent = self._action_parent_state(parent_state, context)
                observer_parent = self._observer_parent_state(action_parent, context)
                self._notify_trial_started(observer_parent, context, _action_name)
                result = _action.run_sync(action_parent, context)
                self._validate_score(result.score)
                self._notify_node(observer_parent, result, context, _action_name)
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

        for trial_index, trial in enumerate(trials):
            action = self.actions[trial.action]
            action_parent = self._action_parent_state(
                trial.parent_state,
                context,
                trial_index=trial_index,
            )
            observer_parent = self._observer_parent_state(action_parent, context)
            self._notify_trial_started(observer_parent, context, trial.action)
            result = action.run_sync(action_parent, context)
            self._validate_score(result.score)
            self.tree = self.backend.tell(
                self.tree,
                trial.trial_id,
                (result.state, result.score),
            )
            self._notify_node(observer_parent, result, context, trial.action)
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

        async def run_trial(
            trial_index: int,
            trial: Any,
        ) -> tuple[Any, GenerationResult[StateT], StateT | None]:
            action = self.actions[trial.action]
            action_parent = self._action_parent_state(
                trial.parent_state,
                context,
                trial_index=trial_index,
            )
            observer_parent = self._observer_parent_state(action_parent, context)
            self._notify_trial_started(observer_parent, context, trial.action)
            result = await action.run_async(action_parent, context)
            return trial, result, observer_parent

        tasks = [
            asyncio.create_task(run_trial(trial_index, trial))
            for trial_index, trial in enumerate(trials)
        ]

        for task in asyncio.as_completed(tasks):
            trial, result, observer_parent = await task
            self._validate_score(result.score)
            self.tree = self.backend.tell(
                self.tree,
                trial.trial_id,
                (result.state, result.score),
            )
            self._notify_node(observer_parent, result, context, trial.action)
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
        for candidate_k in range(k, 0, -1):
            try:
                return self.backend.top_k(self.tree, k=candidate_k)
            except RuntimeError as exc:
                if "cannot extract top" not in str(exc):
                    raise
        return []

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
        event = {
            "event_type": event_type,
            "step_index": self.step_index,
            "profile": self.config.profile.value,
            "algorithm": self.config.algorithm_kind.value,
            "payload": payload,
        }
        self.history.append(event)
        if self.observer is not None:
            self.observer.on_run_event(event)

    def _notify_node(
        self,
        parent_state: StateT | None,
        result: GenerationResult[StateT],
        context: ExplorerContext,
        action_name: str,
    ) -> None:
        node_id = f"n{self._next_node_id}"
        self._next_node_id += 1
        parent_id = (
            self._state_to_node_id.get(id(parent_state))
            if parent_state is not None
            else "root"
        )
        parent_id = parent_id or "root"
        depth = self._node_depths.get(parent_id, 0) + 1
        self._state_to_node_id[id(result.state)] = node_id
        self._node_depths[node_id] = depth
        self.node_logs.append(
            {
                "id": node_id,
                "parent_id": parent_id,
                "score": result.score,
                "action": action_name,
                "profile": context.profile.value,
                "algorithm": self.config.algorithm_kind.value,
                "step_index": context.step_index,
                "depth": depth,
                "state_repr": repr(result.state),
                "metadata": self._json_safe(result.metadata),
            }
        )

        if self.observer is None:
            return

        self.observer.on_node_generated(
            parent_state=parent_state,
            result=result,
            context=context,
            action_name=action_name,
            algorithm=self.config.algorithm_kind.value,
        )

    def _notify_trial_started(
        self,
        parent_state: StateT | None,
        context: ExplorerContext,
        action_name: str,
    ) -> None:
        if self.observer is None:
            return

        self.observer.on_trial_started(
            parent_state=parent_state,
            context=context,
            action_name=action_name,
            algorithm=self.config.algorithm_kind.value,
        )

    def _observer_parent_state(
        self,
        parent_state: StateT | None,
        context: ExplorerContext,
    ) -> StateT | None:
        if context.metadata.get("force_root_parent"):
            return None
        if parent_state is not None:
            return parent_state
        observer_parent = context.metadata.get("observer_parent_state")
        return observer_parent

    def _action_parent_state(
        self,
        parent_state: StateT | None,
        context: ExplorerContext,
        *,
        trial_index: int = 0,
    ) -> StateT | None:
        if context.metadata.get("force_root_parent"):
            return None
        action_parents = context.metadata.get("action_parent_states")
        if isinstance(action_parents, list) and action_parents:
            return action_parents[trial_index % len(action_parents)]
        if "action_parent_state" in context.metadata:
            return context.metadata.get("action_parent_state")
        if parent_state is not None:
            return parent_state
        action_parent = context.metadata.get("action_parent_state")
        return action_parent

    def _json_safe(self, value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return self._json_safe(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)
