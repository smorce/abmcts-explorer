from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .core import (
    ABMCTSExplorer,
    ActionSpec,
    AlgorithmKind,
    ExecutionMode,
    ExplorerConfig,
    ExplorerContext,
    ExplorerObserver,
    GenerationResult,
    SearchProfile,
    call_llamas_server,
)
from .profile_decider import (
    ProfileDeciderConfig,
    ProfileDecision,
    RuleBasedProfileDecider,
    SearchDiagnostics,
)
from .research_logger import DeepResearchLogger
from .web_search import SearXNGSearchClient, WebSearchResponse, WebSearchResult


@dataclass(frozen=True)
class DeepResearchState:
    topic: str
    draft: str
    evidence_notes: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    depth: int = 0


@dataclass(frozen=True)
class DeepResearchRunnerConfig:
    total_budget: int = 80
    epoch_budget: int = 10
    best_k: int = 5
    search_limit: int = 5
    min_wide_epochs: int = 2
    wide_batch_size: int = 8
    deep_batch_size: int = 5
    report_model: str | None = None
    report_max_tokens: int = 4096


@dataclass(frozen=True)
class DeepResearchReport:
    topic: str
    report: str
    best_candidates: list[tuple[DeepResearchState, float]]
    diagnostics_history: list[SearchDiagnostics]
    profile_history: list[ProfileDecision]
    web_results: list[WebSearchResult]
    node_logs: list[dict[str, Any]] = field(default_factory=list)
    logger: DeepResearchLogger | None = None


class WebSearchClient(Protocol):
    def search(self, query: str, limit: int = 5) -> WebSearchResponse:
        ...


def build_deep_research_prompt(
    parent_state: DeepResearchState | None,
    context: ExplorerContext,
) -> list[dict[str, str]]:
    parent_state = _effective_parent_state(parent_state, context)
    web_results = context.metadata.get("web_results", [])
    web_context = "\n".join(
        f"- {item.get('title', '')}: {item.get('description', '')} ({item.get('url', '')})"
        for item in web_results
    )
    mode = context.profile.value

    if parent_state is None:
        user_content = f"""
調査テーマ:
{context.task}

探索モード:
{mode}

検索結果:
{web_context}

観点を広げながら、暫定回答、根拠メモ、出典URL、未解決論点を整理してください。
"""
    else:
        user_content = f"""
調査テーマ:
{context.task}

探索モード:
{mode}

現在の暫定回答:
{parent_state.draft}

既存の根拠:
{chr(10).join(parent_state.evidence_notes)}

既存の出典:
{chr(10).join(parent_state.source_urls)}

未解決論点:
{chr(10).join(parent_state.open_questions)}

追加検索結果:
{web_context}

GO_WIDE では別観点や反証候補を増やしてください。
GO_DEEP では根拠対応、矛盾確認、最終回答への統合を厚くしてください。
"""

    return [
        {
            "role": "system",
            "content": (
                "You are a rigorous DeepResearch assistant. "
                "Write in Japanese. Separate claims, evidence, sources, "
                "uncertainty, and next questions."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def parse_deep_research_state(
    raw: str,
    parent_state: DeepResearchState | None,
    context: ExplorerContext,
) -> DeepResearchState:
    parent_state = _effective_parent_state(parent_state, context)
    text = raw.strip()
    web_results = context.metadata.get("web_results", [])
    urls = tuple(
        str(item.get("url", ""))
        for item in web_results
        if item.get("url")
    )
    evidence = tuple(
        line.strip("- ").strip()
        for line in text.splitlines()
        if "根拠" in line or "出典" in line or "evidence" in line.lower()
    )
    open_questions = tuple(
        line.strip("- ").strip()
        for line in text.splitlines()
        if "未解決" in line or "疑問" in line or "question" in line.lower()
    )
    return DeepResearchState(
        topic=context.task,
        draft=text,
        evidence_notes=evidence,
        source_urls=urls,
        open_questions=open_questions,
        depth=0 if parent_state is None else parent_state.depth + 1,
    )


def _effective_parent_state(
    parent_state: DeepResearchState | None,
    context: ExplorerContext,
) -> DeepResearchState | None:
    if parent_state is not None:
        return parent_state
    if context.profile != SearchProfile.GO_DEEP:
        return None
    seed = context.metadata.get("deep_seed_state")
    return seed if isinstance(seed, DeepResearchState) else None


def score_deep_research_state(
    state: DeepResearchState,
    parent_state: DeepResearchState | None,
    context: ExplorerContext,
) -> float:
    if not state.draft.strip():
        return 0.0

    length_score = min(len(state.draft) / 5000, 1.0)
    evidence_score = min(len(state.evidence_notes) / 6, 1.0)
    source_score = min(len(set(state.source_urls)) / 5, 1.0)
    question_score = min(len(state.open_questions) / 4, 1.0)
    depth_bonus = min(state.depth / 4, 1.0)

    score = (
        length_score * 0.25
        + evidence_score * 0.25
        + source_score * 0.25
        + question_score * 0.15
        + depth_bonus * 0.10
    )
    return max(0.0, min(score, 1.0))


def default_deep_research_actions() -> list[ActionSpec[DeepResearchState]]:
    return [
        ActionSpec[DeepResearchState](
            name="broaden_research_angles",
            model="llama-research-fast",
            prompt_builder=build_deep_research_prompt,
            parser=parse_deep_research_state,
            scorer=score_deep_research_state,
            temperature=0.7,
            max_tokens=2200,
        ),
        ActionSpec[DeepResearchState](
            name="verify_and_synthesize",
            model="llama-research-precise",
            prompt_builder=build_deep_research_prompt,
            parser=parse_deep_research_state,
            scorer=score_deep_research_state,
            temperature=0.2,
            max_tokens=3200,
        ),
    ]


class DeepResearchRunner:
    def __init__(
        self,
        *,
        search_client: WebSearchClient | None = None,
        actions: list[ActionSpec[DeepResearchState]] | None = None,
        config: DeepResearchRunnerConfig | None = None,
        decider: RuleBasedProfileDecider | None = None,
        report_builder: Callable[[str, list[tuple[DeepResearchState, float]]], str]
        | None = None,
        logger: DeepResearchLogger | None = None,
        observer: ExplorerObserver[DeepResearchState] | None = None,
    ) -> None:
        self.search_client = search_client or SearXNGSearchClient()
        self.logger = logger or DeepResearchLogger()
        self.actions = self._wrap_actions(actions or default_deep_research_actions())
        self.config = config or DeepResearchRunnerConfig()
        self.decider = decider or RuleBasedProfileDecider(
            ProfileDeciderConfig(
                min_initial_nodes=self.config.epoch_budget,
                min_epochs_before_switch=1,
            )
        )
        self.report_builder = report_builder
        self.observer = observer

    def _wrap_actions(
        self,
        actions: list[ActionSpec[DeepResearchState]],
    ) -> list[ActionSpec[DeepResearchState]]:
        wrapped: list[ActionSpec[DeepResearchState]] = []
        for action in actions:
            prompt_builder = action.prompt_builder
            parser = action.parser
            generator = action.generator

            if prompt_builder is not None:

                def logging_prompt_builder(
                    parent_state: DeepResearchState | None,
                    context: ExplorerContext,
                    *,
                    _action_name: str = action.name,
                    _prompt_builder=prompt_builder,
                ) -> list[dict[str, str]]:
                    messages = _prompt_builder(parent_state, context)
                    self.logger.log_llm_messages(
                        action_name=_action_name,
                        profile=context.profile.value,
                        step_index=context.step_index,
                        messages=messages,
                    )
                    return messages

            else:
                logging_prompt_builder = None

            if parser is not None:

                def logging_parser(
                    raw: str,
                    parent_state: DeepResearchState | None,
                    context: ExplorerContext,
                    *,
                    _action_name: str = action.name,
                    _parser=parser,
                ) -> DeepResearchState:
                    self.logger.log_llm_response(
                        action_name=_action_name,
                        profile=context.profile.value,
                        step_index=context.step_index,
                        raw_response=raw,
                    )
                    return _parser(raw, parent_state, context)

            else:
                logging_parser = None

            if generator is not None:

                def logging_generator(
                    parent_state: DeepResearchState | None,
                    context: ExplorerContext,
                    *,
                    _action_name: str = action.name,
                    _generator=generator,
                ):
                    self.logger.log_event(
                        "generator_called",
                        {
                            "action_name": _action_name,
                            "profile": context.profile.value,
                            "step_index": context.step_index,
                        },
                    )
                    return _generator(parent_state, context)

            else:
                logging_generator = None

            wrapped.append(
                ActionSpec[DeepResearchState](
                    name=action.name,
                    generator=logging_generator,
                    model=action.model,
                    prompt_builder=logging_prompt_builder,
                    parser=logging_parser,
                    scorer=action.scorer,
                    temperature=action.temperature,
                    max_tokens=action.max_tokens,
                    enabled=action.enabled,
                    tags=set(action.tags),
                    llm_kwargs=dict(action.llm_kwargs),
                )
            )
        return wrapped

    async def run(self, topic: str) -> DeepResearchReport:
        self.logger.log_event(
            "run_started",
            {
                "topic": topic,
                "total_budget": self.config.total_budget,
                "epoch_budget": self.config.epoch_budget,
            },
        )
        metadata: dict[str, Any] = {"web_results": []}
        self._set_root_parent_metadata(metadata)
        explorer = ABMCTSExplorer[DeepResearchState](
            task=topic,
            actions=self.actions,
            config=self._config_for_profile(SearchProfile.GO_WIDE, metadata),
            observer=self.observer,
        )

        consumed = 0
        epoch_index = 0
        best: list[tuple[DeepResearchState, float]] = []
        web_results: list[WebSearchResult] = []
        diagnostics_history: list[SearchDiagnostics] = []
        profile_history: list[ProfileDecision] = []

        while consumed < self.config.total_budget:
            self.logger.log_event(
                "epoch_started",
                {
                    "epoch_index": epoch_index,
                    "profile": explorer.config.profile.value,
                    "consumed_budget": consumed,
                },
            )
            web_results = self._refresh_web_results(
                topic=topic,
                profile=explorer.config.profile,
                best=best,
                existing=web_results,
            )
            metadata["web_results"] = [self._web_result_to_dict(item) for item in web_results]
            self.logger.log_event(
                "web_results_refreshed",
                {
                    "epoch_index": epoch_index,
                    "web_result_count": len(web_results),
                    "top_urls": [item.url for item in web_results[:5]],
                },
            )

            epoch_budget = min(
                self.config.epoch_budget,
                self.config.total_budget - consumed,
            )

            if explorer.config.execution_mode == ExecutionMode.ASYNC_ASK_TELL:
                best = await explorer.run_async(budget=epoch_budget)
            else:
                best = await asyncio.to_thread(explorer.run, epoch_budget)

            consumed += epoch_budget
            diagnostics = self._diagnose(
                total_budget=self.config.total_budget,
                consumed_budget=consumed,
                best=best,
                web_results=web_results,
            )
            diagnostics_history.append(diagnostics)
            self.logger.log_event(
                "diagnostics_collected",
                {
                    "epoch_index": epoch_index,
                    "total_nodes": diagnostics.total_nodes,
                    "best_score": diagnostics.best_score,
                    "top_k_mean_score": diagnostics.top_k_mean_score,
                    "top_k_score_std": diagnostics.top_k_score_std,
                    "diversity_score": diagnostics.diversity_score,
                    "uncertainty_score": diagnostics.uncertainty_score,
                    "remaining_budget_ratio": diagnostics.remaining_budget_ratio,
                    "metadata": diagnostics.metadata,
                },
            )

            decision = self.decider.decide(
                diagnostics=diagnostics,
                current_profile=explorer.config.profile,
            )

            if (
                explorer.config.profile == SearchProfile.GO_WIDE
                and epoch_index + 1 < self.config.min_wide_epochs
            ):
                decision = ProfileDecision(
                    profile=SearchProfile.GO_WIDE,
                    confidence=max(decision.confidence, 0.75),
                    reasons=decision.reasons,
                    explanation=decision.explanation + " | forced_initial_go_wide",
                    suggested_batch_size=self.config.wide_batch_size,
                    suggested_budget=decision.suggested_budget,
                )
            elif explorer.config.profile == SearchProfile.GO_DEEP:
                decision = ProfileDecision(
                    profile=SearchProfile.GO_WIDE,
                    confidence=max(decision.confidence, 0.9),
                    reasons=decision.reasons,
                    explanation=decision.explanation + " | alternate_after_go_deep",
                    suggested_batch_size=self.config.wide_batch_size,
                    suggested_budget=decision.suggested_budget,
                )
            elif explorer.config.profile == SearchProfile.GO_WIDE and best:
                decision = ProfileDecision(
                    profile=SearchProfile.GO_DEEP,
                    confidence=max(decision.confidence, 0.9),
                    reasons=decision.reasons,
                    explanation=decision.explanation + " | alternate_after_go_wide",
                    suggested_batch_size=self.config.deep_batch_size,
                    suggested_budget=decision.suggested_budget,
                )

            profile_history.append(decision)
            self.logger.log_event(
                "profile_decided",
                {
                    "epoch_index": epoch_index,
                    "profile": decision.profile.value,
                    "confidence": decision.confidence,
                    "reasons": [reason.value for reason in decision.reasons],
                    "explanation": decision.explanation,
                    "suggested_batch_size": decision.suggested_batch_size,
                },
            )

            if decision.profile != explorer.config.profile:
                if decision.profile == SearchProfile.GO_DEEP and best:
                    best_score, deep_seed_states = self._tied_best_states(best)
                    self._set_seed_parent_metadata(
                        metadata,
                        states=deep_seed_states,
                        seed_key="deep_seed_state",
                    )
                    self.logger.log_event(
                        "deep_seed_selected",
                        {
                            "epoch_index": epoch_index,
                            "seed_count": len(deep_seed_states),
                            "seed_score": best_score,
                            "seed_depth": deep_seed_states[0].depth,
                            "seed_preview": deep_seed_states[0].draft[:240],
                        },
                    )
                elif decision.profile == SearchProfile.GO_WIDE and best:
                    best_score, wide_parent_states = self._tied_best_states(best)
                    self._set_seed_parent_metadata(
                        metadata,
                        states=wide_parent_states,
                        seed_key="wide_seed_state",
                    )
                    metadata.pop("deep_seed_state", None)
                    self.logger.log_event(
                        "wide_parent_selected",
                        {
                            "epoch_index": epoch_index,
                            "parent_count": len(wide_parent_states),
                            "parent_score": best_score,
                            "parent_depth": wide_parent_states[0].depth,
                            "parent_preview": wide_parent_states[0].draft[:240],
                        },
                    )
                else:
                    self._set_root_parent_metadata(metadata)

                self.logger.log_event(
                    "profile_switched",
                    {
                        "from": explorer.config.profile.value,
                        "to": decision.profile.value,
                    },
                )
                explorer.switch_config(
                    self._config_for_profile(decision.profile, metadata),
                    preserve_tree=False,
                )

            epoch_index += 1

        report = self._build_report(topic, best)
        self.logger.log_event(
            "run_finished",
            {
                "profile_history": [decision.profile.value for decision in profile_history],
                "web_result_count": len(web_results),
                "best_candidate_count": len(best),
                "report_chars": len(report),
            },
        )
        if self.observer is not None:
            self.observer.on_run_event(
                {
                    "event_type": "run_finished",
                    "step_index": consumed,
                    "profile": profile_history[-1].profile.value
                    if profile_history
                    else SearchProfile.GO_WIDE.value,
                    "algorithm": "deep_research",
                    "payload": {
                        "web_result_count": len(web_results),
                        "best_candidate_count": len(best),
                        "report_chars": len(report),
                    },
                }
            )
        return DeepResearchReport(
            topic=topic,
            report=report,
            best_candidates=best,
            diagnostics_history=diagnostics_history,
            profile_history=profile_history,
            web_results=web_results,
            node_logs=explorer.node_logs,
            logger=self.logger,
        )

    def _config_for_profile(
        self,
        profile: SearchProfile,
        metadata: dict[str, Any],
    ) -> ExplorerConfig:
        if profile == SearchProfile.GO_DEEP:
            return ExplorerConfig(
                algorithm_kind=AlgorithmKind.ABMCTSM,
                profile=SearchProfile.GO_DEEP,
                execution_mode=ExecutionMode.ASK_TELL,
                budget=self.config.epoch_budget,
                batch_size=self.config.deep_batch_size,
                best_k=self.config.best_k,
                algo_kwargs={"max_process_workers": self.config.deep_batch_size},
                metadata=metadata,
            )

        if (
            "action_parent_state" not in metadata
            and "action_parent_states" not in metadata
        ):
            self._set_root_parent_metadata(metadata)

        return ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSA,
            profile=SearchProfile.GO_WIDE,
            execution_mode=ExecutionMode.ASYNC_ASK_TELL,
            budget=self.config.epoch_budget,
            batch_size=self.config.wide_batch_size,
            best_k=self.config.best_k,
            metadata=metadata,
        )

    @staticmethod
    def _tied_best_states(
        best: list[tuple[DeepResearchState, float]],
    ) -> tuple[float, list[DeepResearchState]]:
        best_score = max(score for _, score in best)
        return (
            best_score,
            [
                state
                for state, score in best
                if abs(score - best_score) <= 1e-9
            ],
        )

    @staticmethod
    def _set_seed_parent_metadata(
        metadata: dict[str, Any],
        *,
        states: list[DeepResearchState],
        seed_key: str,
    ) -> None:
        metadata.pop("force_root_parent", None)
        metadata.pop("wide_seed_state", None)
        if seed_key != "deep_seed_state":
            metadata.pop("deep_seed_state", None)
        seed_state = states[0]
        metadata[seed_key] = seed_state
        metadata["action_parent_state"] = seed_state
        metadata["action_parent_states"] = states
        metadata["observer_parent_state"] = seed_state

    @staticmethod
    def _set_root_parent_metadata(metadata: dict[str, Any]) -> None:
        metadata["force_root_parent"] = True
        metadata.pop("deep_seed_state", None)
        metadata.pop("wide_seed_state", None)
        metadata.pop("action_parent_state", None)
        metadata.pop("action_parent_states", None)
        metadata.pop("observer_parent_state", None)

    def _refresh_web_results(
        self,
        *,
        topic: str,
        profile: SearchProfile,
        best: list[tuple[DeepResearchState, float]],
        existing: list[WebSearchResult],
    ) -> list[WebSearchResult]:
        queries = self._queries_for_epoch(topic=topic, profile=profile, best=best)
        by_url = {item.url: item for item in existing if item.url}

        for query in queries:
            self.logger.log_event(
                "web_search_started",
                {
                    "query": query,
                    "profile": profile.value,
                    "limit": self.config.search_limit,
                },
            )
            response = self.search_client.search(query, limit=self.config.search_limit)
            self.logger.log_event(
                "web_search_finished",
                {
                    "query": query,
                    "success": response.success,
                    "result_count": len(response.results),
                    "error": response.error,
                },
            )
            if not response.success:
                continue
            for item in response.results:
                if item.url and item.url not in by_url:
                    by_url[item.url] = item

        return list(by_url.values())

    def _queries_for_epoch(
        self,
        *,
        topic: str,
        profile: SearchProfile,
        best: list[tuple[DeepResearchState, float]],
    ) -> list[str]:
        if profile == SearchProfile.GO_DEEP and best:
            questions = [
                question
                for state, _ in best[:2]
                for question in state.open_questions[:2]
            ]
            if questions:
                return [f"{topic} {question}" for question in questions[:3]]
            return [
                *self._base_search_queries(topic),
                f"{self._compact_topic_query(topic)} 根拠",
                f"{self._compact_topic_query(topic)} 反証",
                f"{self._compact_topic_query(topic)} 最新",
            ]

        return [
            *self._base_search_queries(topic),
            f"{self._compact_topic_query(topic)} 論点",
            f"{self._compact_topic_query(topic)} 課題",
            f"{self._compact_topic_query(topic)} 反論",
        ]

    def _base_search_queries(self, topic: str) -> list[str]:
        compact = self._compact_topic_query(topic)
        queries = [compact]

        if "オンポリシー" in topic or "蒸留" in topic:
            queries.extend(
                [
                    "オンポリシー蒸留 on-policy distillation",
                    "on-policy distillation language models",
                ]
            )

        if "エキスパート" in topic or "専門モデル" in topic:
            queries.extend(
                [
                    "expert models distillation mixture of experts",
                    "specialist models knowledge distillation",
                ]
            )

        deduped: list[str] = []
        for query in queries:
            if query and query not in deduped:
                deduped.append(query)
        return deduped[:5]

    @staticmethod
    def _compact_topic_query(topic: str) -> str:
        if len(topic) <= 80:
            return topic

        terms = [
            "エキスパートモデル",
            "専門モデル",
            "オンポリシー蒸留",
            "on-policy distillation",
            "knowledge distillation",
            "mixture of experts",
            "MoE",
            "個別学習",
            "統合",
        ]
        matched = [term for term in terms if term.lower() in topic.lower()]
        if matched:
            return " ".join(dict.fromkeys(matched))

        return topic[:80]

    def _diagnose(
        self,
        *,
        total_budget: int,
        consumed_budget: int,
        best: list[tuple[DeepResearchState, float]],
        web_results: list[WebSearchResult],
    ) -> SearchDiagnostics:
        scores = [score for _, score in best]
        best_score = max(scores) if scores else 0.0
        mean_score = sum(scores) / len(scores) if scores else 0.0
        score_std = self._score_std(scores)
        depths = [state.depth for state, _ in best]
        source_count = len({url for state, _ in best for url in state.source_urls})
        open_question_count = sum(len(state.open_questions) for state, _ in best)
        draft_count = len({state.draft[:200] for state, _ in best})

        diversity_denominator = max(len(best), 1)
        diversity_score = min(
            1.0,
            (draft_count / diversity_denominator * 0.6)
            + (min(source_count, len(web_results)) / max(len(web_results), 1) * 0.4),
        )
        uncertainty_score = min(1.0, open_question_count / max(len(best) * 4, 1))

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
            remaining_budget_ratio=max(
                0.0,
                (total_budget - consumed_budget) / max(total_budget, 1),
            ),
            metadata={
                "web_result_count": len(web_results),
                "open_question_count": open_question_count,
            },
        )

    def _build_report(
        self,
        topic: str,
        best: list[tuple[DeepResearchState, float]],
    ) -> str:
        if self.report_builder is not None:
            return self.report_builder(topic, best)

        if self.config.report_model:
            candidate_text = "\n\n".join(
                f"Candidate score={score:.3f}\n{state.draft}"
                for state, score in best
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a DeepResearch report writer. "
                        "Write a detailed Japanese report with claims, evidence, "
                        "uncertainties, and next actions."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Theme:\n{topic}\n\nCandidates:\n{candidate_text}",
                },
            ]
            self.logger.log_llm_messages(
                action_name="final_report",
                profile="report",
                step_index=-1,
                messages=messages,
            )
            return call_llamas_server(
                model=self.config.report_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a DeepResearch report writer. "
                            "Write a detailed Japanese report with claims, evidence, "
                            "uncertainties, and next actions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"テーマ:\n{topic}\n\n探索候補:\n{candidate_text}",
                    },
                ],
                temperature=0.2,
                max_tokens=self.config.report_max_tokens,
            )

        lines = [f"# {topic}", "", "## 要約"]
        if not best:
            return "\n".join(lines + ["有効な候補が生成されませんでした。"])

        top_state, top_score = best[0]
        lines.extend([f"最高スコア: {top_score:.3f}", "", top_state.draft, ""])
        lines.append("## 根拠メモ")
        lines.extend(f"- {item}" for item in top_state.evidence_notes[:12])
        lines.append("")
        lines.append("## 参照URL")
        lines.extend(f"- {url}" for url in dict.fromkeys(top_state.source_urls))
        lines.append("")
        lines.append("## 未解決論点")
        lines.extend(f"- {item}" for item in top_state.open_questions[:12])
        return "\n".join(lines)

    @staticmethod
    def _score_std(scores: list[float]) -> float:
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        return variance**0.5

    @staticmethod
    def _web_result_to_dict(result: WebSearchResult) -> dict[str, str]:
        return {
            "title": result.title,
            "url": result.url,
            "description": result.description,
        }
