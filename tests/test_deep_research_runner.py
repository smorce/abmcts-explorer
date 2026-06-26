from __future__ import annotations

import asyncio

from abmcts_explorer import (
    ActionSpec,
    DeepResearchRunner,
    DeepResearchRunnerConfig,
    DeepResearchState,
    ExplorerContext,
    GenerationResult,
    ProfileDecision,
    SearchProfile,
    WebSearchResponse,
    WebSearchResult,
)


class FakeSearchClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, limit: int = 5) -> WebSearchResponse:
        self.queries.append(query)
        return WebSearchResponse(
            success=True,
            results=[
                WebSearchResult(
                    title=f"title {len(self.queries)}",
                    url=f"https://example.com/{len(self.queries)}",
                    description=f"description for {query}",
                    position=1,
                )
            ],
        )


def make_research_action() -> ActionSpec[DeepResearchState]:
    calls = {"count": 0}

    def generate(
        parent_state: DeepResearchState | None,
        context: ExplorerContext,
    ) -> GenerationResult[DeepResearchState]:
        calls["count"] += 1
        web_results = context.metadata["web_results"]
        urls = tuple(item["url"] for item in web_results)
        state = DeepResearchState(
            topic=context.task,
            draft=(
                f"profile={context.profile.value}; "
                f"step={calls['count']}; "
                f"sources={len(urls)}"
            ),
            evidence_notes=tuple(f"evidence {index}" for index, _ in enumerate(urls[:3])),
            source_urls=urls,
            open_questions=() if context.profile == SearchProfile.GO_WIDE else ("verify more",),
            depth=0 if parent_state is None else parent_state.depth + 1,
        )
        score = 0.86 if context.profile == SearchProfile.GO_WIDE else 0.93
        return GenerationResult(state=state, score=score)

    return ActionSpec(name="deterministic_research", generator=generate)


def test_deep_research_runner_switches_from_wide_to_deep() -> None:
    search_client = FakeSearchClient()
    runner = DeepResearchRunner(
        search_client=search_client,
        actions=[make_research_action()],
        config=DeepResearchRunnerConfig(
            total_budget=6,
            epoch_budget=2,
            best_k=2,
            search_limit=1,
            min_wide_epochs=1,
            wide_batch_size=2,
            deep_batch_size=2,
        ),
    )

    report = asyncio.run(runner.run("AI agent evaluation"))

    selected_profiles = [decision.profile for decision in report.profile_history]

    assert search_client.queries
    assert selected_profiles[0] == SearchProfile.GO_WIDE
    assert SearchProfile.GO_DEEP in selected_profiles
    assert report.web_results
    assert report.best_candidates
    assert report.report


def test_go_deep_uses_all_tied_best_wide_candidates_as_parents() -> None:
    search_client = FakeSearchClient()
    deep_parent_drafts: list[str | None] = []
    calls = {"count": 0}

    def generate(
        parent_state: DeepResearchState | None,
        context: ExplorerContext,
    ) -> GenerationResult[DeepResearchState]:
        calls["count"] += 1
        if context.profile == SearchProfile.GO_DEEP:
            deep_parent_drafts.append(parent_state.draft if parent_state else None)

        draft = f"{context.profile.value}-{calls['count']}"
        state = DeepResearchState(
            topic=context.task,
            draft=draft,
            source_urls=("https://example.com/source",),
            depth=0 if parent_state is None else parent_state.depth + 1,
        )
        score = 0.95 if draft in {"go_wide-1", "go_wide-2"} else 0.80
        if context.profile == SearchProfile.GO_DEEP:
            score = 0.90
        return GenerationResult(state=state, score=score)

    class AlwaysDeepDecider:
        def decide(self, diagnostics, current_profile) -> ProfileDecision:
            return ProfileDecision(
                profile=SearchProfile.GO_DEEP,
                confidence=1.0,
                reasons=[],
                explanation="test forces deep",
            )

    runner = DeepResearchRunner(
        search_client=search_client,
        actions=[ActionSpec(name="tracked_research", generator=generate)],
        config=DeepResearchRunnerConfig(
            total_budget=4,
            epoch_budget=2,
            best_k=2,
            search_limit=1,
            min_wide_epochs=0,
            wide_batch_size=2,
            deep_batch_size=2,
        ),
        decider=AlwaysDeepDecider(),
    )

    asyncio.run(runner.run("seed test"))

    assert deep_parent_drafts
    assert set(deep_parent_drafts[:2]) == {"go_wide-1", "go_wide-2"}
