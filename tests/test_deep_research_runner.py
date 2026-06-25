from __future__ import annotations

import asyncio

from abmcts_explorer import (
    ActionSpec,
    DeepResearchRunner,
    DeepResearchRunnerConfig,
    DeepResearchLogger,
    DeepResearchState,
    ExplorerContext,
    GenerationResult,
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
            evidence_notes=tuple(f"根拠 {index}" for index, _ in enumerate(urls[:3])),
            source_urls=urls,
            open_questions=() if context.profile == SearchProfile.GO_WIDE else ("追加検証",),
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

    report = asyncio.run(runner.run("AIエージェント評価"))

    selected_profiles = [decision.profile for decision in report.profile_history]

    assert search_client.queries
    assert selected_profiles[0] == SearchProfile.GO_WIDE
    assert SearchProfile.GO_DEEP in selected_profiles
    assert report.web_results
    assert report.best_candidates
    assert "AIエージェント評価" in report.report
