from __future__ import annotations

import asyncio

from abmcts_explorer import (
    ActionSpec,
    DeepResearchLogger,
    DeepResearchRunner,
    DeepResearchRunnerConfig,
    DeepResearchState,
    ExplorerContext,
    GenerationResult,
    WebSearchResponse,
    WebSearchResult,
)


class FakeSearchClient:
    def search(self, query: str, limit: int = 5) -> WebSearchResponse:
        return WebSearchResponse(
            success=True,
            results=[
                WebSearchResult(
                    title="title",
                    url=f"https://example.com/{len(query)}",
                    description=query,
                    position=1,
                )
            ],
        )


def make_action() -> ActionSpec[DeepResearchState]:
    def generate(
        parent_state: DeepResearchState | None,
        context: ExplorerContext,
    ) -> GenerationResult[DeepResearchState]:
        state = DeepResearchState(
            topic=context.task,
            draft=f"profile={context.profile.value}",
            source_urls=tuple(
                item["url"] for item in context.metadata.get("web_results", [])
            ),
            depth=0 if parent_state is None else parent_state.depth + 1,
        )
        return GenerationResult(state=state, score=0.8)

    return ActionSpec(name="logged_generator", generator=generate)


def test_deep_research_logger_outputs_three_log_forms(tmp_path) -> None:
    logger = DeepResearchLogger()
    runner = DeepResearchRunner(
        search_client=FakeSearchClient(),
        actions=[make_action()],
        logger=logger,
        config=DeepResearchRunnerConfig(
            total_budget=4,
            epoch_budget=2,
            best_k=2,
            search_limit=1,
            min_wide_epochs=1,
            wide_batch_size=2,
            deep_batch_size=2,
        ),
    )

    report = asyncio.run(runner.run("logging target"))
    paths = logger.save(tmp_path)

    assert report.logger is logger
    assert logger.events
    assert logger.to_llm_context()["timeline"]
    assert "run_started" in logger.to_timeline_jsonl()
    assert "# DeepResearch Log" in logger.to_human_markdown()
    assert paths["llm_context"].exists()
    assert paths["timeline"].exists()
    assert paths["human"].exists()
