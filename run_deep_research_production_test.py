from __future__ import annotations

import asyncio
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


TOPIC = (
    "「エキスパートモデルを個別に学習してオンポリシー蒸留で統合する」という"
    "アプローチが挙げられます。これは従来の一元的な巨大モデルとは異なる哲学に"
    "基づいており、複数の小規模な専門モデルを独立にトレーニングし、その知識を"
    "統合することで大規模モデルの性能を上回ることを目指しています。この"
    "トレーニングの詳細を解説したレポート"
)

MODEL = "Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL"


def build_actions() -> list[ActionSpec[DeepResearchState]]:
    return [
        ActionSpec[DeepResearchState](
            name="wide_web_research",
            model=MODEL,
            prompt_builder=build_deep_research_prompt,
            parser=parse_deep_research_state,
            scorer=score_deep_research_state,
            temperature=0.7,
            max_tokens=1400,
        ),
        ActionSpec[DeepResearchState](
            name="deep_evidence_synthesis",
            model=MODEL,
            prompt_builder=build_deep_research_prompt,
            parser=parse_deep_research_state,
            scorer=score_deep_research_state,
            temperature=0.2,
            max_tokens=1800,
        ),
    ]


async def main() -> None:
    runner = DeepResearchRunner(
        actions=build_actions(),
        config=DeepResearchRunnerConfig(
            total_budget=6,
            epoch_budget=2,
            best_k=2,
            search_limit=5,
            min_wide_epochs=1,
            wide_batch_size=2,
            deep_batch_size=2,
            report_model=MODEL,
            report_max_tokens=4096,
        ),
    )
    result = await runner.run(TOPIC)

    output = Path(__file__).with_name("deep_research_production_report.md")
    log_paths = result.logger.save(Path(__file__).with_name("deep_research_logs")) if result.logger else {}
    profile_lines = "\n".join(
        f"- {index + 1}: {decision.profile.value} "
        f"confidence={decision.confidence:.2f} {decision.explanation}"
        for index, decision in enumerate(result.profile_history)
    )
    source_lines = "\n".join(
        f"- [{item.title}]({item.url})"
        for item in result.web_results[:10]
    )
    output.write_text(
        "# DeepResearch Production Test\n\n"
        f"## Theme\n\n{TOPIC}\n\n"
        "## Profile History\n\n"
        f"{profile_lines}\n\n"
        "## Web Sources\n\n"
        f"{source_lines}\n\n"
        "## Report\n\n"
        f"{result.report}\n",
        encoding="utf-8",
    )

    print(f"REPORT_PATH={output}")
    for name, path in log_paths.items():
        print(f"LOG_{name.upper()}={path}")
    print("PROFILE_HISTORY=" + ",".join(d.profile.value for d in result.profile_history))
    print(f"WEB_RESULT_COUNT={len(result.web_results)}")
    print(f"BEST_COUNT={len(result.best_candidates)}")
    print(result.report[:2000])


if __name__ == "__main__":
    asyncio.run(main())
