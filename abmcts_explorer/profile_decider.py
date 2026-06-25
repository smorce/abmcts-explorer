from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .core import (
    AlgorithmKind,
    ExecutionMode,
    ExplorerConfig,
    SearchProfile,
    call_llamas_server,
)


class ProfileDecisionReason(str, Enum):
    INSUFFICIENT_DIVERSITY = "insufficient_diversity"
    HIGH_UNCERTAINTY = "high_uncertainty"
    EARLY_EXPLORATION = "early_exploration"
    PLATEAU = "plateau"
    STRONG_CANDIDATE_FOUND = "strong_candidate_found"
    LATE_STAGE_REFINEMENT = "late_stage_refinement"
    NEED_VERIFICATION = "need_verification"
    RESOURCE_PRESSURE = "resource_pressure"


@dataclass(frozen=True)
class SearchDiagnostics:
    total_nodes: int
    max_depth: int
    avg_depth: float
    best_score: float
    top_k_mean_score: float
    top_k_score_std: float
    score_improvement_recent: float
    diversity_score: float
    uncertainty_score: float
    judge_disagreement: float
    plateau_steps: int
    remaining_budget_ratio: float
    failed_generation_ratio: float = 0.0
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProfileDecision:
    profile: SearchProfile
    confidence: float
    reasons: list[ProfileDecisionReason]
    explanation: str
    suggested_batch_size: int | None = None
    suggested_budget: int | None = None


@dataclass(frozen=True)
class ProfileDeciderConfig:
    min_initial_nodes: int = 30
    min_diversity_score: float = 0.45
    high_uncertainty_threshold: float = 0.60
    high_judge_disagreement_threshold: float = 0.50
    strong_candidate_score: float = 0.82
    top_k_convergence_std: float = 0.08
    plateau_steps_threshold: int = 3
    low_recent_improvement: float = 0.01
    late_budget_ratio: float = 0.30
    min_epochs_before_switch: int = 2
    switch_confidence_threshold: float = 0.60


class RuleBasedProfileDecider:
    def __init__(self, config: ProfileDeciderConfig | None = None) -> None:
        self.config = config or ProfileDeciderConfig()
        self._last_profile: SearchProfile | None = None
        self._epochs_since_switch: int = 999

    def decide(
        self,
        diagnostics: SearchDiagnostics,
        current_profile: SearchProfile,
    ) -> ProfileDecision:
        cfg = self.config
        wide_score = 0.0
        deep_score = 0.0
        reasons: list[ProfileDecisionReason] = []

        if diagnostics.total_nodes < cfg.min_initial_nodes:
            wide_score += 0.35
            reasons.append(ProfileDecisionReason.EARLY_EXPLORATION)

        if diagnostics.diversity_score < cfg.min_diversity_score:
            wide_score += 0.30
            reasons.append(ProfileDecisionReason.INSUFFICIENT_DIVERSITY)

        if diagnostics.uncertainty_score >= cfg.high_uncertainty_threshold:
            wide_score += 0.25
            reasons.append(ProfileDecisionReason.HIGH_UNCERTAINTY)

        if diagnostics.judge_disagreement >= cfg.high_judge_disagreement_threshold:
            wide_score += 0.20
            reasons.append(ProfileDecisionReason.HIGH_UNCERTAINTY)

        if (
            diagnostics.best_score >= cfg.strong_candidate_score
            and diagnostics.top_k_score_std <= cfg.top_k_convergence_std
        ):
            deep_score += 0.40
            reasons.append(ProfileDecisionReason.STRONG_CANDIDATE_FOUND)

        if diagnostics.remaining_budget_ratio <= cfg.late_budget_ratio:
            deep_score += 0.30
            reasons.append(ProfileDecisionReason.LATE_STAGE_REFINEMENT)

        if (
            diagnostics.plateau_steps >= cfg.plateau_steps_threshold
            and diagnostics.score_improvement_recent <= cfg.low_recent_improvement
        ):
            if diagnostics.max_depth <= 2:
                wide_score += 0.25
            else:
                deep_score += 0.25
            reasons.append(ProfileDecisionReason.PLATEAU)

        if wide_score > deep_score:
            next_profile = SearchProfile.GO_WIDE
            confidence = min(1.0, 0.5 + wide_score - deep_score)
        else:
            next_profile = SearchProfile.GO_DEEP
            confidence = min(1.0, 0.5 + deep_score - wide_score)

        if (
            next_profile != current_profile
            and self._epochs_since_switch < cfg.min_epochs_before_switch
        ):
            next_profile = current_profile
            confidence = 0.50

        if confidence < cfg.switch_confidence_threshold:
            next_profile = current_profile

        if next_profile != current_profile:
            self._epochs_since_switch = 0
        else:
            self._epochs_since_switch += 1

        suggested_batch_size = self._suggest_batch_size(next_profile, diagnostics)

        return ProfileDecision(
            profile=next_profile,
            confidence=confidence,
            reasons=reasons,
            explanation=(
                f"wide_score={wide_score:.2f}, "
                f"deep_score={deep_score:.2f}, "
                f"selected={next_profile.value}"
            ),
            suggested_batch_size=suggested_batch_size,
        )

    def _suggest_batch_size(
        self,
        profile: SearchProfile,
        diagnostics: SearchDiagnostics,
    ) -> int:
        if profile == SearchProfile.GO_WIDE:
            if diagnostics.failed_generation_ratio > 0.20:
                return 8
            return 24

        if profile == SearchProfile.GO_DEEP:
            return 5

        return 8


@dataclass(frozen=True)
class LLMProfileAdvice:
    preferred_profile: SearchProfile
    confidence: float
    reason: str
    risks: list[str]


def ask_llm_profile_advice(
    diagnostics: SearchDiagnostics,
    current_profile: SearchProfile,
) -> LLMProfileAdvice:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a search-control advisor. "
                "Return only JSON. Choose GO_WIDE or GO_DEEP. "
                "Do not optimize for sounding confident. "
                "Prefer GO_WIDE when uncertainty or diversity problems remain. "
                "Prefer GO_DEEP when a strong candidate needs refinement."
            ),
        },
        {
            "role": "user",
            "content": f"""
Current profile:
{current_profile.value}

Diagnostics:
{diagnostics}

Return JSON:
{{
  "preferred_profile": "go_wide | go_deep",
  "confidence": 0.0,
  "reason": "...",
  "risks": ["..."]
}}
""",
        },
    ]

    call_llamas_server(
        model="llama-control",
        messages=messages,
        temperature=0.1,
        max_tokens=512,
    )
    raise NotImplementedError(
        "Parse and validate the LLM profile advice response before use."
    )


class HybridProfileDecider:
    def __init__(
        self,
        rule_decider: RuleBasedProfileDecider,
        use_llm_advice: bool = True,
    ) -> None:
        self.rule_decider = rule_decider
        self.use_llm_advice = use_llm_advice

    def decide(
        self,
        diagnostics: SearchDiagnostics,
        current_profile: SearchProfile,
    ) -> ProfileDecision:
        rule_decision = self.rule_decider.decide(
            diagnostics=diagnostics,
            current_profile=current_profile,
        )

        if not self.use_llm_advice:
            return rule_decision

        if rule_decision.confidence >= 0.75:
            return rule_decision

        advice = ask_llm_profile_advice(diagnostics, current_profile)

        if advice.preferred_profile == rule_decision.profile:
            return ProfileDecision(
                profile=rule_decision.profile,
                confidence=min(1.0, rule_decision.confidence + 0.10),
                reasons=rule_decision.reasons,
                explanation=rule_decision.explanation
                + f" | llm_agreed: {advice.reason}",
                suggested_batch_size=rule_decision.suggested_batch_size,
                suggested_budget=rule_decision.suggested_budget,
            )

        if rule_decision.confidence >= 0.60:
            return rule_decision

        if advice.confidence >= 0.75:
            return ProfileDecision(
                profile=advice.preferred_profile,
                confidence=advice.confidence,
                reasons=rule_decision.reasons,
                explanation=f"LLM advice adopted: {advice.reason}",
                suggested_batch_size=(
                    24 if advice.preferred_profile == SearchProfile.GO_WIDE else 5
                ),
            )

        return rule_decision


@dataclass(frozen=True)
class AutonomousRunConfig:
    total_budget: int = 300
    epoch_budget: int = 25
    diagnostics_top_k: int = 10
    allow_profile_switch: bool = True


def explorer_config_for_decision(
    decision: ProfileDecision,
    current_config: ExplorerConfig,
) -> ExplorerConfig:
    if decision.profile == SearchProfile.GO_WIDE:
        return ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSA,
            profile=SearchProfile.GO_WIDE,
            execution_mode=ExecutionMode.ASYNC_ASK_TELL,
            budget=current_config.budget,
            batch_size=decision.suggested_batch_size or 24,
            best_k=current_config.best_k,
            metadata=current_config.metadata,
        )

    if decision.profile == SearchProfile.GO_DEEP:
        batch_size = decision.suggested_batch_size or 5
        return ExplorerConfig(
            algorithm_kind=AlgorithmKind.ABMCTSM,
            profile=SearchProfile.GO_DEEP,
            execution_mode=ExecutionMode.ASK_TELL,
            budget=current_config.budget,
            batch_size=batch_size,
            best_k=current_config.best_k,
            algo_kwargs={"max_process_workers": batch_size},
            metadata=current_config.metadata,
        )

    raise ValueError(f"Unsupported profile: {decision.profile}")


def simple_profile_policy(diagnostics: SearchDiagnostics) -> SearchProfile:
    if diagnostics.total_nodes < 30:
        return SearchProfile.GO_WIDE

    if diagnostics.diversity_score < 0.45:
        return SearchProfile.GO_WIDE

    if diagnostics.uncertainty_score > 0.60:
        return SearchProfile.GO_WIDE

    if diagnostics.best_score > 0.85 and diagnostics.top_k_score_std < 0.08:
        return SearchProfile.GO_DEEP

    if diagnostics.remaining_budget_ratio < 0.25:
        return SearchProfile.GO_DEEP

    if diagnostics.plateau_steps >= 3:
        if diagnostics.max_depth <= 2:
            return SearchProfile.GO_WIDE
        return SearchProfile.GO_DEEP

    return SearchProfile.GO_WIDE
