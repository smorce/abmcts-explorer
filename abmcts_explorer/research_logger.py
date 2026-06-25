from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchLogEvent:
    sequence: int
    timestamp: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)


class DeepResearchLogger:
    def __init__(self) -> None:
        self._events: list[ResearchLogEvent] = []
        self._llm_messages: list[dict[str, Any]] = []

    @property
    def events(self) -> list[ResearchLogEvent]:
        return list(self._events)

    @property
    def llm_messages(self) -> list[dict[str, Any]]:
        return list(self._llm_messages)

    def log_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self._events.append(
            ResearchLogEvent(
                sequence=len(self._events) + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=event_type,
                payload=payload or {},
            )
        )

    def log_llm_messages(
        self,
        *,
        action_name: str,
        profile: str,
        step_index: int,
        messages: list[dict[str, str]],
    ) -> None:
        self._llm_messages.append(
            {
                "sequence": len(self._llm_messages) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action_name": action_name,
                "profile": profile,
                "step_index": step_index,
                "messages": messages,
            }
        )
        self.log_event(
            "llm_messages_built",
            {
                "action_name": action_name,
                "profile": profile,
                "step_index": step_index,
                "message_count": len(messages),
            },
        )

    def log_llm_response(
        self,
        *,
        action_name: str,
        profile: str,
        step_index: int,
        raw_response: str,
    ) -> None:
        self.log_event(
            "llm_response_received",
            {
                "action_name": action_name,
                "profile": profile,
                "step_index": step_index,
                "response_chars": len(raw_response),
                "response_preview": raw_response[:500],
            },
        )

    def to_llm_context(self, max_events: int = 50) -> dict[str, Any]:
        return {
            "timeline": [asdict(event) for event in self._events[-max_events:]],
            "llm_call_count": len(self._llm_messages),
            "latest_llm_messages": self._llm_messages[-5:],
        }

    def to_timeline_jsonl(self) -> str:
        return "\n".join(
            json.dumps(asdict(event), ensure_ascii=False)
            for event in self._events
        )

    def to_human_markdown(self) -> str:
        lines = ["# DeepResearch Log", ""]
        for event in self._events:
            lines.append(f"## {event.sequence}. {event.event_type} ({event.timestamp})")
            if event.payload:
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(event.payload, ensure_ascii=False, indent=2))
                lines.append("```")
            lines.append("")
        return "\n".join(lines)

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)

        llm_context_path = directory / "llm_context_log.json"
        timeline_path = directory / "timeline_log.jsonl"
        human_path = directory / "human_log.md"

        llm_context_path.write_text(
            json.dumps(self.to_llm_context(max_events=1000), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        timeline_path.write_text(self.to_timeline_jsonl(), encoding="utf-8")
        human_path.write_text(self.to_human_markdown(), encoding="utf-8")

        return {
            "llm_context": llm_context_path,
            "timeline": timeline_path,
            "human": human_path,
        }
