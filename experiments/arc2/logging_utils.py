from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


class ResearchLogger:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir
        self.logs_dir = save_dir / "logs"
        self.llm_io_dir = save_dir / "llm_io"
        self.tree_dir = save_dir / "tree"
        for directory in (self.logs_dir, self.llm_io_dir, self.tree_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.events_path = self.logs_dir / "events.jsonl"
        self.progress_path = self.logs_dir / "progress.md"
        self.llm_calls_path = self.llm_io_dir / "llm_calls.jsonl"
        self.nodes_path = self.tree_dir / "nodes.jsonl"
        self.edges_path = self.tree_dir / "edges.jsonl"
        self._lock = threading.Lock()
        self._call_count = 0

        self.human_logger = logging.getLogger("deepresearch")
        self.human_logger.setLevel(logging.INFO)
        self.human_logger.propagate = True
        log_path = self.logs_dir / "research.log"
        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path
            for handler in self.human_logger.handlers
        ):
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            self.human_logger.addHandler(handler)

        if not self.progress_path.exists():
            self.progress_path.write_text("# DeepResearch Progress\n\n", encoding="utf-8")

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, default=_json_default)
        with self._lock:
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
                file.flush()

    def log_event(self, event: str, **payload: Any) -> None:
        self._append_jsonl(
            self.events_path,
            {
                "ts": self.now(),
                "event": event,
                **payload,
            },
        )

    def log_human(self, message: str, **payload: Any) -> None:
        if payload:
            suffix = " ".join(f"{key}={value}" for key, value in payload.items())
            self.human_logger.info("%s %s", message, suffix)
        else:
            self.human_logger.info(message)

    def log_progress(self, heading: str, body: str) -> None:
        with self._lock:
            with self.progress_path.open("a", encoding="utf-8") as file:
                file.write(f"## {heading}\n\n{body}\n\n")
                file.flush()

    def log_llm_call(
        self,
        *,
        node_id: str | None,
        action: str,
        perspective: str,
        role: str,
        system: str,
        user: str,
        response: str,
    ) -> None:
        with self._lock:
            self._call_count += 1
            call_id = f"call_{self._call_count:04d}"

        payload = {
            "ts": self.now(),
            "call_id": call_id,
            "node_id": node_id,
            "action": action,
            "perspective": perspective,
            "role": role,
            "system": system,
            "user": user,
            "response": response,
        }
        self._append_jsonl(self.llm_calls_path, payload)
        call_path = self.llm_io_dir / f"{call_id}.json"
        call_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def log_node(self, state: Any) -> None:
        payload = asdict(state) if is_dataclass(state) else dict(state)
        self._append_jsonl(self.nodes_path, payload)

    def log_edge(
        self,
        *,
        parent_id: str | None,
        child_id: str,
        action: str,
        score_delta: float | None,
    ) -> None:
        self._append_jsonl(
            self.edges_path,
            {
                "ts": self.now(),
                "parent_id": parent_id,
                "child_id": child_id,
                "action": action,
                "score_delta": score_delta,
            },
        )

    def write_final_artifact(self, filename: str, content: str) -> Path:
        path = self.save_dir / filename
        path.write_text(content, encoding="utf-8")
        return path
