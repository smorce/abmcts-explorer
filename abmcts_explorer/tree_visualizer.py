from __future__ import annotations

import json
import queue
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .core import ExplorerContext, ExplorerObserver, GenerationResult


@dataclass(frozen=True)
class TreeNodeLog:
    id: str
    parent_id: str | None
    label: str
    score: float
    action: str
    profile: str
    algorithm: str
    step_index: int
    depth: int
    summary: str
    state: Any
    metadata: dict[str, Any]
    created_at: float


class TreeVisualizerObserver(ExplorerObserver[Any]):
    def __init__(self, *, title: str = "AB-MCTS Explorer") -> None:
        self.title = title
        self._lock = threading.Lock()
        self._state_to_node: dict[int, str] = {}
        self._node_logs: list[TreeNodeLog] = []
        self._run_events: list[dict[str, Any]] = []
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._next_node_number = 1

    @property
    def node_logs(self) -> list[TreeNodeLog]:
        with self._lock:
            return list(self._node_logs)

    @property
    def run_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._run_events)

    def on_node_generated(
        self,
        *,
        parent_state: Any | None,
        result: GenerationResult[Any],
        context: ExplorerContext,
        action_name: str,
        algorithm: str,
    ) -> None:
        with self._lock:
            node_id = f"n{self._next_node_number}"
            self._next_node_number += 1
            parent_id = self._state_to_node.get(id(parent_state)) if parent_state is not None else "root"
            self._state_to_node[id(result.state)] = node_id
            depth = 1 if parent_id == "root" else self._depth_for(parent_id) + 1
            node = TreeNodeLog(
                id=node_id,
                parent_id=parent_id,
                label=self._label_for(result.state, node_id),
                score=result.score,
                action=action_name,
                profile=context.profile.value,
                algorithm=algorithm,
                step_index=context.step_index,
                depth=depth,
                summary=self._summary_for(result.state),
                state=self._serialize_value(result.state),
                metadata=self._serialize_mapping(result.metadata),
                created_at=time.time(),
            )
            self._node_logs.append(node)
            self._publish({"type": "node", "node": asdict(node)})

    def on_run_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._run_events.append(event)
            self._publish({"type": "run_event", "event": event})

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        stream: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            stream.put(
                {
                    "type": "snapshot",
                    "title": self.title,
                    "nodes": [asdict(node) for node in self._node_logs],
                    "events": list(self._run_events),
                }
            )
            self._subscribers.append(stream)
        return stream

    def unsubscribe(self, stream: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if stream in self._subscribers:
                self._subscribers.remove(stream)

    def save_node_log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(node), ensure_ascii=False) for node in self.node_logs]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _publish(self, payload: dict[str, Any]) -> None:
        for subscriber in list(self._subscribers):
            subscriber.put(payload)

    def _depth_for(self, node_id: str | None) -> int:
        if node_id is None or node_id == "root":
            return 0
        for node in reversed(self._node_logs):
            if node.id == node_id:
                return node.depth
        return 0

    def _label_for(self, state: Any, fallback: str) -> str:
        data = self._serialize_value(state)
        if isinstance(data, dict):
            metadata = data.get("metadata")
            if isinstance(metadata, dict) and metadata.get("call") is not None:
                return f"N{metadata['call']}"
            text = data.get("text") or data.get("draft") or data.get("topic")
            if text:
                return str(text).strip().splitlines()[0][:18] or fallback
        return fallback

    def _summary_for(self, state: Any) -> str:
        data = self._serialize_value(state)
        if isinstance(data, dict):
            text = data.get("text") or data.get("draft") or data.get("topic")
            if text:
                return self._clip(str(text), 640)
        return self._clip(str(data), 640)

    def _serialize_mapping(self, value: dict[str, Any]) -> dict[str, Any]:
        serialized = self._serialize_value(value)
        return serialized if isinstance(serialized, dict) else {"value": serialized}

    def _serialize_value(self, value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return self._serialize_value(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._serialize_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    @staticmethod
    def _clip(value: str, limit: int) -> str:
        value = value.strip()
        if len(value) <= limit:
            return value
        return value[: limit - 1] + "…"


class TreeVisualizerServer:
    def __init__(
        self,
        observer: TreeVisualizerObserver,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.observer = observer
        self.host = host
        self.port = port
        self._server = self._build_server()
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/"

    def start(self, *, open_browser: bool = True) -> str:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        if open_browser:
            webbrowser.open(self.url)
        return self.url

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _build_server(self) -> ThreadingHTTPServer:
        observer = self.observer

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/":
                    self._send_html()
                    return
                if path == "/events":
                    self._send_events()
                    return
                if path == "/snapshot":
                    self._send_json(
                        {
                            "title": observer.title,
                            "nodes": [asdict(node) for node in observer.node_logs],
                            "events": observer.run_events,
                        }
                    )
                    return
                self.send_error(404, "Not found")

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_html(self) -> None:
                body = build_visualizer_html(observer.title).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_events(self) -> None:
                stream = observer.subscribe()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while True:
                        payload = stream.get(timeout=20)
                        line = "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                        self.wfile.write(line.encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionError, queue.Empty):
                    pass
                finally:
                    observer.unsubscribe(stream)

        return ThreadingHTTPServer((self.host, self.port), Handler)


def build_visualizer_html(title: str) -> str:
    payload = json.dumps({"title": title}, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #171717;
      --muted: #6b7280;
      --line: #a3a3a3;
      --red: #c40000;
      --blue: #2f9bf4;
      --green: #55c96d;
      --orange: #ff8a3d;
      --panel: #ffffff;
      --surface: #f7f7f8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      background: var(--surface);
      font-family: "Segoe UI", "Noto Sans JP", system-ui, sans-serif;
    }}
    .shell {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      padding: 18px 28px;
      background: var(--panel);
      border-bottom: 1px solid #e5e7eb;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(24px, 3vw, 42px);
      line-height: 1.1;
      font-weight: 800;
      letter-spacing: 0;
    }}
    h1 span {{ color: var(--red); }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(92px, auto));
      gap: 10px;
    }}
    .metric {{
      min-width: 92px;
      padding: 8px 10px;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
    }}
    .metric b {{
      display: block;
      font-size: 18px;
    }}
    .metric span {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 11px;
    }}
    main {{
      position: relative;
      min-height: 0;
      overflow: hidden;
    }}
    svg {{
      display: block;
      width: 100%;
      height: calc(100vh - 92px);
      min-height: 620px;
      background: white;
      cursor: grab;
    }}
    svg:active {{ cursor: grabbing; }}
    .edge {{
      stroke: var(--line);
      stroke-width: 2;
      fill: none;
    }}
    .edge.best {{
      stroke: var(--red);
      stroke-width: 4;
    }}
    .node circle {{
      stroke: rgba(0,0,0,.24);
      stroke-width: 2;
      transition: r .18s ease, filter .18s ease;
    }}
    .node:hover circle {{
      r: 25;
      filter: drop-shadow(0 6px 10px rgba(0,0,0,.18));
    }}
    .node text {{
      pointer-events: none;
      text-anchor: middle;
      fill: white;
      font-weight: 700;
    }}
    .node .label {{ font-size: 10px; }}
    .node .score {{ font-size: 10px; transform: translateY(11px); }}
    .tooltip {{
      position: fixed;
      max-width: 420px;
      padding: 12px 14px;
      border: 2px solid var(--red);
      border-radius: 8px;
      background: rgba(255,255,255,.98);
      box-shadow: 0 14px 34px rgba(0,0,0,.16);
      font-size: 13px;
      line-height: 1.45;
      pointer-events: none;
      opacity: 0;
      transform: translate(8px, 8px);
      transition: opacity .12s ease;
      z-index: 4;
      white-space: pre-wrap;
    }}
    .tooltip.visible {{ opacity: 1; }}
    .status {{
      position: absolute;
      left: 28px;
      bottom: 24px;
      max-width: min(520px, calc(100vw - 56px));
      padding: 24px 30px;
      background: white;
      border: 4px solid var(--red);
      border-radius: 8px;
      font-size: clamp(20px, 3vw, 36px);
      line-height: 1.35;
      font-weight: 800;
    }}
    .status small {{
      display: block;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }}
    @media (max-width: 820px) {{
      header {{
        align-items: stretch;
        flex-direction: column;
        padding: 16px;
      }}
      .stats {{ grid-template-columns: repeat(2, 1fr); }}
      svg {{ height: calc(100vh - 188px); min-height: 560px; }}
      .status {{
        left: 16px;
        bottom: 16px;
        padding: 16px 18px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1><span>AB-MCTS</span> 木探索ライブ</h1>
      <div class="stats">
        <div class="metric"><b id="nodeCount">0</b><span>nodes</span></div>
        <div class="metric"><b id="bestScore">0.00</b><span>best score</span></div>
        <div class="metric"><b id="maxDepth">0</b><span>max depth</span></div>
        <div class="metric"><b id="profile">waiting</b><span>profile</span></div>
      </div>
    </header>
    <main>
      <svg id="tree" role="img" aria-label="AB-MCTS search tree"></svg>
      <div class="status" id="status">探索待機中<small>ノードに hover すると詳細を表示します</small></div>
      <div class="tooltip" id="tooltip"></div>
    </main>
  </div>
  <script>
    const boot = {payload};
    const nodes = new Map();
    const root = {{
      id: "root",
      parent_id: null,
      label: "Root",
      score: null,
      action: "start",
      profile: "root",
      algorithm: "root",
      step_index: 0,
      depth: 0,
      summary: boot.title
    }};
    nodes.set(root.id, root);

    const svg = document.getElementById("tree");
    const tooltip = document.getElementById("tooltip");
    const state = {{ scale: 1, offsetX: 0, offsetY: 0, dragging: false, startX: 0, startY: 0 }};

    function colorFor(node) {{
      if (node.id === "root") return "#2f9bf4";
      if ((node.score ?? 0) >= 0.72) return "#ff8a3d";
      if (node.profile === "go_deep") return "#55c96d";
      return "#2f9bf4";
    }}

    function childrenOf(parentId) {{
      return [...nodes.values()].filter(node => node.parent_id === parentId);
    }}

    function layout() {{
      const levels = new Map();
      for (const node of nodes.values()) {{
        const depth = node.depth || 0;
        if (!levels.has(depth)) levels.set(depth, []);
        levels.get(depth).push(node);
      }}
      const width = svg.clientWidth || 1200;
      const positions = new Map();
      const maxDepth = Math.max(...levels.keys());
      for (const [depth, row] of levels) {{
        row.sort((a, b) => a.id.localeCompare(b.id, undefined, {{ numeric: true }}));
        const y = 80 + depth * 92;
        row.forEach((node, index) => {{
          const spacing = width / (row.length + 1);
          positions.set(node.id, {{ x: spacing * (index + 1), y }});
        }});
      }}
      return {{ positions, maxDepth }};
    }}

    function render() {{
      const {{ positions, maxDepth }} = layout();
      const values = [...nodes.values()];
      const best = values.filter(n => n.id !== "root").sort((a, b) => (b.score ?? 0) - (a.score ?? 0))[0];
      const bestPath = new Set();
      let cursor = best;
      while (cursor) {{
        bestPath.add(cursor.id);
        cursor = nodes.get(cursor.parent_id);
      }}

      svg.innerHTML = "";
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("transform", `translate(${{state.offsetX}} ${{state.offsetY}}) scale(${{state.scale}})`);
      svg.appendChild(g);

      for (const node of values) {{
        if (!node.parent_id) continue;
        const from = positions.get(node.parent_id);
        const to = positions.get(node.id);
        if (!from || !to) continue;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", from.x);
        line.setAttribute("y1", from.y + 23);
        line.setAttribute("x2", to.x);
        line.setAttribute("y2", to.y - 23);
        line.setAttribute("class", bestPath.has(node.id) && bestPath.has(node.parent_id) ? "edge best" : "edge");
        g.appendChild(line);
      }}

      for (const node of values) {{
        const pos = positions.get(node.id);
        if (!pos) continue;
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("class", "node");
        group.setAttribute("transform", `translate(${{pos.x}} ${{pos.y}})`);
        group.addEventListener("mousemove", event => showTooltip(event, node));
        group.addEventListener("mouseleave", hideTooltip);

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("r", "23");
        circle.setAttribute("fill", colorFor(node));
        group.appendChild(circle);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("class", "label");
        label.setAttribute("y", "-2");
        label.textContent = String(node.label || node.id).slice(0, 10);
        group.appendChild(label);

        const score = document.createElementNS("http://www.w3.org/2000/svg", "text");
        score.setAttribute("class", "score");
        score.textContent = node.score == null ? "" : Number(node.score).toFixed(2);
        group.appendChild(score);
        g.appendChild(group);
      }}

      const nonRoot = values.length - 1;
      document.getElementById("nodeCount").textContent = String(nonRoot);
      document.getElementById("bestScore").textContent = best ? Number(best.score).toFixed(2) : "0.00";
      document.getElementById("maxDepth").textContent = String(maxDepth);
      document.getElementById("profile").textContent = best ? best.profile : "waiting";
      const status = document.getElementById("status");
      status.innerHTML = best
        ? `良い解へ探索中<small>best=${{Number(best.score).toFixed(3)}} / action=${{best.action}}</small>`
        : `探索待機中<small>ノードに hover すると詳細を表示します</small>`;
    }}

    function showTooltip(event, node) {{
      tooltip.textContent = [
        `${{node.label}}  score=${{node.score == null ? "-" : Number(node.score).toFixed(3)}}`,
        `action: ${{node.action}}`,
        `profile: ${{node.profile}}`,
        `algorithm: ${{node.algorithm}}`,
        `step: ${{node.step_index}} depth: ${{node.depth}}`,
        "",
        node.summary || ""
      ].join("\\n");
      tooltip.style.left = `${{event.clientX + 12}}px`;
      tooltip.style.top = `${{event.clientY + 12}}px`;
      tooltip.classList.add("visible");
    }}

    function hideTooltip() {{
      tooltip.classList.remove("visible");
    }}

    function applyMessage(message) {{
      if (message.type === "snapshot") {{
        for (const node of message.nodes || []) nodes.set(node.id, node);
      }}
      if (message.type === "node") {{
        nodes.set(message.node.id, message.node);
      }}
      render();
    }}

    svg.addEventListener("wheel", event => {{
      event.preventDefault();
      const delta = event.deltaY > 0 ? 0.92 : 1.08;
      state.scale = Math.max(0.35, Math.min(2.2, state.scale * delta));
      render();
    }}, {{ passive: false }});

    svg.addEventListener("pointerdown", event => {{
      state.dragging = true;
      state.startX = event.clientX - state.offsetX;
      state.startY = event.clientY - state.offsetY;
      svg.setPointerCapture(event.pointerId);
    }});
    svg.addEventListener("pointermove", event => {{
      if (!state.dragging) return;
      state.offsetX = event.clientX - state.startX;
      state.offsetY = event.clientY - state.startY;
      render();
    }});
    svg.addEventListener("pointerup", () => state.dragging = false);

    window.addEventListener("resize", render);
    const events = new EventSource("/events");
    events.onmessage = event => applyMessage(JSON.parse(event.data));
    render();
  </script>
</body>
</html>"""
