from __future__ import annotations

import json
from urllib.request import urlopen

from abmcts_explorer.core import ExplorerContext, GenerationResult, SearchProfile
from abmcts_explorer.engine_cli import EngineCliState
from abmcts_explorer.tree_visualizer import TreeVisualizerObserver, TreeVisualizerServer


def test_tree_visualizer_observer_records_node_logs() -> None:
    observer = TreeVisualizerObserver(title="tree smoke")
    state = EngineCliState(text="candidate", depth=0)
    context = ExplorerContext(
        task="visualizer smoke",
        step_index=0,
        profile=SearchProfile.GO_WIDE,
    )

    observer.on_trial_started(
        parent_state=None,
        context=context,
        action_name="expand",
        algorithm="abmctsa",
    )

    assert observer.active_node_ids == ["root"]

    observer.on_node_generated(
        parent_state=None,
        result=GenerationResult(state=state, score=0.72, metadata={"kind": "test"}),
        context=context,
        action_name="expand",
        algorithm="abmctsa",
    )

    logs = observer.node_logs

    assert len(logs) == 1
    assert logs[0].parent_id == "root"
    assert logs[0].score == 0.72
    assert logs[0].summary == "candidate"
    assert observer.active_node_ids == []


def test_tree_visualizer_server_serves_snapshot() -> None:
    observer = TreeVisualizerObserver(title="server smoke")
    observer.on_run_event(
        {
            "event_type": "run_finished",
            "step_index": 1,
            "profile": "go_wide",
            "algorithm": "abmctsa",
            "payload": {"best_count": 1},
        }
    )
    server = TreeVisualizerServer(observer, port=0)
    url = server.start(open_browser=False)

    try:
        with urlopen(url + "snapshot", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.stop()

    assert payload["title"] == "server smoke"
    assert payload["nodes"] == []
    assert payload["events"][0]["event_type"] == "run_finished"
