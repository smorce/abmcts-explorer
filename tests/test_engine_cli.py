from __future__ import annotations

import json

from abmcts_explorer.engine_cli import main


def test_engine_cli_writes_demo_output(tmp_path) -> None:
    output = tmp_path / "result.json"

    exit_code = main(
        [
            "--task",
            "cli smoke",
            "--profile",
            "go_wide",
            "--budget",
            "4",
            "--batch-size",
            "2",
            "--best-k",
            "2",
            "--output",
            str(output),
        ]
    )

    data = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert data["task"] == "cli smoke"
    assert data["profile"] == "go_wide"
    assert len(data["best"]) == 2


def test_engine_cli_accepts_jsonl_input(tmp_path) -> None:
    input_jsonl = tmp_path / "states.jsonl"
    output = tmp_path / "result.json"
    input_jsonl.write_text(
        '{"text":"alpha","score":0.4}\n{"text":"beta","score":0.9}\n',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--task",
            "jsonl smoke",
            "--profile",
            "go_wide",
            "--budget",
            "4",
            "--batch-size",
            "2",
            "--best-k",
            "2",
            "--input-jsonl",
            str(input_jsonl),
            "--output",
            str(output),
        ]
    )

    data = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert data["best"][0]["score"] >= data["best"][1]["score"]


def test_engine_cli_auto_profile_records_decisions(tmp_path) -> None:
    output = tmp_path / "auto.json"

    exit_code = main(
        [
            "--task",
            "auto smoke",
            "--profile",
            "auto",
            "--budget",
            "8",
            "--epoch-budget",
            "4",
            "--batch-size",
            "2",
            "--best-k",
            "2",
            "--output",
            str(output),
        ]
    )

    data = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert data["profile"] == "auto"
    assert data["profile_decisions"]
    assert data["diagnostics"]
