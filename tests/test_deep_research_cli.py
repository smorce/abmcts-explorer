from __future__ import annotations

import importlib.util
from pathlib import Path


def load_build_parser():
    module_path = Path(__file__).resolve().parents[1] / "deep_research_cli.py"
    spec = importlib.util.spec_from_file_location("deep_research_cli", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_parser


def test_deep_research_cli_parser_defaults() -> None:
    build_parser = load_build_parser()
    args = build_parser().parse_args(
        [
            "--topic",
            "test topic",
            "--model",
            "test-model",
        ]
    )

    assert args.topic == "test topic"
    assert args.model == "test-model"
    assert args.total_budget == 1000
    assert args.epoch_budget == 25
    assert args.wide_batch_size == 16
    assert args.deep_batch_size == 5
