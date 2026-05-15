# ruff: noqa: E402

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.load import load_config
from src.log.config import configure_logging
from src.pcmci.runner import run_pcmci, save_outputs

if TYPE_CHECKING:
    from src.config.model import RunConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = ROOT / "config"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PCMCI causal inference configs.")
    parser.add_argument("--config", type=Path, help="Run a single JSON config file.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f"Directory with JSON configs searched recursively. "
        f"Defaults to {DEFAULT_CONFIG_DIR}.",
    )
    parser.add_argument(
        "--method",
        help="Only run configs whose dependence method matches this value.",
    )
    parser.add_argument(
        "--frequency",
        choices=["weekly", "monthly"],
        help="Only run configs whose dataset frequency matches this value.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity. Defaults to INFO.",
    )
    return parser


def execute_config(config_path: Path, config: RunConfig | None = None) -> None:
    logger.info("Executing config: %s", config_path)
    if config is None:
        config = load_config(config_path)
    payload = run_pcmci(config)
    paths = save_outputs(
        config,
        payload["pcmci"],
        payload["run_result"],
        payload["results"],
    )
    graph_text = paths["graph"].read_text()

    logger.info("[%s] saved summary to %s", config.name, paths["summary"])
    logger.info("[%s] saved links to %s", config.name, paths["links"])
    logger.info("[%s] saved graph to %s", config.name, paths["graph"])
    if "graph_plot" in paths:
        logger.info("[%s] saved graph plot to %s", config.name, paths["graph_plot"])
    if "ts_graph_plot" in paths:
        logger.info(
            "[%s] saved time series graph to %s", config.name, paths["ts_graph_plot"]
        )
    if "networkx_plot" in paths:
        logger.info("[%s] saved networkx graph to %s", config.name, paths["networkx_plot"])
    logger.info(graph_text)


def infer_frequency_from_path(path: Path) -> str | None:
    for part in path.parts:
        normalized = part.lower()
        if normalized in {"weekly", "monthly"}:
            return normalized
    return None


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    if args.config is not None:
        execute_config(args.config)
        return

    for config_path in sorted(args.config_dir.rglob("*.json")):
        config = None
        if args.method is not None or args.frequency is not None:
            config = load_config(config_path)
        if (
            args.method is not None
            and config.dependence.method.lower() != args.method.lower()
        ):
            continue
        if args.frequency is not None:
            frequency = infer_frequency_from_path(config.data.path)
            if frequency != args.frequency.lower():
                continue
        execute_config(config_path, config=config)


if __name__ == "__main__":
    main()
