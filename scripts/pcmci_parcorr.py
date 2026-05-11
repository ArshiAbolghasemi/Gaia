from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config.load import load_config
from src.log.config import configure_logging
from src.pcmci.runner import run_pcmci, save_outputs

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_CONFIG_DIR = ROOT / "config"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PCMCI + ParCorr causal inference configs."
    )
    parser.add_argument("--config", type=Path, help="Run a single JSON config file.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f"Directory with JSON configs searched recursively. "
        f"Defaults to {DEFAULT_CONFIG_DIR}.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity. Defaults to INFO.",
    )
    return parser


def execute_config(config_path: Path) -> None:
    logger.info("Executing config: %s", config_path)
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


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    if args.config is not None:
        execute_config(args.config)
        return

    for config_path in sorted(args.config_dir.rglob("*.json")):
        execute_config(config_path)


if __name__ == "__main__":
    main()
