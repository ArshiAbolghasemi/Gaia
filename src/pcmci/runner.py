from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI

from src.config.model import LinkResult, RunConfig, RunResult
from src.dataset.data import load_dataset, prepare_numeric_frame
from src.dataset.feature import select_feature
from src.pcmci.dependence import build_dependence_test
from src.pcmci.graph import (
    build_top_link_pcmci_results,
    format_tigramite_graph,
    plot_network_graph,
    plot_tigramite_graph,
    plot_tigramite_time_series_graph,
)

logger = logging.getLogger(__name__)


def run_pcmci(config: RunConfig) -> dict[str, Any]:
    logger.info("Starting PCMCI run: %s", config.name)

    logger.debug("Loading dataset from: %s", config.data.path)
    dataset = load_dataset(config.data.path)
    logger.debug("Dataset loaded with %d rows", len(dataset))

    logger.debug("Selecting features with groups: %s", config.features.include_groups)
    selected = select_feature(dataset, config.data, config.features)

    logger.debug("Preparing numeric frame")
    numeric = prepare_numeric_frame(selected, config.data)
    logger.info(
        "Numeric frame ready: %d rows x %d columns — %s",
        len(numeric),
        len(numeric.columns),
        numeric.columns.tolist(),
    )

    logger.debug("Building tigramite DataFrame")
    dataframe = pp.DataFrame(numeric.to_numpy(), var_names=numeric.columns.tolist())

    logger.debug("Building dependence test: method=%s", config.dependence.method)
    dependence_test = build_dependence_test(config.dependence)

    logger.info(
        "Running PCMCI with %s on %d variables "
        "(tau_min=%d, tau_max=%d, pc_alpha=%s, alpha_level=%s)",
        config.dependence.method.upper(),
        len(numeric.columns),
        config.pcmci.tau_min,
        config.pcmci.tau_max,
        config.pcmci.pc_alpha,
        config.pcmci.alpha_level,
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=dependence_test)

    results = pcmci.run_pcmci(
        tau_min=config.pcmci.tau_min,
        tau_max=config.pcmci.tau_max,
        pc_alpha=config.pcmci.pc_alpha,
        alpha_level=config.pcmci.alpha_level,
        fdr_method=config.pcmci.fdr_method,
        max_conds_dim=config.pcmci.max_conds_dim,
        max_combinations=config.pcmci.max_combinations,
        max_conds_py=config.pcmci.max_conds_py,
        max_conds_px=config.pcmci.max_conds_px,
    )
    logger.debug("PCMCI run complete; extracting significant links")

    all_links = extract_links(
        variable_names=numeric.columns.tolist(),
        p_matrix=results["p_matrix"],
        val_matrix=results["val_matrix"],
        alpha_level=config.pcmci.alpha_level,
    )
    logger.info(
        "Found %d significant links (alpha_level=%s)",
        len(all_links),
        config.pcmci.alpha_level,
    )

    run_result = RunResult(
        config_name=config.name,
        selected_columns=numeric.columns.tolist(),
        row_count=len(numeric),
        links=all_links,
    )

    return {
        "numeric": numeric,
        "pcmci": pcmci,
        "results": results,
        "run_result": run_result,
    }


def extract_links(
    *,
    variable_names: list[str],
    p_matrix: np.ndarray,
    val_matrix: np.ndarray,
    alpha_level: float,
) -> list[LinkResult]:
    logger.debug(
        "Extracting links from p_matrix shape=%s, alpha_level=%s",
        p_matrix.shape,
        alpha_level,
    )
    links: list[LinkResult] = []
    skipped_nan = 0

    for source_index, source_name in enumerate(variable_names):
        for target_index, target_name in enumerate(variable_names):
            for tau_index in range(1, p_matrix.shape[2]):
                p_value = float(p_matrix[source_index, target_index, tau_index])
                effect_size = float(val_matrix[source_index, target_index, tau_index])
                if np.isnan(p_value) or np.isnan(effect_size):
                    skipped_nan += 1
                    continue
                if p_value > alpha_level:
                    continue
                links.append(
                    LinkResult(
                        source=source_name,
                        target=target_name,
                        lag=-tau_index,
                        p_value=p_value,
                        effect_size=effect_size,
                    )
                )

    if skipped_nan:
        logger.debug("Skipped %d entries with NaN p_value or effect_size", skipped_nan)

    sorted_links = sorted(
        links,
        key=lambda link: (
            link.p_value,
            -abs(link.effect_size),
            link.target,
            link.source,
            link.lag,
        ),
    )
    logger.debug("Top link: %s", sorted_links[0] if sorted_links else "none")
    return sorted_links


def save_outputs(
    config: RunConfig,
    pcmci: PCMCI,
    run_result: RunResult,
    pcmci_results: dict,
) -> dict[str, Path]:
    output_dir = _build_output_dir(config)
    logger.info("Saving outputs to: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = _build_output_stem(config)
    top_links = run_result.links[: config.output.max_links]
    logger.debug(
        "Using top %d of %d links (max_links=%d)",
        len(top_links),
        len(run_result.links),
        config.output.max_links,
    )

    summary_path = output_dir / f"{prefix}_summary.json"
    links_path = output_dir / f"{prefix}.csv"
    graph_path = output_dir / f"{prefix}_graph.txt"

    filtered_pcmci_results = build_top_link_pcmci_results(
        pcmci_results,
        top_links,
        run_result.selected_columns,
    )
    graph_text = format_tigramite_graph(
        pcmci,
        filtered_pcmci_results,
        title=f"Causal graph for {config.name}",
        alpha_level=config.pcmci.alpha_level,
    )

    summary_payload = {
        "config_name": run_result.config_name,
        "row_count": run_result.row_count,
        "selected_columns": run_result.selected_columns,
        "dependence": asdict(config.dependence),
        "pcmci": asdict(config.pcmci),
        "feature_selection": asdict(config.features),
        "output_directory": str(output_dir),
        "output_stem": prefix,
        "significant_link_count": len(run_result.links),
        "displayed_link_count": len(top_links),
        "top_links": [asdict(link) for link in top_links],
    }

    logger.debug("Writing summary JSON: %s", summary_path)
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    logger.debug("Writing links CSV: %s", links_path)
    _write_links_csv(links_path, top_links)

    logger.debug("Writing causal graph text: %s", graph_path)
    graph_path.write_text(graph_text)

    output_paths = {
        "summary": summary_path,
        "links": links_path,
        "graph": graph_path,
    }

    if config.output.save_tigramite_plots:
        graph_plot_path = output_dir / f"{prefix}_graph.png"
        ts_graph_plot_path = output_dir / f"{prefix}_ts_graph.png"
        logger.info("Saving tigramite plots: %s, %s", graph_plot_path, ts_graph_plot_path)
        plot_tigramite_graph(
            filtered_pcmci_results,
            run_result.selected_columns,
            graph_plot_path,
        )
        plot_tigramite_time_series_graph(
            filtered_pcmci_results,
            run_result.selected_columns,
            ts_graph_plot_path,
        )
        output_paths["graph_plot"] = graph_plot_path
        output_paths["ts_graph_plot"] = ts_graph_plot_path

    if config.output.save_networkx_plot:
        network_plot_path = output_dir / f"{prefix}_networkx.png"
        logger.info("Saving networkx plot: %s", network_plot_path)
        plot_network_graph(
            top_links,
            network_plot_path,
            title=f"Network view for {config.name}",
        )
        output_paths["networkx_plot"] = network_plot_path

    logger.info(
        "All outputs saved — %d file(s): %s",
        len(output_paths),
        ", ".join(output_paths.keys()),
    )
    return output_paths


def _write_links_csv(path: Path, links: list[LinkResult]) -> None:
    logger.debug("Writing %d link(s) to CSV: %s", len(links), path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "target", "lag", "p_value", "effect_size"],
        )
        writer.writeheader()
        for link in links:
            writer.writerow(asdict(link))


def _build_output_dir(config: RunConfig) -> Path:
    crop_name = config.data.path.stem.lower()
    output_dir = (
        config.output.directory / crop_name / "pcmci" / config.dependence.method.lower()
    )
    logger.debug("Output directory: %s", output_dir)
    return output_dir


def _build_output_stem(config: RunConfig) -> str:
    if config.output.run_name:
        logger.debug("Using explicit run_name as output stem: %s", config.output.run_name)
        return config.output.run_name
    stem = "_".join(config.features.include_groups)
    logger.debug("Derived output stem from include_groups: %s", stem)
    return stem
