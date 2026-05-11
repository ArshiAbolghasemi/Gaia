from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
from tigramite import plotting as tp

if TYPE_CHECKING:
    from pathlib import Path

    from config.model import LinkResult


def format_causal_graph(links: list[LinkResult], title: str) -> str:
    header = [title, "=" * len(title)]
    if not links:
        return "\n".join([*header, "No significant links found."])

    body = [
        (
            f"{link.source} --(lag={link.lag}, p={link.p_value:.4g}, "
            f"effect={link.effect_size:.4f})-> {link.target}"
        )
        for link in links
    ]
    return "\n".join([*header, *body])


def plot_tigramite_graph(
    pcmci_results: dict,
    var_names: list[str],
    output_path: Path,
) -> None:
    tp.plot_graph(
        val_matrix=pcmci_results["val_matrix"],
        graph=pcmci_results["graph"],
        var_names=var_names,
        link_colorbar_label="cross-correlation",
        node_colorbar_label="auto-correlation",
        save_name=str(output_path),
    )
    plt.close("all")


def plot_tigramite_time_series_graph(
    pcmci_results: dict,
    var_names: list[str],
    output_path: Path,
) -> None:
    tp.plot_time_series_graph(
        val_matrix=pcmci_results["val_matrix"],
        graph=pcmci_results["graph"],
        var_names=var_names,
        link_colorbar_label="cross-correlation",
        save_name=str(output_path),
    )
    plt.close("all")


def plot_network_graph(
    links: list[LinkResult],
    output_path: Path,
    title: str,
) -> None:
    graph = nx.DiGraph()
    for link in links:
        graph.add_edge(
            link.source,
            link.target,
            weight=abs(link.effect_size),
            label=f"lag {link.lag}\np={link.p_value:.3g}",
        )

    figure, axis = plt.subplots(figsize=(14, 10))
    if graph.number_of_nodes() == 0:
        axis.text(0.5, 0.5, "No significant links found", ha="center", va="center")
        axis.set_axis_off()
        figure.suptitle(title)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return

    positions = nx.spring_layout(graph, seed=7, k=1.4 / max(graph.number_of_nodes(), 1))
    node_colors = ["#1d4ed8" for _ in graph.nodes()]
    edge_widths = [
        max(1.2, min(5.0, graph[source][target]["weight"] * 12.0))
        for source, target in graph.edges()
    ]
    edge_labels = {
        (source, target): graph[source][target]["label"] for source, target in graph.edges()
    }

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=1250,
        alpha=0.95,
        ax=axis,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        width=edge_widths,  # type: ignore[arg-type]
        edge_color="#475569",
        arrows=True,
        arrowsize=18,
        alpha=0.8,
        ax=axis,
    )
    nx.draw_networkx_labels(graph, positions, font_size=8, font_color="white", ax=axis)
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=7,
        label_pos=0.5,
        ax=axis,
    )

    axis.set_title(title)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
