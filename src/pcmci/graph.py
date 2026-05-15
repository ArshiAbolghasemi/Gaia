from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from tigramite import plotting as tp

try:
    from pyvis.network import Network
except ModuleNotFoundError:
    Network = None

GRAPH_DIMENSIONS = 3

if TYPE_CHECKING:
    from pathlib import Path

    from tigramite.pcmci import PCMCI

    from src.config.model import LinkResult


def format_tigramite_graph(
    pcmci: PCMCI,
    pcmci_results: dict,
    title: str,
    alpha_level: float,
) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        buffer.write(f"{title}\n")
        buffer.write(f"{'=' * len(title)}\n")
        pcmci.print_significant_links(
            p_matrix=pcmci_results["p_matrix"],
            val_matrix=pcmci_results["val_matrix"],
            conf_matrix=pcmci_results.get("conf_matrix"),
            graph=pcmci_results.get("graph"),
            ambiguous_triples=pcmci_results.get("ambiguous_triples"),
            alpha_level=alpha_level,
        )
    return buffer.getvalue().strip()


def build_top_link_pcmci_results(
    pcmci_results: dict,
    links: list[LinkResult],
    var_names: list[str],
) -> dict:
    variable_index = {name: index for index, name in enumerate(var_names)}
    graph = pcmci_results.get("graph")
    conf_matrix = pcmci_results.get("conf_matrix")
    ambiguous_triples = pcmci_results.get("ambiguous_triples")

    filtered_results = {
        "p_matrix": np.ones_like(pcmci_results["p_matrix"]),
        "val_matrix": np.zeros_like(pcmci_results["val_matrix"]),
    }

    if graph is not None:
        filtered_results["graph"] = np.full_like(graph, "")
    if conf_matrix is not None:
        filtered_results["conf_matrix"] = np.zeros_like(conf_matrix)
    if ambiguous_triples is not None:
        filtered_results["ambiguous_triples"] = []

    for link in links:
        source_index = variable_index[link.source]
        target_index = variable_index[link.target]
        lag_index = abs(link.lag)

        filtered_results["p_matrix"][source_index, target_index, lag_index] = pcmci_results[
            "p_matrix"
        ][source_index, target_index, lag_index]
        filtered_results["val_matrix"][source_index, target_index, lag_index] = (
            pcmci_results["val_matrix"][source_index, target_index, lag_index]
        )

        if graph is not None:
            filtered_results["graph"][source_index, target_index, lag_index] = graph[
                source_index, target_index, lag_index
            ]
        if conf_matrix is not None:
            filtered_results["conf_matrix"][source_index, target_index, lag_index] = (
                conf_matrix[source_index, target_index, lag_index]
            )

    return filtered_results


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
    pcmci_results: dict,
    var_names: list[str],
    output_path: Path,
    title: str,
    alpha_level: float,
) -> None:
    if Network is None:
        msg = "Interactive graph output requires the optional 'pyvis' dependency."
        raise ModuleNotFoundError(msg)

    graph = _get_graph_array(pcmci_results["graph"])
    edges = _extract_edges_with_types(
        var_names=var_names,
        graph_3d=graph,
        matrices={
            "p_matrix": pcmci_results["p_matrix"],
            "val_matrix": pcmci_results["val_matrix"],
        },
        options={
            "tau_min": 1,
            "tau_max": graph.shape[2] - 1,
            "alpha_level": alpha_level,
            "filter_by_p": False,
            "include_lagzero_links": False,
            "dedup_bidirected": True,
        },
    )

    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.barnes_hut(
        gravity=-9000,
        central_gravity=0.25,
        spring_length=220,
        spring_strength=0.02,
        damping=0.9,
    )
    net.show_buttons(filter_=["physics", "layout", "interaction"])

    for idx, name in enumerate(var_names):
        net.add_node(
            idx,
            label=name,
            title=name,
            group=_guess_group(name),
            shape="dot",
            size=18,
        )

    for edge in edges:
        width = max(1.0, min(8.0, abs(edge["stat"]) * 6.0))
        style = _style_for(edge["edge_type"])
        lag = edge["lag"]
        net.add_edge(
            edge["source_idx"],
            edge["target_idx"],
            label=f"tau={lag}",
            title=(
                f"<b>{edge['source']} {edge['edge_type']} {edge['target']}</b><br>"
                f"lag tau = {lag}<br>"
                f"stat = {edge['stat']:.4f}<br>"
                f"p = {edge['pval']:.4g}"
            ),
            width=width,
            smooth={"type": "curvedCW", "roundness": 0.15 + 0.05 * max(lag - 1, 0)},
            physics=True,
            etype=edge["edge_type"],
            stat=float(edge["stat"]),
            pval=float(edge["pval"]),
            lag=int(edge["lag"]),
            source_label=edge["source"],
            target_label=edge["target"],
            **style,
        )

    _add_legend(net)
    html = net.generate_html()
    html = html.replace(
        "</body>",
        _build_info_panel_html(title=title) + "\n</body>",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _get_graph_array(results_graph: np.ndarray | list[np.ndarray]) -> np.ndarray:
    if isinstance(results_graph, np.ndarray):
        if results_graph.ndim == GRAPH_DIMENSIONS:
            return results_graph
        msg = f"Unexpected 'graph' shape: {results_graph.shape}"
        raise ValueError(msg)
    if isinstance(results_graph, (list, tuple)):
        return np.stack(results_graph, axis=0)
    msg = "pcmci_results['graph'] has unexpected type."
    raise TypeError(msg)


def _extract_edges_with_types(
    *,
    var_names: list[str],
    graph_3d: np.ndarray,
    matrices: dict[str, np.ndarray],
    options: dict[str, int | float | bool],
) -> list[dict[str, str | int | float]]:
    p_matrix = matrices["p_matrix"]
    val_matrix = matrices["val_matrix"]
    tau_min = int(options["tau_min"])
    tau_max = int(options["tau_max"])
    alpha_level = float(options["alpha_level"])
    filter_by_p = bool(options["filter_by_p"])
    include_lagzero_links = bool(options["include_lagzero_links"])
    dedup_bidirected = bool(options["dedup_bidirected"])
    node_count, second_dim, lag_count = graph_3d.shape
    if node_count != second_dim:
        msg = f"Unexpected graph shape: {graph_3d.shape}"
        raise ValueError(msg)

    start_lag = 0 if include_lagzero_links else max(1, tau_min)
    end_lag = min(lag_count - 1, tau_max)
    edges: list[dict[str, str | int | float]] = []

    for source_index in range(node_count):
        for target_index in range(node_count):
            for lag in range(start_lag, end_lag + 1):
                edge_type = graph_3d[source_index, target_index, lag]
                if isinstance(edge_type, bytes):
                    edge_type = edge_type.decode("utf-8", errors="ignore")
                edge_type = str(edge_type or "").strip()
                if not edge_type:
                    continue
                if dedup_bidirected and edge_type == "<->" and source_index > target_index:
                    continue

                p_value = float(p_matrix[source_index, target_index, lag])
                stat = float(val_matrix[source_index, target_index, lag])
                if filter_by_p and not (np.isfinite(p_value) and p_value <= alpha_level):
                    continue

                edges.append(
                    {
                        "source_idx": source_index,
                        "target_idx": target_index,
                        "source": var_names[source_index],
                        "target": var_names[target_index],
                        "lag": lag,
                        "stat": stat,
                        "pval": p_value,
                        "edge_type": edge_type,
                    }
                )

    return edges


def _style_for(edge_type: str) -> dict:
    if edge_type == "-->":
        return {
            "arrows": {"to": {"enabled": True, "type": "arrow"}},
            "dashes": False,
            "color": {"color": "#333333"},
        }
    if edge_type == "o->":
        return {
            "arrows": {
                "to": {"enabled": True, "type": "arrow"},
                "from": {"enabled": True, "type": "circle"},
            },
            "dashes": True,
            "color": {"color": "#d35400"},
        }
    if edge_type == "<->":
        return {
            "arrows": {
                "to": {"enabled": True, "type": "arrow"},
                "from": {"enabled": True, "type": "arrow"},
            },
            "dashes": False,
            "color": {"color": "#1f77b4"},
        }
    return {
        "arrows": {"to": {"enabled": True, "type": "arrow"}},
        "dashes": False,
        "color": {"color": "#7f7f7f"},
    }


def _guess_group(name: str) -> str:
    for token in ["corn", "oats", "soybeans", "wheat", "ssta", "soi", "meiv2"]:
        if token in name.lower():
            return token.upper()
    return "OTHER"


def _add_legend(net: object) -> None:
    base_x, base_y = -800, -600
    net.add_node(
        "LEG",
        label="Legend",
        shape="box",
        x=base_x,
        y=base_y,
        fixed={"x": True, "y": True},
        physics=False,
    )
    rows = [
        ("-->", "#333333", False, {"to": {"enabled": True, "type": "arrow"}}),
        (
            "o->",
            "#d35400",
            True,
            {
                "to": {"enabled": True, "type": "arrow"},
                "from": {"enabled": True, "type": "circle"},
            },
        ),
        (
            "<->",
            "#1f77b4",
            False,
            {
                "to": {"enabled": True, "type": "arrow"},
                "from": {"enabled": True, "type": "arrow"},
            },
        ),
    ]
    for index, (label, color, dashes, arrows) in enumerate(rows, start=1):
        y_pos = base_y + index * 60
        left_node = f"L{index}"
        right_node = f"R{index}"
        net.add_node(
            left_node,
            label=label,
            shape="ellipse",
            x=base_x - 30,
            y=y_pos,
            fixed={"x": True, "y": True},
            physics=False,
        )
        net.add_node(
            right_node,
            label="",
            shape="dot",
            size=1,
            x=base_x + 120,
            y=y_pos,
            fixed={"x": True, "y": True},
            physics=False,
        )
        net.add_edge(
            left_node,
            right_node,
            label=label,
            physics=False,
            dashes=dashes,
            arrows=arrows,
            color={"color": color},
        )


def _build_info_panel_html(*, title: str) -> str:
    panel_config = {"title": title}
    panel_json = json.dumps(panel_config)
    return f"""
<style>
  #infoPanel {{
    position: fixed;
    top: 20px;
    right: 20px;
    width: min(34%, 520px);
    max-height: 90vh;
    overflow: auto;
    padding: 14px 16px;
    background: #ffffff;
    color: #333333;
    border: 1px solid #dddddd;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    display: none;
    z-index: 1000;
  }}
  #infoPanel h3 {{ margin: 0 0 10px 0; font-size: 18px; }}
  #infoPanel .muted {{ color: #666666; font-size: 12px; }}
  #infoPanel ul {{ margin: 8px 0 0 18px; padding: 0; }}
  #infoPanel li {{ margin: 4px 0; }}
</style>
<div id="infoPanel">
  <h3>Node causes</h3>
  <div class="muted">Click a node to list all causes (parents across lags).</div>
  <hr/>
  <div><b>Selected:</b> <span id="nodeName">-</span></div>
  <div id="parentsList" style="margin-top:8px;"></div>
</div>
<script type="text/javascript">
(function() {{
  const panelConfig = {panel_json};
  function fmtP(p) {{
    if (typeof p !== "number" || !isFinite(p)) return String(p);
    if (p === 0) return "0";
    return p < 1e-3 ? p.toExponential(2) : p.toFixed(4);
  }}

  function renderParents(nodeId) {{
    const node = nodes.get(nodeId);
    if (!node) return;

    const causes = [];
    const edgeData = edges.get();
    for (let index = 0; index < edgeData.length; index += 1) {{
      const edge = edgeData[index];
      const edgeType = (edge.etype || "").trim();
      if (!edgeType) continue;
      const isDirectedIn =
        (edgeType === "-->" || edgeType === "o->") && edge.to === nodeId;
      const isBidirected =
        edgeType === "<->" && (edge.to === nodeId || edge.from === nodeId);
      if (!isDirectedIn && !isBidirected) continue;

      const parentId = edge.to === nodeId ? edge.from : edge.to;
      const parentNode = nodes.get(parentId);
      causes.push({{
        parentName: parentNode ? parentNode.label : String(parentId),
        edgeType,
        lag: edge.lag,
        stat: edge.stat,
        pval: edge.pval,
      }});
    }}

    causes.sort((left, right) => {{
      if (left.lag !== right.lag) return left.lag - right.lag;
      return Math.abs(right.stat) - Math.abs(left.stat);
    }});

    const listHtml = causes.length
      ? "<ul>" + causes.map((cause) =>
          "<li><b>" + cause.parentName + "</b> " + cause.edgeType +
          " <b>" + node.label + "</b> - tau=" + cause.lag +
          ", stat=" +
          (typeof cause.stat === "number" ? cause.stat.toFixed(4) : cause.stat) +
          ", p=" + fmtP(cause.pval) + "</li>"
        ).join("") + "</ul>"
      : "<em>No causes found for this node at current settings.</em>";

    document.getElementById("nodeName").textContent = node.label;
    document.getElementById("parentsList").innerHTML = listHtml;
    document.getElementById("infoPanel").style.display = "block";
    document.title = panelConfig.title;
  }}

  network.on("selectNode", (params) => {{
    if (params.nodes && params.nodes.length > 0) {{
      renderParents(params.nodes[0]);
    }}
  }});
  network.on("deselectNode", () => {{
    document.getElementById("infoPanel").style.display = "none";
  }});
}})();
</script>
"""
