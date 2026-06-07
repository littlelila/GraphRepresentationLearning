from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import networkx as nx
import pandas as pd

def filter_diseases_by_annotaion_count(
        hpoa: pd.DataFrame, 
        min_annotations: int = 5, 
        max_annotations: int = 50, 
        max_diseases: int | None = 1000, 
        random_state: int = 5,
) -> pd.DataFrame:
    """
    Filter disease-phenotpye annotations to diseases with a manageble number of HPO annotations.

    Parameters
    ----------
    hpoa:
        HPO annotation dataFrame.
    min_annotatjions:
        Minimum number of unique HPO terms per disease.
    max_annotations:
        Maxiumu number of unique HPO terms per disease.
    max_diseases:
        Optional maxiumum number of diseases to keep.
    random_state:
        Random seed used if max_diseases is applied.

    Returns
    --------
    pandas.DataFrame
        Filterede disease-phenotpye annotations.
    """
    required_cols = {"database_id", "hpo_id"}
    missing = required_cols - set(hpoa.columns)

    if missing:
        raise ValueError(f"Missing required columns in hpoa: {missing}")
    
    annotation_counts = hpoa.groupby("database_id")["hpo_id"].nunique()

    valid_diseases = annotation_counts[(annotation_counts >= min_annotations) & (annotation_counts <= max_annotations)].index

    filtered = hpoa[hpoa["database_id"].isin(valid_diseases)].copy()

    if max_diseases is not None:
        diseases = (
                filtered["database_id"]
                .drop_duplicates()
                .sample(
                    n=min(max_diseases, filtered["database_id"].nunique()),
                    random_state=random_state,
                )
        )
        
        filtered = filtered[filtered["database_id"].isin(diseases)].copy()
    
    return filtered.reset_index(drop=True)



def build_disease_phenotype_graph(hpoa: pd.DataFrame) -> nx.Graph:
    """
    Build a simple disease-phenotype graph.

    Nodes:
        disease nodes from database_id
        phenotype nodes from hpo_id

    Edges:
        disease --has_phenotype-- phenotype
    """
    required_columns = {"database_id", "disease_name", "hpo_id"}
    missing = required_columns - set(hpoa.columns)

    if missing:
        raise ValueError(f"Missing required columns in hpoa: {missing}")

    graph = nx.Graph()

    for _, row in hpoa.iterrows():
        disease_id = row["database_id"]
        disease_name = row["disease_name"]
        hpo_id = row["hpo_id"]

        graph.add_node(
            disease_id,
            node_type="disease",
            label=disease_name,
        )

        graph.add_node(
            hpo_id,
            node_type="phenotype",
            label=hpo_id,
        )

        graph.add_edge(
            disease_id,
            hpo_id,
            edge_type="has_phenotype",
        )

    return graph


def add_hpo_hierarchy_edges(
    graph: nx.Graph,
    hpo,
    include_only_relevant_terms: bool = True,
) -> nx.Graph:
    """
    Add HPO hierarchy edges to an existing disease-phenotype graph.

    obonet loads HPO edges as:
        child HPO term -> parent HPO term

    For Node2Vec, the graph is undirected, so this direction is not critical.
    """
    phenotype_nodes = {
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == "phenotype"
    }

    for child, parent, _ in hpo.edges(data=True):
        if include_only_relevant_terms:
            if child not in phenotype_nodes and parent not in phenotype_nodes:
                continue

        child_label = hpo.nodes[child].get("name", child) if child in hpo.nodes else child
        parent_label = hpo.nodes[parent].get("name", parent) if parent in hpo.nodes else parent

        graph.add_node(
            child,
            node_type="phenotype",
            label=child_label,
        )

        graph.add_node(
            parent,
            node_type="phenotype",
            label=parent_label,
        )

        graph.add_edge(
            child,
            parent,
            edge_type="is_a",
        )

    return graph


def get_graph_statistics(graph: nx.Graph) -> dict[str, int]:
    """
    Compute simple graph statistics.
    """
    node_type_counts = {}

    for _, data in graph.nodes(data=True):
        node_type = data.get("node_type", "unknown")
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

    edge_type_counts = {}

    for _, _, data in graph.edges(data=True):
        edge_type = data.get("edge_type", "unknown")
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

    stats = {
        "nodes_total": graph.number_of_nodes(),
        "edges_total": graph.number_of_edges(),
    }

    for node_type, count in node_type_counts.items():
        stats[f"nodes_{node_type}"] = count

    for edge_type, count in edge_type_counts.items():
        stats[f"edges_{edge_type}"] = count

    return stats


def export_graph_edges(graph: nx.Graph, output_path: str | Path) -> None:
    """
    Export graph edges to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for source, target, data in graph.edges(data=True):
        rows.append(
            {
                "source": source,
                "target": target,
                "edge_type": data.get("edge_type", "unknown"),
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def export_graph_nodes(graph: nx.Graph, output_path: str | Path) -> None:
    """
    Export graph nodes to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for node_id, data in graph.nodes(data=True):
        rows.append(
            {
                "node_id": node_id,
                "node_type": data.get("node_type", "unknown"),
                "label": data.get("label", node_id),
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)