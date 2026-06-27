from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import umap
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity

def load_graph_from_csv(
    nodes_path: str | Path,
    edges_path: str | Path,
) -> nx.Graph:
    """
    Load a NetworkX graph from node and edge CSV files.

    Parameters
    ----------
    nodes_path:
        Path to nodes CSV file. Expected columns:
        node_id, node_type, label

    edges_path:
        Path to edges CSV file. Expected columns:
        source, target, edge_type

    Returns
    -------
    networkx.Graph
        Undireted graph with node and edge attributes.
    
    """
    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)

    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")
    
    nodes = pd.read_csv(nodes_path, dtype=str)
    edges = pd.read_csv(edges_path, dtype=str)

    required_node_columns = {"node_id", "node_type", "label"}
    required_edge_columns = {"source", "target", "edge_type"}

    missing_node_columns = required_node_columns - set(nodes.columns)
    missing_edge_columns = required_edge_columns - set(edges.columns)

    if missing_node_columns:
        raise ValueError(f"Missing node columns: {missing_node_columns}")
    
    if missing_edge_columns:
        raise ValueError(f"Missing edge columns: {missing_edge_columns}")
    
    graph = nx.Graph()

    for _, row in nodes.iterrows():
        graph.add_node(
            row["node_id"],
            node_type=row["node_type"],
            label=row["label"],
        )

    for _, row in edges.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            edge_type=row["edge_type"],
        )

    return graph


def train_node2vec(
    graph: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 30,
    num_walks: int = 100,
    window: int = 10,
    min_count: int = 1,
    batch_words: int = 128,
    workers: int = 4,
    seed: int = 5,
):
    """
    Train a Node2Vec model on a graph.

    Parameters
    ----------
    graph:
        NetworkX graph.
    dimensions:
        Embedding dimensionality.
    walk_length:
        Length of each random walk.
    num_walks:
        Number of walks started from each node.
    window:
        Context window size for skip-gram training.
    min_count:
        Minimum node occurrence count.
    batch_words:
        Training batch size.
    workers:
        Number of worker threads.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    gensim.models.Word2Vec
        Trained Word2Vec model containing node embeddings.
    """

    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        seed=seed,
    )

    model = node2vec.fit(
        window=window,
        min_count=min_count,
        batch_words=batch_words,
        seed=seed,
    )

    return model


def extract_node_embeddings(
    model,
    nodes: list[str],
    dimensions: int,
) -> pd.DataFrame:
    """
    Extract embeddings for selected nodes.

    Parameters
    ----------
    model:
        Trained Noced2Vec / Word2Vec model.
    nodes:
        Node IDs to extract.
    dimensions:
        Embedding dimensionality.

    Returns
    -------
    pandas.DataFrame
        One row per node, one column per embedding dimesion.
    """

    rows = []

    for node in nodes:
        if node in model.wv:
            vector = model.wv[node]
        else:
            vector = np.zeros(dimensions)
        
        row = {"node_id": node}

        for i, value in enumerate(vector):
            row[f"dim_{i}"] = float(value)

        rows.append(row)

    return pd.DataFrame(rows)


def get_nodes_by_type(
    graph: nx.Graph,
    node_type: str,
) -> list[str]:
    """
    Return all node IDs of a given node_type.
    """
    return [
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == node_type
    ]


def get_node_metadata(graph: nx.Graph) -> pd. DataFrame:
    """
    Convert graph node metadata to a dataFrame
    """
    rows = []

    for node, data in graph.nodes(data=True):
        rows.append(
            {
                "node_id": node,
                "node_type": data.get("node_type", "unknown"),
                "label": data.get("label", node)
            }
        )
    
    return pd.DataFrame(rows)


def save_embeddings(
    embeddings: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save embeddings to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings.to_csv(output_path, index=False)