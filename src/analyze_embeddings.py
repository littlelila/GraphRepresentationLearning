from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.metrics.pairwise import cosine_similarity


def load_disease_embeddings(path: str | Path) -> pd.DataFrame:
    """
    Load disease embeddings from CSV.

    Expected columns:
        node_id, dim_0, dim_1, ..., node_type, label
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    
    embeddings = pd.read_csv(path, dtype={"node_id": str})

    if "node_id" not in embeddings.columns:
        raise ValueError("Embedding file must contain a 'node_id' column.")
    
    return embeddings


def get_embedding_columns(embeddings: pd.DataFrame) -> list[str]:
    """
    Return all embedding dimesion columns.
    """
    embedding_columns = [
        column
        for column in embeddings.columns
        if column.startswith("dim_")
    ]

    if not embedding_columns:
        raise ValueError("No embedding column found. Expected columns named dim_0, dim_1, ...")
    
    return embedding_columns



def compute_nearest_neighbors(
    embeddings: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Compute top_k nearest neigbors using cosine similarity.

    Parameters
    ----------
    embeddings:
        DataFrame with one row per disease and embedding columns dim_*
    top_k:
        Number of nearest neigbors to return per disease.

    Returns
    -------
    pandas.DataFrame
        Neigbor table with columns:
        disease_id, disease_lable, neighbor_id, neighbor_lable, rank, cosine_similarity
    """
    embedding_columns = get_embedding_columns(embeddings)

    disease_ids = embeddings["node_id"].tolist()
    disease_labels = embeddings.get("label", embeddings["node_id"]).tolist()

    label_by_id = dict(zip(disease_ids, disease_labels))

    matrix = embeddings[embedding_columns].to_numpy(dtype=float)

    similarity_matrix = cosine_similarity(matrix)

    rows = []

    for i, disease_id in enumerate(disease_ids):
        similarities = similarity_matrix[i]

        # Sort descending by similarity
        # The first item is the disease itself, so skip it.
        neighbor_indices = np.argsort(-similarities)

        rank = 1

        for j in neighbor_indices:
            if i == j:
                continue

            rows.append(
                {
                    "disease_id": disease_id,
                    "disease_label": label_by_id.get(disease_id, disease_id),
                    "neighbor_id": disease_ids[j],
                    "neighbor_label": label_by_id.get(disease_ids[j], disease_ids[j]),
                    "rank": rank,
                    "cosine_similarity": float(similarities[j])
                }
            )

            rank += 1

            if rank > top_k:
                break

    return pd.DataFrame(rows)


def build_disease_to_hpo_map(hpoa: pd.DataFrame) -> dict[str, set[str]]:
    """
    Build a mapping:
        disease_id -> set of directly annotated HPO terms
    """
    required_columns = {"database_id", "hpo_id"}
    missing = required_columns - set(hpoa.columns)

    if missing:
        raise ValueError(f"Missing required HPOA columns: {missing}")
    
    return (
        hpoa
        .groupby("database_id")["hpo_id"]
        .apply(lambda values: set(values.dropna()))
        .to_dict()
    )


def build_hpo_label_map(hpo) -> dict[str, str]:
    """
    Build a mapping:
        HPO ID -> readable HPO label
    """
    return {
        node_id: data.get("name", node_id)
        for node_id, data in hpo.nodes(data=True)
    }


def explain_neighbor_pairs_with_shared_hpo(
        neighbors: pd.DataFrame,
        hpoa: pd.DataFrame,
        hpo,
        max_terms_per_pair: int = 20,
) -> pd.DataFrame:
    """
    Explain disease-neighbor pairs using directly shared HPO terms.

    Currently simple, TODO: later add ancestor-based explanations.
    """
    disease_to_hpos = build_disease_to_hpo_map(hpoa)
    hpo_labels = build_hpo_label_map(hpo)

    rows = []

    for _, row in neighbors.iterrows():
        disease_id = row["disease_id"]
        neighbor_id = row["neighbor_id"]

        disease_hpos = disease_to_hpos.get(disease_id, set())
        neighbor_hpos = disease_to_hpos.get(neighbor_id, set())

        shared_hpos = sorted(disease_hpos & neighbor_hpos)

        if not shared_hpos:
            rows.append(
                {
                    "disease_id": disease_id,
                    "neighbor_id": neighbor_id,
                    "rank": row["rank"],
                    "cosine_similarity": row["cosine_similarity"],
                    "shared_hpo_id": None,
                    "shared_hpo_label": None,
                    "num_shared_hpo_terms": 0,
                    "disease_hpo_count": len(disease_hpos),
                    "neighbor_hpo_count": len(neighbor_hpos),
                }
            )
            continue

        for hpo_id in shared_hpos[:max_terms_per_pair]:
            rows.append(
                {
                    "disease_id": disease_id,
                    "neighbor_id": neighbor_id,
                    "rank": row["rank"],
                    "cosine_similarity": row["cosine_similarity"],
                    "shared_hpo_id": hpo_id,
                    "shared_hpo_label": hpo_labels.get(hpo_id, hpo_id),
                    "num_shared_hpo_terms": len(shared_hpos),
                    "disease_hpo_count": len(disease_hpos),
                    "neighbor_hpo_count": len(neighbor_hpos),
                }
            )
            
    return pd.DataFrame(rows)


def summarize_neighbor_explainability(
    neighbors: pd.DataFrame,
    hpoa: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one row per disease-neighbor pair with simple explainability metrics.

    Metrics:
        number of shared HPO terms
        Jaccard similarity over direct HPO annotations

    This helps to inspect whether embedding similarity corresponds to explicit phenotype overlap.
    """
    disease_to_hpos = build_disease_to_hpo_map(hpoa)

    rows = []

    for _, row in neighbors.iterrows():
        disease_id = row["disease_id"]
        neighbor_id = row["neighbor_id"]

        disease_hpos = disease_to_hpos.get(disease_id, set())
        neighbor_hpos = disease_to_hpos.get(neighbor_id, set())

        intersection = disease_hpos & neighbor_hpos
        union = disease_hpos | neighbor_hpos

        if union:
            jaccard = len(intersection) / len(union)
        else:
            jaccard = 0.0

        rows.append(
            {
                "disease_id": disease_id,
                "disease_label": row.get("disease_label", disease_id),
                "neighbor_id": neighbor_id,
                "neighbor_label": row.get("neighbor_label", neighbor_id),
                "rank": row["rank"],
                "cosine_similarity": row["cosine_similarity"],
                "num_shared_hpo_terms": len(intersection),
                "jaccard_hpo_similarity": jaccard,
                "disease_hpo_count": len(disease_hpos),
                "neighbor_hpo_count": len(neighbor_hpos),
            }
        )

    return pd.DataFrame(rows)


def compute_umap_projection(
    embeddings: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 5,
) -> pd.DataFrame:
    """
    Project disease embeddings to 2D using UMAP.

    UMAP is used only for visualization.
    The original high-dimensional similarties should still be kept.
    """
    embedding_columns = get_embedding_columns(embeddings)

    matrix = embeddings[embedding_columns].to_numpy(dtype=float)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    coordinates = reducer.fit_transform(matrix)

    projection = pd.DataFrame(
        {
            "node_id": embeddings["node_id"],
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        }
    )

    metadata_columns = [
        column
        for column in ["node_type", "label"]
        if column in embeddings.columns
    ]

    projection = projection.merge(
        embeddings[["node_id", *metadata_columns]],
        on="node_id",
        how="left",
    )

    return projection


def save_dataframe(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save a dataframe to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output_path, index=False)